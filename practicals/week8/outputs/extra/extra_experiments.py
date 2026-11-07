"""Extra week 8 experiments run on the lab server (reuses helpers from train_esm.py).

1. peak GPU memory / time per step for frozen, partial, full (batch sizes as in the main runs)
2. checkpoint inference -> per-label F1, threshold sweep, confusion matrices, example proteins (part D)
3. data-efficiency with a frozen ESM-2 35M head (part B)
4. random train/validation split vs the provided partition (part E)
"""
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import train_esm as T

OUT = Path("outputs/extra"); OUT.mkdir(parents=True, exist_ok=True)
LABELS = T.TARGET_COLUMNS; MAXLEN = 400; NAME = "facebook/esm2_t12_35M_UR50D"
dev = torch.device("cuda")
train, val, test = T.load_splits(Path("data/Swissprot_Train_Validation_dataset.csv"), Path("data/hpa_testset.csv"))
tok = AutoTokenizer.from_pretrained(NAME)

def build(n_unfrozen):
    bb = AutoModel.from_pretrained(NAME)
    m = T.ProteinFoundationClassifier(bb, bb.config.hidden_size, len(LABELS)).to(dev)
    T.configure_trainability(m, n_unfrozen); return m

def loader(df, bs, shuffle=False, workers=4):
    return DataLoader(T.ProteinDataset(df, tok, LABELS, MAXLEN), batch_size=bs, shuffle=shuffle, num_workers=workers)

@torch.no_grad()
def logits_of(m, df, bs=32):
    m.eval(); out = []
    for b in loader(df, bs):
        out.append(m(b["input_ids"].to(dev), b["attention_mask"].to(dev)).cpu().numpy())
    return np.vstack(out)

def sig(x): return 1 / (1 + np.exp(-x))
def f1s(y, p, avg): return f1_score(y, p, average=avg, zero_division=0)

# ---- 1. peak memory -------------------------------------------------------
mem = {}
for name, nun, bs in (("frozen", 0, 8), ("partial", 2, 4), ("full", -1, 4)):
    T.set_seed(42); m = build(nun)
    crit = nn.BCEWithLogitsLoss(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m.train(); 
    if nun == 0: m.backbone.eval()
    t0 = None
    for i, b in enumerate(loader(train, bs, True, 2)):
        if i == 5: torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(); loss = crit(m(b["input_ids"].to(dev), b["attention_mask"].to(dev)), b["label"].to(dev)); loss.backward(); opt.step()
        if i == 30: break
    torch.cuda.synchronize()
    mem[name] = {"batch_size": bs, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                 "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "sec_per_step": (time.perf_counter() - t0) / 25}
    print("memory", name, mem[name], flush=True); del m, opt; torch.cuda.empty_cache()
json.dump(mem, open(OUT / "peak_memory.json", "w"), indent=2)

# ---- 2. inference from checkpoints (part D) -------------------------------
yv = val[LABELS].to_numpy(int); yt = test[LABELS].to_numpy(int)
THR = [0.20, 0.35, 0.50, 0.65]
summary = {}
for name, nun in (("frozen", 0), ("partial", 2), ("full", -1)):
    m = build(nun); m.load_state_dict(torch.load(f"outputs/esm35m_{name}/best_model.pt", map_location=dev))
    lv, lt = logits_of(m, val), logits_of(m, test); np.savez(OUT / f"logits_{name}.npz", val=lv, test=lt)
    pv, pt = sig(lv), sig(lt)
    sweep = [{"threshold": t, "val_macro_f1": f1s(yv, pv >= t, "macro"), "hpa_macro_f1": f1s(yt, pt >= t, "macro"),
              "val_micro_f1": f1s(yv, pv >= t, "micro"), "hpa_micro_f1": f1s(yt, pt >= t, "micro")} for t in THR]
    best_t = max(sweep, key=lambda r: r["val_macro_f1"])["threshold"]   # chosen on validation only
    per_label = pd.DataFrame({"label": LABELS, "val_prevalence": yv.mean(0), "hpa_prevalence": yt.mean(0),
        "val_f1": f1s(yv, pv >= 0.5, None), "hpa_f1_at_0.5": f1s(yt, pt >= 0.5, None), f"hpa_f1_at_selected_{best_t}": f1s(yt, pt >= best_t, None),
        "hpa_n_positive": yt.sum(0), "hpa_n_predicted_at_selected": (pt >= best_t).sum(0)})
    per_label.to_csv(OUT / f"per_label_{name}.csv", index=False)
    pd.DataFrame(sweep).to_csv(OUT / f"threshold_sweep_{name}.csv", index=False)
    cm = multilabel_confusion_matrix(yt, (pt >= best_t).astype(int))
    pd.DataFrame([{"label": l, "tn": c[0, 0], "fp": c[0, 1], "fn": c[1, 0], "tp": c[1, 1]} for l, c in zip(LABELS, cm)]).to_csv(OUT / f"confusion_{name}.csv", index=False)
    summary[name] = {"selected_threshold_on_val": best_t, "hpa_macro_f1_selected": f1s(yt, pt >= best_t, "macro"), "hpa_micro_f1_selected": f1s(yt, pt >= best_t, "micro"),
                     "hpa_exact_selected": float(((pt >= best_t).astype(int) == yt).all(1).mean())}
    print("inference", name, summary[name], flush=True)
    if name == "full":
        pred = (pt >= best_t).astype(int); rows = []
        for i in range(len(test)):
            rows.append({"protein_id": test.protein_id[i], "length": len(test.sequence[i]),
                         "true": ", ".join(l for l, v in zip(LABELS, yt[i]) if v), "predicted": ", ".join(l for l, v in zip(LABELS, pred[i]) if v),
                         "max_prob": float(pt[i].max()), "n_fp": int(((pred[i] == 1) & (yt[i] == 0)).sum()), "n_fn": int(((pred[i] == 0) & (yt[i] == 1)).sum())})
        ex = pd.DataFrame(rows); ex.to_csv(OUT / "hpa_predictions_full.csv", index=False)
    del m; torch.cuda.empty_cache()
json.dump(summary, open(OUT / "threshold_summary.json", "w"), indent=2)

# ---- shared trainer for B and E (frozen ESM-2 35M + head, as in the main frozen run) -----
def train_frozen(tr_df, va_df, tag, epochs=5, final_test=False):
    T.set_seed(42); m = build(0)
    pos = tr_df[LABELS].sum().to_numpy(); pw = (len(tr_df) - pos) / (pos + 1e-6)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    trl, val_l = loader(tr_df, 8, True), loader(va_df, 8)
    best, state, hist = -1, None, []
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter(); tm = T.run_epoch(m, trl, crit, opt, dev, True); vm = T.run_epoch(m, val_l, crit, opt, dev, False)
        hist.append({"epoch": ep, "seconds": time.perf_counter() - t0, "train_macro_f1": tm["macro_f1"], "val_macro_f1": vm["macro_f1"], "val_micro_f1": vm["micro_f1"], "val_loss": vm["loss"]})
        if vm["macro_f1"] > best: best, state = vm["macro_f1"], {k: v.clone() for k, v in m.state_dict().items()}
    res = {"tag": tag, "n_train": len(tr_df), "n_val": len(va_df), "best_val_macro_f1": best, "best_epoch": int(np.argmax([h["val_macro_f1"] for h in hist])) + 1, "history": hist}
    if final_test:
        m.load_state_dict(state); r = T.run_epoch(m, loader(test, 8), crit, opt, dev, False); res["hpa"] = r
    print("train", tag, {k: v for k, v in res.items() if k != "history"}, flush=True)
    del m, opt; torch.cuda.empty_cache(); return res

# ---- 3. data efficiency (part B) --------------------------------------------
eff = []
for frac in (0.10, 0.25, 0.50, 1.00):
    sub = train.sample(frac=frac, random_state=42).reset_index(drop=True) if frac < 1 else train
    eff.append(train_frozen(sub, val, f"frac_{frac}", final_test=(frac == 1.0)))
    json.dump(eff, open(OUT / "data_efficiency.json", "w"), indent=2)

# ---- 4. random split (part E) ------------------------------------------------
pool = pd.concat([train, val], ignore_index=True).sample(frac=1.0, random_state=123).reset_index(drop=True)
rv = pool.iloc[:len(val)].reset_index(drop=True); rt = pool.iloc[len(val):].reset_index(drop=True)
prev = pd.DataFrame({"label": LABELS, "provided_train": train[LABELS].mean().values, "provided_val": val[LABELS].mean().values,
                     "random_train": rt[LABELS].mean().values, "random_val": rv[LABELS].mean().values})
prev.to_csv(OUT / "split_prevalence.csv", index=False)
rs = train_frozen(rt, rv, "random_split")
json.dump(rs, open(OUT / "random_split.json", "w"), indent=2)
print("ALL_EXTRA_DONE", flush=True)
