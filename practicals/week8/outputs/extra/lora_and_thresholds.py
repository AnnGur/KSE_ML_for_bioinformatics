"""Per-label thresholds (validation-only) and LoRA fine-tuning of ESM-2 35M (week 8 extras)."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import train_esm as T

OUT = Path("outputs/extra"); LABELS = T.TARGET_COLUMNS
train, val, test = T.load_splits(Path("data/Swissprot_Train_Validation_dataset.csv"), Path("data/hpa_testset.csv"))
yv = val[LABELS].to_numpy(int); yt = test[LABELS].to_numpy(int)
sig = lambda x: 1 / (1 + np.exp(-x))
f1 = lambda y, p, avg: f1_score(y, p, average=avg, zero_division=0)

# ---- per-label thresholds, chosen on validation only (grid 0.05 .. 0.95) -----
grid = np.round(np.arange(0.05, 0.96, 0.05), 2); res = {}
for name in ("frozen", "partial", "full"):
    z = np.load(OUT / f"logits_{name}.npz"); pv, pt = sig(z["val"]), sig(z["test"])
    best = [float(grid[int(np.argmax([f1_score(yv[:, j], pv[:, j] >= t, zero_division=0) for t in grid]))]) for j in range(len(LABELS))]
    thr = np.array(best); g = {t: f1(yv, pv >= t, "macro") for t in (0.5, 0.65)}
    best_global = max(g, key=g.get)
    res[name] = {"per_label_thresholds": dict(zip(LABELS, best)),
        "val_macro_f1_per_label_thr": f1(yv, pv >= thr, "macro"), "hpa_macro_f1_per_label_thr": f1(yt, pt >= thr, "macro"),
        "hpa_micro_f1_per_label_thr": f1(yt, pt >= thr, "micro"), "hpa_exact_per_label_thr": float(((pt >= thr).astype(int) == yt).all(1).mean()),
        "val_macro_f1_global_0.65": g[0.65], "hpa_macro_f1_global_0.65": f1(yt, pt >= 0.65, "macro"),
        "hpa_per_label_f1": dict(zip(LABELS, f1(yt, pt >= thr, None).tolist()))}
    print("perlabel", name, {k: v for k, v in res[name].items() if k != "hpa_per_label_f1"}, flush=True)
json.dump(res, open(OUT / "per_label_thresholds.json", "w"), indent=2)

# ---- LoRA ---------------------------------------------------------------------
from peft import LoraConfig, get_peft_model
dev = torch.device("cuda"); NAME = "facebook/esm2_t12_35M_UR50D"; BS = 4; EPOCHS = 5; MAXLEN = 400
T.set_seed(42); tok = AutoTokenizer.from_pretrained(NAME); bb = AutoModel.from_pretrained(NAME)
cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query", "key", "value"], bias="none")
bb = get_peft_model(bb, cfg)
model = T.ProteinFoundationClassifier(bb, bb.config.hidden_size, len(LABELS)).to(dev)
for p in model.classifier.parameters(): p.requires_grad = True
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad); total = sum(p.numel() for p in model.parameters())
print(f"LoRA trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)
mk = lambda df, sh: DataLoader(T.ProteinDataset(df, tok, LABELS, MAXLEN), batch_size=BS, shuffle=sh, num_workers=4)
trl, vall, tel = mk(train, True), mk(val, False), mk(test, False)
pos = train[LABELS].sum().to_numpy(); pw = (len(train) - pos) / (pos + 1e-6)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32, device=dev))
lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
opt = torch.optim.AdamW([{"params": lora_params, "lr": 2e-4}, {"params": model.classifier.parameters(), "lr": 1e-3}])
torch.cuda.reset_peak_memory_stats(); best, hist = -1, []
for ep in range(1, EPOCHS + 1):
    t0 = time.perf_counter(); tm = T.run_epoch(model, trl, crit, opt, dev, True); vm = T.run_epoch(model, vall, crit, opt, dev, False)
    row = {"epoch": ep, "seconds": time.perf_counter() - t0, **{f"train_{k}": v for k, v in tm.items()}, **{f"val_{k}": v for k, v in vm.items()}}
    hist.append(row); print(row, flush=True)
    if vm["macro_f1"] > best: best = vm["macro_f1"]; torch.save({k: v for k, v in model.state_dict().items() if "lora_" in k or "classifier" in k}, OUT / "lora_best.pt")
sd = model.state_dict(); sd.update(torch.load(OUT / "lora_best.pt", map_location=dev)); model.load_state_dict(sd)
test_m = T.run_epoch(model, tel, crit, opt, dev, False)
pd.DataFrame(hist).to_csv(OUT / "lora_history.csv", index=False)
out = {"trainable_parameters": trainable, "total_parameters": total, "peak_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
       "mean_seconds_per_epoch": float(np.mean([h["seconds"] for h in hist])), "best_val_macro_f1": best,
       "best_epoch": int(np.argmax([h["val_macro_f1"] for h in hist])) + 1, "hpa": test_m, "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "targets": ["query", "key", "value"], "lr_lora": 2e-4, "lr_head": 1e-3},
       "batch_size": BS, "max_length": MAXLEN, "epochs": EPOCHS, "seed": 42}
json.dump(out, open(OUT / "lora_results.json", "w"), indent=2); print("LORA_RESULT", out, flush=True); print("ALL_LORA_DONE", flush=True)
