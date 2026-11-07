"""Parametrised LoRA run (week 8 robustness experiments): seed, model size, checkpoint selection, early stopping, logits."""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import train_esm as T

ap = argparse.ArgumentParser()
ap.add_argument("--model-name", required=True); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--tag", required=True)
ap.add_argument("--select", choices=("f1", "loss"), default="f1"); ap.add_argument("--patience", type=int, default=0)
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch-size", type=int, default=4)
a = ap.parse_args()
OUT = Path("outputs/seeds"); OUT.mkdir(parents=True, exist_ok=True)
LABELS = T.TARGET_COLUMNS; dev = torch.device("cuda")
train, val, test = T.load_splits(Path("data/Swissprot_Train_Validation_dataset.csv"), Path("data/hpa_testset.csv"))
T.set_seed(a.seed); tok = AutoTokenizer.from_pretrained(a.model_name); bb = AutoModel.from_pretrained(a.model_name)
bb = get_peft_model(bb, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query", "key", "value"], bias="none"))
model = T.ProteinFoundationClassifier(bb, bb.config.hidden_size, len(LABELS)).to(dev)
for p in model.classifier.parameters(): p.requires_grad = True
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad); total = sum(p.numel() for p in model.parameters())
mk = lambda df, sh: DataLoader(T.ProteinDataset(df, tok, LABELS, 400), batch_size=a.batch_size, shuffle=sh, num_workers=4)
trl, vall, tel = mk(train, True), mk(val, False), mk(test, False)
pos = train[LABELS].sum().to_numpy(); pw = (len(train) - pos) / (pos + 1e-6)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32, device=dev))
opt = torch.optim.AdamW([{"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad], "lr": 2e-4}, {"params": model.classifier.parameters(), "lr": 1e-3}])
keep = lambda: {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k or "classifier" in k}
torch.cuda.reset_peak_memory_stats(); hist = []; best_val = None; best_state = None; bad = 0
for ep in range(1, a.epochs + 1):
    t0 = time.perf_counter(); tm = T.run_epoch(model, trl, crit, opt, dev, True); vm = T.run_epoch(model, vall, crit, opt, dev, False)
    hist.append({"epoch": ep, "seconds": time.perf_counter() - t0, **{f"train_{k}": v for k, v in tm.items()}, **{f"val_{k}": v for k, v in vm.items()}}); print(hist[-1], flush=True)
    crit_val = vm["macro_f1"] if a.select == "f1" else -vm["loss"]
    if best_val is None or crit_val > best_val: best_val, best_state, bad, best_ep = crit_val, keep(), 0, ep
    else: bad += 1
    if a.patience and bad >= a.patience: print("early stop at epoch", ep, flush=True); break
sd = model.state_dict(); sd.update(best_state); model.load_state_dict(sd)
@torch.no_grad()
def logits(loader):
    model.eval(); return np.vstack([model(b["input_ids"].to(dev), b["attention_mask"].to(dev)).cpu().numpy() for b in loader])
lv, lt = logits(vall), logits(tel); np.savez(OUT / f"{a.tag}_logits.npz", val=lv, test=lt)
from sklearn.metrics import f1_score, accuracy_score
sig = lambda x: 1 / (1 + np.exp(-x)); yv = val[LABELS].to_numpy(int); yt = test[LABELS].to_numpy(int)
m = lambda y, l: {"macro_f1": f1_score(y, sig(l) >= .5, average="macro", zero_division=0), "micro_f1": f1_score(y, sig(l) >= .5, average="micro", zero_division=0), "exact_match": accuracy_score(y, (sig(l) >= .5).astype(int))}
res = {"tag": a.tag, "seed": a.seed, "model": a.model_name, "select": a.select, "patience": a.patience, "best_epoch": best_ep, "epochs_run": len(hist),
       "trainable_parameters": trainable, "total_parameters": total, "peak_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
       "mean_seconds_per_epoch": float(np.mean([h["seconds"] for h in hist])), "val": m(yv, lv), "hpa": m(yt, lt)}
json.dump(res, open(OUT / f"{a.tag}.json", "w"), indent=2); pd.DataFrame(hist).to_csv(OUT / f"{a.tag}_history.csv", index=False); print("RESULT", res, flush=True)
