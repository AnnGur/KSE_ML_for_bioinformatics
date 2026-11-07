"""Logits of existing seed-42 models (for bootstrap CIs), memory of the frozen 150M run."""
import json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import train_esm as T
LABELS = T.TARGET_COLUMNS; dev = torch.device("cuda"); OUT = Path("outputs/seeds"); OUT.mkdir(parents=True, exist_ok=True)
train, val, test = T.load_splits(Path("data/Swissprot_Train_Validation_dataset.csv"), Path("data/hpa_testset.csv"))
def loader(df, bs=16): return DataLoader(T.ProteinDataset(df, tok, LABELS, 400), batch_size=bs, num_workers=4)
@torch.no_grad()
def logits(m, df):
    m.eval(); return np.vstack([m(b["input_ids"].to(dev), b["attention_mask"].to(dev)).cpu().numpy() for b in loader(df)])
def build(name, lora):
    global tok; tok = AutoTokenizer.from_pretrained(name); bb = AutoModel.from_pretrained(name)
    if lora: bb = get_peft_model(bb, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query", "key", "value"], bias="none"))
    return T.ProteinFoundationClassifier(bb, bb.config.hidden_size, len(LABELS)).to(dev)
jobs = [("lora35_s42", "facebook/esm2_t12_35M_UR50D", True, "outputs/extra/lora_best.pt"),
        ("lora150_s42", "facebook/esm2_t30_150M_UR50D", True, "outputs/extra/lora150m_best.pt"),
        ("frozen150_s42", "facebook/esm2_t30_150M_UR50D", False, "outputs/esm150m_frozen/best_model.pt")]
for tag, name, lora, ckpt in jobs:
    m = build(name, lora); sd = m.state_dict(); sd.update(torch.load(ckpt, map_location=dev)); m.load_state_dict(sd)
    np.savez(OUT / f"{tag}_logits.npz", val=logits(m, val), test=logits(m, test)); print("logits", tag, flush=True); del m; torch.cuda.empty_cache()
# peak memory, frozen 150M (batch 8, length 400, 30 training steps)
T.set_seed(42); tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D"); bb = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
m = T.ProteinFoundationClassifier(bb, bb.config.hidden_size, len(LABELS)).to(dev); T.configure_trainability(m, 0)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3); crit = nn.BCEWithLogitsLoss()
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); m.train(); m.backbone.eval()
for i, b in enumerate(DataLoader(T.ProteinDataset(train, tok, LABELS, 400), batch_size=8, shuffle=True, num_workers=2)):
    opt.zero_grad(); crit(m(b["input_ids"].to(dev), b["attention_mask"].to(dev)), b["label"].to(dev)).backward(); opt.step()
    if i == 30: break
res = {"frozen150_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "batch_size": 8}; json.dump(res, open(OUT / "frozen150_memory.json", "w")); print("memory", res, flush=True); print("POST_DONE", flush=True)
