"""Fine-tune ResNet-18 on the training images of each split and save image
embeddings for all listings.

For each split in SPLITS_JSON, the last residual block (layer4) and a new
regression head (512-64-1, sigmoid) are fine-tuned on the training images to
predict the harm score (L1 loss), with early stopping on the validation
images. The head is then removed, and the 512-d penultimate-layer embedding of
every listing is L2-normalised and saved. Listings without an image get a
zero vector.

Usage: python cnn_embeddings.py DATA_JSON SPLITS_JSON OUT_DIR
Images are read from the paths in the "local_image_path" field of DATA_JSON.
Writes OUT_DIR/embeddings_seed{0,1,2}.npy and OUT_DIR/cnn_log.txt.
"""

import os, sys, json, random, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
DATA, SPLITS, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "cnn_log.txt"), "w", encoding="utf-8")

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

TORCH_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

data = json.load(open(DATA, encoding="utf-8"))
y = np.array([float(r["harm_score"]) for r in data])
N = len(data)
splits = json.load(open(SPLITS, encoding="utf-8"))
log(f"Listings: {N} | device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else " (WARNING: no GPU)"))

t0 = time.time(); IMGS = []
for r in data:
    p = r.get("local_image_path"); img = None
    if isinstance(p, str):
        try:
            img = Image.open(p).convert("RGB"); img.load()
        except Exception:
            img = None
    IMGS.append(img)
HAS_IMG = np.array([im is not None for im in IMGS])
log(f"Images loaded: {HAS_IMG.sum()} / {N} ({time.time()-t0:.0f}s)")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class ImgDS(Dataset):
    def __init__(self, idx, tf): self.idx, self.tf = list(idx), tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        return self.tf(IMGS[i]), torch.tensor(y[i], dtype=torch.float32)

def build_cnn():
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    for p in m.parameters(): p.requires_grad = False
    for n, p in m.named_parameters():
        if "layer4" in n or "fc" in n: p.requires_grad = True
    return m.to(DEVICE)

def train_cnn(tr_idx, va_idx):
    set_seed(TORCH_SEED)
    tr = [i for i in tr_idx if HAS_IMG[i]]; va = [i for i in va_idx if HAS_IMG[i]]
    g = torch.Generator(); g.manual_seed(TORCH_SEED)
    tl = DataLoader(ImgDS(tr, train_tf), batch_size=32, shuffle=True, num_workers=0, generator=g)
    vl = DataLoader(ImgDS(va, eval_tf), batch_size=32, shuffle=False, num_workers=0)
    m = build_cnn()
    opt = torch.optim.Adam([{"params": m.fc.parameters(), "lr": 1e-3}, {"params": m.layer4.parameters(), "lr": 1e-4}])
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.5)
    crit = nn.L1Loss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(50):
        m.train(); s = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(m(xb).squeeze(1), yb); loss.backward(); opt.step()
            s += loss.item() * len(yb)
        m.eval(); vm = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                vm += torch.abs(m(xb.to(DEVICE)).squeeze(1) - yb.to(DEVICE)).sum().item()
        vm /= len(va); sch.step()
        log(f"    epoch {ep+1:02d}  train MAE {s/len(tr):.4f}  val MAE {vm:.4f}")
        if vm < best:
            best, bad = vm, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 10: log(f"    early stop at epoch {ep+1}"); break
    m.load_state_dict(best_state)
    log(f"    CNN trained on {len(tr)} train images, best val MAE {best:.4f}")
    return m

def embed(m):
    m.fc = nn.Identity(); m.eval()
    E = np.zeros((N, 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            if HAS_IMG[i]:
                e = m(eval_tf(IMGS[i]).unsqueeze(0).to(DEVICE))
                E[i] = torch.nn.functional.normalize(e, p=2, dim=1).squeeze().cpu().numpy()
    return E

for s in ["0", "1", "2"]:
    sp = splits[s]
    assert not (set(sp["train"]) & set(sp["test"])) and not (set(sp["val"]) & set(sp["test"]))
    log(f"\n===== split seed {s}: train {len(sp['train'])} / val {len(sp['val'])} / test {len(sp['test'])} =====")
    t0 = time.time()
    cnn = train_cnn(sp["train"], sp["val"])
    E = embed(cnn)
    np.save(os.path.join(OUT, f"embeddings_seed{s}.npy"), E)
    log(f"  saved embeddings_seed{s}.npy ({time.time()-t0:.0f}s)")

log(f"\nDone. Embeddings saved in {OUT}.")
