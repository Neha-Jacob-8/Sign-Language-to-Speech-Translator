# imagetraining.py  — scratch CNN, macOS/MPS safe, progress prints, excludes Indian_TEST
import os, json, random, time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# Use faster matmul on Apple Silicon
torch.set_float32_matmul_precision("high")

# ------------------------- Small CNN (scratch) -------------------------
class SmallSignCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        def blk(i,o):
            return nn.Sequential(
                nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.MaxPool2d(2)
            )
        self.f = nn.Sequential(
            blk(3,32), blk(32,64), blk(64,128), blk(128,256),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  #I  -> [B,256,1,1]
        )
        self.h = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256,256), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256,n_classes)
        )
    def forward(self,x):
        x = self.f(x)
        x = x.reshape(x.size(0), -1)    # safe for non-contiguous tensors (MPS)
        return self.h(x)

# ------------------------- Loss & Metrics -------------------------
class SmoothCE(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps
    def forward(self, logits, y):
        logp = F.log_softmax(logits, dim=-1)
        base = F.nll_loss(logp, y, reduction='none')
        smooth = -logp.mean(dim=-1)
        return (1 - self.eps) * base + self.eps * smooth

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

# ------------------------- Dataset helpers -------------------------
# ------------------------- Dataset & Loader Utilities -------------------------

# allowed image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _is_img(p: Path) -> bool:
    # check if path is a valid image file (exists and has correct extension)
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def build_train_imagefolder_excluding_test(root: Path, transform):
    """Build dataset from root directory but skip any folder named 'Indian_TEST'."""
    
    # collect all class folders except 'Indian_TEST'
    class_dirs = [d for d in root.iterdir() if d.is_dir() and d.name != "Indian_TEST"]
    # sort class folders alphabetically to keep consistent label ordering
    class_dirs = sorted(class_dirs, key=lambda p: p.name)
    # assign each class folder a numeric label (0,1,2,…)
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

    samples = []  # list to hold all (image_path, label_index) pairs
    for d in class_dirs:  # loop through each class folder
        for f in d.iterdir():  # loop through each file in the folder
            if _is_img(f):  # include only valid image files
                samples.append((str(f), class_to_idx[d.name]))  # store image path and its label

    # create a standard ImageFolder so transforms work normally
    ds = datasets.ImageFolder(str(root), transform=transform)
    # override default samples with our custom filtered list
    ds.samples = samples
    # store only the class indices (labels) for easy access
    ds.targets = [s[1] for s in samples]
    # update the class-to-index mapping
    ds.class_to_idx = class_to_idx
    # keep the ordered list of class names
    ds.classes = list(class_to_idx.keys())
    return ds  # return the prepared dataset


def make_weighted_sampler(idxs, targets):
    # extract target labels corresponding to given indices
    sub_t = [targets[i] for i in idxs]
    # count how many samples exist per class
    cnt = Counter(sub_t); total = len(sub_t)
    # compute inverse frequency weights to balance rare classes
    weights = [1.0 / max(1e-6, cnt[targets[i]] / total) for i in idxs]
    # create a WeightedRandomSampler using these weights
    return WeightedRandomSampler(weights, num_samples=len(idxs), replacement=True)


def make_loader(root, idxs, transform, batch_size, num_workers, train, targets):
    # build the base dataset (excluding Indian_TEST)
    ds = build_train_imagefolder_excluding_test(root, transform=transform)
    # select only the subset of samples specified by idxs (train/val split)
    sub = Subset(ds, idxs)
    if train:
        # for training, create a weighted sampler for balanced class distribution
        sampler = make_weighted_sampler(idxs, targets)
        # return DataLoader with sampler (no shuffle needed)
        return DataLoader(sub, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, persistent_workers=(num_workers > 0))
    else:
        # for validation, no sampler — just load data in order
        return DataLoader(sub, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, persistent_workers=(num_workers > 0))

# ------------------------- Train / Eval -------------------------
def train_one_epoch(model, loader, opt, scaler, crit, device, accum_steps, step_log_every=50):
    model.train(); seen=0; loss_sum=0.0; acc_sum=0.0
    t0 = time.time()
    for step,(xb,yb) in enumerate(loader,1):
        xb = xb.to(device); yb = yb.to(device)
        ctx = torch.amp.autocast(device_type=device, dtype=torch.float16) if device!="cpu" else torch.enable_grad()
        with ctx:
            out = model(xb)
            loss = crit(out, yb).mean() / accum_steps
        if scaler: scaler.scale(loss).backward()
        else: loss.backward()
        if step % accum_steps == 0:
            if scaler: scaler.step(opt); scaler.update()
            else: opt.step()
            opt.zero_grad(set_to_none=True)
        bs = xb.size(0); seen += bs
        loss_sum += loss.item() * bs
        acc_sum  += accuracy(out.detach(), yb) * bs
        if step % step_log_every == 0:
            elapsed = time.time()-t0
            print(f"    step {step}/{len(loader)} | {elapsed:.1f}s")
    return loss_sum/seen, acc_sum/seen

@torch.no_grad()
def evaluate(model, loader, crit, device, return_preds=False):
    model.eval(); seen=0; loss_sum=0.0; acc_sum=0.0; yp=[]; yt=[]
    for xb,yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        ctx = torch.amp.autocast(device_type=device, dtype=torch.float16) if device!="cpu" else torch.enable_grad()
        with ctx:
            out = model(xb)
            loss = crit(out, yb).mean()
        bs = xb.size(0); seen += bs
        loss_sum += loss.item() * bs
        acc_sum  += accuracy(out, yb) * bs
        if return_preds:
            yp += out.argmax(1).cpu().tolist()
            yt += yb.cpu().tolist()
    if return_preds:
        return (loss_sum/seen, acc_sum/seen, yt, yp)
    return loss_sum/seen, acc_sum/seen

def save_if_good(model, rep_id, classes, cfg, train_acc, val_acc, report, cm):
    gap = train_acc - val_acc
    good = (val_acc >= cfg["save_min_val_acc"]) and (gap <= cfg["max_gen_gap"])
    Path("checkpoints").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)
    with open(f"reports/{rep_id}.json","w") as f:
        json.dump({
            "id": rep_id,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "gap": gap,
            "thresholds": {"min_val": cfg["save_min_val_acc"], "max_gap": cfg["max_gen_gap"]},
            "classification_report": report,
            "confusion_matrix": cm
        }, f, indent=2)
    if good:
        torch.save({
            "model": model.state_dict(),
            "classes": classes,
            "cfg": cfg,
            "metrics": {"train_acc": train_acc, "val_acc": val_acc, "gap": gap}
        }, f"checkpoints/{rep_id}.pt")
        print(f"✓ saved {rep_id} (val={val_acc*100:.2f}%, gap={gap:.3f})")
    else:
        print(f"✗ not saved {rep_id} (val={val_acc*100:.2f}%, gap={gap:.3f})")

# ------------------------- Main -------------------------
def main():
    CFG = json.load(open("config.json"))
    # safe defaults for macOS
    CFG.setdefault("num_workers", 0)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(CFG["img_size"], scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.15 if CFG.get("flip", False) else 0.0),
        transforms.ColorJitter(0.15, 0.15, 0.1, 0.02),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(CFG["img_size"]*1.14)),
        transforms.CenterCrop(CFG["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # dataset (exclude Indian_TEST if present)
    root = Path(CFG["data_dir"])
    full_set = build_train_imagefolder_excluding_test(root, transform=train_tf)
    classes = full_set.classes
    targets = [y for _, y in full_set.samples]
    num_classes = len(classes)
    print("classes:", classes)
    print("num_classes:", num_classes, "| total images:", len(full_set.samples))
    print("Starting cross-validation...")
        # ---- resume support: detect completed splits from reports/ ----
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    # Collect prefixes like "R1_F3" from filenames "R1_F3_val0.987.json"
    done_prefixes = {
        p.stem.split("_val")[0]
        for p in reports_dir.glob("R*_F*.json")
    }
    if done_prefixes:
        print("Resuming… completed splits detected:", sorted(done_prefixes))

    rskf = RepeatedStratifiedKFold(n_splits=CFG["n_folds"], n_repeats=CFG["repeats"], random_state=CFG["seed"])
    crit = SmoothCE(CFG["label_smoothing"])

    mean_list = []
    split_counter = 0

    for split_idx, (train_idx, val_idx) in enumerate(rskf.split(range(len(full_set.samples)), targets), 1):
        split_counter += 1
        print(f"\n=== split {split_counter:02d} / (folds={CFG['n_folds']} × repeats={CFG['repeats']}) ===")
        rep_prefix = f"R{(split_counter-1)//CFG['n_folds']+1}_F{(split_counter-1)%CFG['n_folds']+1}"
        if rep_prefix in done_prefixes:
            print(f"⏩ skipping {rep_prefix} (report exists)")
            continue

        tr_loader = make_loader(root, train_idx, train_tf, CFG["batch_size"], CFG["num_workers"], True, targets)
        va_loader = make_loader(root, val_idx,   eval_tf,  CFG["batch_size"], CFG["num_workers"], False, targets)

        model = SmallSignCNN(num_classes).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["base_lr"], weight_decay=CFG["weight_decay"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
        scaler = torch.amp.GradScaler(device=device) if device!="cpu" else None

        best_val = 0.0; best_state=None; noimp=0
        for ep in range(1, CFG["epochs"]+1):
            print(f"[split {split_counter:02d}] epoch {ep:02d}/{CFG['epochs']}")
            t0 = time.time()
            tl, ta = train_one_epoch(model, tr_loader, opt, scaler, crit, device, CFG["accum_steps"], step_log_every=50)
            vl, va = evaluate(model, va_loader, crit, device)
            sch.step()
            print(f"  -> train_loss={tl:.4f} acc={ta*100:.2f}% | val_loss={vl:.4f} acc={va*100:.2f}% | {time.time()-t0:.1f}s")

            if va > best_val:
                best_val = va
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                noimp = 0
            else:
                noimp += 1
                if noimp >= CFG["early_stop_patience"]:
                    print("  early stop")
                    break

        if best_state:
            model.load_state_dict(best_state)
            trL, trA = evaluate(model, tr_loader, crit, device)
            vL, vA, yt, yp = evaluate(model, va_loader, crit, device, return_preds=True)
            rep = classification_report(yt, yp, target_names=classes, output_dict=True, digits=4)
            cm  = confusion_matrix(yt, yp).tolist()
            rep_id = f"R{(split_counter-1)//CFG['n_folds']+1}_F{(split_counter-1)%CFG['n_folds']+1}_val{vA:.3f}"
            save_if_good(model, rep_id, classes, CFG, trA, vA, rep, cm)
            mean_list.append(vA)

    if mean_list:
        print("\nmean_val_acc:", round((sum(mean_list)/len(mean_list))*100, 2), "%")
    else:
        print("\nmean_val_acc: 0.00 % (no splits processed)")

if __name__ == "__main__":
    main()
