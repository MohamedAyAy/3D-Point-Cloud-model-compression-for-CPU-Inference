import json, math, time
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from teeth_kd import PointNetSeg, patient_split, NUM_CLASSES as C

dev = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR, SEED = 40, 8, 1e-3, 0
torch.manual_seed(SEED); np.random.seed(SEED)

d = np.load("cache_4096.npz", allow_pickle=True)
P, Y, ids = d["P"], d["Y"], [str(i) for i in d["ids"]]
tr, va, te = patient_split(ids)
pos = {k: i for i, k in enumerate(ids)}
itr = np.array([pos[i] for i in tr]); iva = np.array([pos[i] for i in va])
isub = np.random.RandomState(0).choice(itr, 270, replace=False)   # sous-ensemble du train évalué en mode eval

def batches(idx, bs, shuffle=False):
    idx = np.random.permutation(idx) if shuffle else idx
    for s in range(0, len(idx), bs):
        j = np.sort(idx[s:s+bs])
        yield torch.from_numpy(P[j]).float().to(dev), torch.from_numpy(Y[j].astype(np.int64)).to(dev)

@torch.no_grad()
def evaluate(m, idx):
    m.eval(); cm = torch.zeros(C, C, dtype=torch.long, device=dev); tot, n = 0.0, 0
    for x, y in batches(idx, 8):
        z = m(x)
        tot += F.cross_entropy(z.reshape(-1, C), y.reshape(-1), reduction="sum").item(); n += y.numel()
        cm += torch.bincount((y * C + z.argmax(-1)).flatten(), minlength=C * C).view(C, C)
    tp = cm.diag().float(); gt = cm.sum(1).float(); pr = cm.sum(0).float()
    iou = tp / (gt + pr - tp).clamp(min=1)
    return tot / n, iou[gt > 0].mean().item()

m = PointNetSeg().to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
warm = 3
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm else
        max(0.01, 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, EPOCHS - warm)))))
H = dict(train=[], train_eval=[], val=[], miou=[]); t0 = time.time()
for ep in range(EPOCHS):
    m.train(); tot, nb = 0.0, 0
    for x, y in batches(itr, BS, True):
        loss = F.cross_entropy(m(x).reshape(-1, C), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
    sched.step()
    l_tr_eval, _ = evaluate(m, isub); l_val, miou = evaluate(m, iva)
    H["train"].append(tot / nb); H["train_eval"].append(l_tr_eval); H["val"].append(l_val); H["miou"].append(miou)
    print(f"ep{ep+1}/{EPOCHS} train={tot/nb:.4f} train(eval)={l_tr_eval:.4f} val={l_val:.4f} mIoU={miou:.4f} [{time.time()-t0:.0f}s]", flush=True)

json.dump(H, open("loss_curves.json", "w"))
ep = np.arange(1, EPOCHS + 1); be = int(np.argmin(H["val"])) + 1; bm = int(np.argmax(H["miou"])) + 1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
a1.plot(ep, H["train"], label="train (mode entraînement)"); a1.plot(ep, H["train_eval"], label="train (mode eval, 270 scans)")
a1.plot(ep, H["val"], label="validation"); a1.axvline(be, ls="--", color="gray", lw=1, label=f"min val loss (ép. {be})")
a1.set(xlabel="époque", ylabel="cross-entropy par point", title="Loss"); a1.grid(alpha=.3); a1.legend()
a2.plot(ep, H["miou"], color="tab:green"); a2.axvline(bm, ls="--", color="gray", lw=1, label=f"max mIoU val (ép. {bm})")
a2.set(xlabel="époque", ylabel="mIoU validation", title="mIoU de validation"); a2.grid(alpha=.3); a2.legend()
fig.tight_layout(); fig.savefig("loss_curves.png", dpi=200); print("loss_curves.png", "| min val loss ép.", be, "| max mIoU ép.", bm)