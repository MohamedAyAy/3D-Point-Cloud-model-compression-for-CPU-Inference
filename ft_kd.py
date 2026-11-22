"""ft_kd.py - Re-entrainement des Students reduits : recette douce (lr bas, plus d'epoques), CE seul vs CE + distillation du Teacher.
Usage : python ft_kd.py --ckpt base_s0_best.pt          (logits Teacher : fichier teacher_logits_*.npy deja calcule)
Choix de l'epoque sur VAL ; TEST lu une seule fois par run, a l'epoque choisie sur VAL.
"""
import argparse, copy, glob, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from teeth_kd import PointNetSeg, patient_split, NUM_CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="base_s0_best.pt")
ap.add_argument("--cache", default="cache_4096.npz")
ap.add_argument("--tl", default="")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--T", type=float, default=4.0)
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()
GPU = "cuda" if torch.cuda.is_available() else "cpu"
C = NUM_CLASSES

d = np.load(a.cache, allow_pickle=True)
P, Y, ids = d["P"], d["Y"], [str(i) for i in d["ids"]]
tr, va, te = patient_split(ids)
pos = {k: i for i, k in enumerate(ids)}
I = lambda L: np.array([pos[i] for i in L])
itr, iva, ite = I(tr), I(va), I(te)

tlf = a.tl or (glob.glob("teacher_logits_*.npy") or [""])[0]
assert tlf, "pas de fichier teacher_logits_*.npy : donne --tl chemin.npy"
TL = np.load(tlf); print("logits Teacher :", tlf, TL.shape, TL.dtype, flush=True)
assert TL.shape[:2] == P.shape[:2], "logits non alignes avec le cache"
def np_miou(logits, y):
    pred = logits.argmax(-1).reshape(-1).astype(np.int64)
    conf = np.bincount(y.reshape(-1).astype(np.int64) * C + pred, minlength=C*C).reshape(C, C)
    tp = np.diag(conf).astype(float); iou = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)
    return float(iou[conf.sum(1) > 0].mean())
print(f"controle : mIoU du Teacher (logits caches)  val={np_miou(TL[iva], Y[iva]):.4f}  test={np_miou(TL[ite], Y[ite]):.4f}", flush=True)

def batches(idx, bs, shuffle=False):
    idx = np.random.permutation(idx) if shuffle else idx
    for s in range(0, len(idx), bs):
        j = np.sort(idx[s:s+bs])
        yield (torch.from_numpy(P[j]).float(), torch.from_numpy(Y[j].astype(np.int64)),
               torch.from_numpy(TL[j].astype(np.float32)))

@torch.no_grad()
def evaluate(m, idx, dev):
    m.eval(); cm = torch.zeros(C, C, dtype=torch.long)
    for x, y, _ in batches(idx, 8):
        p = m(x.to(dev)).argmax(-1).cpu()
        cm += torch.bincount((y*C + p).flatten(), minlength=C*C).view(C, C)
    tp = cm.diag().float(); gt = cm.sum(1).float(); pr = cm.sum(0).float()
    iou = tp / (gt + pr - tp).clamp(min=1)
    return iou[gt > 0].mean().item()

class SlimNet(nn.Module):
    def __init__(self, c3=1024, c4=256):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(3, 64), nn.Linear(64, 128), nn.Linear(128, c3)
        self.h1, self.h2 = nn.Linear(128 + c3, c4), nn.Linear(c4, 128)
        self.out = nn.Linear(128, C); self.r = nn.ReLU()
    def forward(self, x):
        l = self.r(self.l2(self.r(self.l1(x))))
        g = self.r(self.l3(l)).max(1, keepdim=True)[0].expand(-1, l.size(1), -1)
        return self.out(self.r(self.h2(self.r(self.h1(torch.cat([l, g], -1))))))

def fuse(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    lin = nn.Linear(conv.in_channels, conv.out_channels)
    lin.weight.data = conv.weight.data[:, :, 0] * s[:, None]
    b = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    lin.bias.data = (b - bn.running_mean) * s + bn.bias
    return lin

def from_base(base, c3, c4):
    f1, f2, h = base.f1, base.f2, base.head
    l1, l2, l3 = fuse(f1[0], f1[1]), fuse(f1[3], f1[4]), fuse(f2[0], f2[1])
    h1, h2 = fuse(h[0], h[1]), fuse(h[3], h[4])
    i3 = l3.weight.data.abs().sum(1).topk(c3)[1].sort()[0]
    h1w = h1.weight.data[:, torch.cat([torch.arange(128), 128 + i3])]
    i4 = h1w.abs().sum(1).topk(c4)[1].sort()[0]
    m = SlimNet(c3, c4)
    m.l1.load_state_dict(l1.state_dict()); m.l2.load_state_dict(l2.state_dict())
    m.l3.weight.data, m.l3.bias.data = l3.weight.data[i3].clone(), l3.bias.data[i3].clone()
    m.h1.weight.data, m.h1.bias.data = h1w[i4].clone(), h1.bias.data[i4].clone()
    m.h2.weight.data, m.h2.bias.data = h2.weight.data[:, i4].clone(), h2.bias.data.clone()
    m.out.weight.data, m.out.bias.data = h[6].weight.data[:, :, 0].clone(), h[6].bias.data.clone()
    return m

def run(base, c3, c4, mode, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = from_base(base, c3, c4).to(GPU); ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(itr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, state, curve = -1.0, None, []
    for ep in range(a.epochs):
        m.train()
        for x, y, t in batches(itr, a.bs, True):
            x, y, t = x.to(GPU), y.to(GPU), t.to(GPU)
            z = m(x).reshape(-1, C); loss = ce(z, y.reshape(-1))
            if mode == "kd":
                kl = F.kl_div(F.log_softmax(z / a.T, -1), F.softmax(t.reshape(-1, C) / a.T, -1), reduction="batchmean")
                loss = (1 - a.alpha) * loss + a.alpha * a.T * a.T * kl
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        v = evaluate(m, iva, GPU); curve.append(round(v, 4))
        if v > best: best, state = v, copy.deepcopy(m.state_dict())
    m.load_state_dict(state)
    return best, evaluate(m, ite, GPU), curve

base = PointNetSeg(); sd = torch.load(a.ckpt, map_location="cpu")
base.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd); base.eval()

CFG = [(1024, 256), (512, 256), (256, 128), (128, 64)]
out = []; t0 = time.time()
for c3, c4 in CFG:
    for mode in ("ce", "kd"):
        V, T_, curves = [], [], []
        for s in range(a.seeds):
            v, t, c = run(base, c3, c4, mode, s); V.append(v); T_.append(t); curves.append(c)
            print(f"  {c3}/{c4} {mode} seed{s}: val={v:.4f} test={t:.4f}  courbe val={c[::4]}  [{time.time()-t0:.0f}s]", flush=True)
        out.append(dict(cfg=f"{c3}/{c4}", mode=mode, val=(float(np.mean(V)), float(np.std(V))),
                        test=(float(np.mean(T_)), float(np.std(T_))), curves=curves))

print(f"\n=== RESUME (lr={a.lr}, {a.epochs} epoques, {a.seeds} seeds ; recette ancienne : lr 3e-4, 8 ep, CE) ===")
print("Reference ancienne recette (CE) : 1024/256 val 0.5916 | 512/256 0.5452 | 256/128 0.4604 | 128/64 0.4172")
print("Student de base non re-entraine : val 0.6114 test 0.6034")
for r in out:
    print(f"{r['cfg']:>9s} {r['mode']:3s}  val {r['val'][0]:.4f}±{r['val'][1]:.4f}   test {r['test'][0]:.4f}±{r['test'][1]:.4f}")
json.dump(out, open("ft_kd_results.json", "w"), indent=1)