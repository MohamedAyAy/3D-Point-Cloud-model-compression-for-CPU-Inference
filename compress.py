"""compress.py - S4/S5 : fusion BN, pruning, INT8 dynamique, benchmark CPU.
Usage (dans le dossier du projet, a cote de teeth_kd.py et cache_4096.npz):
  python compress.py --ckpt base_s0_best.pt --teacher teacher_dgcnn_final.pt
Fine-tuning + evaluations VAL sur GPU (si dispo), latence/taille mesurees sur CPU 1 thread.
Hyperparametres choisis sur VAL uniquement ; TEST imprime une seule fois a la fin.
"""
import argparse, copy, io, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.utils.prune as prune
from teeth_kd import PointNetSeg, DGCNNSeg, patient_split, NUM_CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="base_s0_best.pt")
ap.add_argument("--teacher", default="teacher_dgcnn_final.pt")
ap.add_argument("--cache", default="cache_4096.npz")
ap.add_argument("--sparsities", default="0.5,0.7,0.9")
ap.add_argument("--ft_epochs", type=int, default=5)
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
GPU = "cuda" if torch.cuda.is_available() else "cpu"
print("device fine-tuning/eval :", GPU, flush=True)

# ---------- donnees (meme split patient que l'entrainement) ----------
d = np.load(a.cache, allow_pickle=True)
P, Y, ids = d["P"], d["Y"], [str(i) for i in d["ids"]]
tr, va, te = patient_split(ids)
pos = {k: i for i, k in enumerate(ids)}
I = lambda L: np.array([pos[i] for i in L])
itr, iva, ite = I(tr), I(va), I(te)

def batches(idx, bs, shuffle=False):
    idx = np.random.permutation(idx) if shuffle else idx
    for s in range(0, len(idx), bs):
        j = np.sort(idx[s:s+bs])
        yield torch.from_numpy(P[j]).float(), torch.from_numpy(Y[j].astype(np.int64))

@torch.no_grad()
def evaluate(m, idx, dev="cpu"):
    m.eval(); cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for x, y in batches(idx, 8):
        p = m(x.to(dev)).argmax(-1).cpu()
        cm += torch.bincount((y*NUM_CLASSES + p).flatten(), minlength=NUM_CLASSES**2).view(NUM_CLASSES, NUM_CLASSES)
    tp = cm.diag().float(); gt = cm.sum(1).float(); pr = cm.sum(0).float()
    iou = tp / (gt + pr - tp).clamp(min=1); present = gt > 0
    return (tp.sum()/cm.sum()).item(), iou[present].mean().item()   # OA, mIoU

# ---------- 1. PointNet -> version "Linear" avec BN fusionnee ----------
def fuse(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    lin = nn.Linear(conv.in_channels, conv.out_channels)
    lin.weight.data = conv.weight.data[:, :, 0] * s[:, None]
    b = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    lin.bias.data = (b - bn.running_mean) * s + bn.bias
    return lin

class FusedPointNet(nn.Module):
    def __init__(self, m):
        super().__init__()
        f1, f2, h = m.f1, m.f2, m.head
        self.l1, self.l2 = fuse(f1[0], f1[1]), fuse(f1[3], f1[4])
        self.l3 = fuse(f2[0], f2[1])
        self.h1, self.h2 = fuse(h[0], h[1]), fuse(h[3], h[4])
        self.out = nn.Linear(128, NUM_CLASSES)
        self.out.weight.data = h[6].weight.data[:, :, 0].clone(); self.out.bias.data = h[6].bias.data.clone()
        self.r = nn.ReLU()
    def forward(self, x):                           # [B,N,3] -> [B,N,C]
        l = self.r(self.l2(self.r(self.l1(x))))
        g = self.r(self.l3(l)).max(1, keepdim=True)[0].expand(-1, l.size(1), -1)
        return self.out(self.r(self.h2(self.r(self.h1(torch.cat([l, g], -1))))))

def load_sd(path):
    sd = torch.load(path, map_location="cpu")
    return sd["model"] if isinstance(sd, dict) and "model" in sd else sd

base = PointNetSeg(); base.load_state_dict(load_sd(a.ckpt)); base.eval()
fused = FusedPointNet(base).eval()
x2 = torch.from_numpy(P[:2]).float()
print("verif fusion (doit etre ~0) :", (base(x2) - fused(x2)).abs().max().item(), flush=True)

# ---------- utilitaires taille / latence ----------
def nnz(m): return sum((p != 0).sum().item() for p in m.parameters()), sum(p.numel() for p in m.parameters())
def size_mb(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell()/1e6
@torch.no_grad()
def latency(m, runs=30):
    torch.set_num_threads(1); x = torch.from_numpy(P[ite[:1]]).float(); m.eval()
    for _ in range(3): m(x)
    t = []
    for _ in range(runs):
        s = time.perf_counter(); m(x); t.append((time.perf_counter()-s)*1000)
    return float(np.median(t))

rows, VAL = [], {}
def record(name, m, dev="cpu", note=""):
    m = m.to(dev); oa, mi = evaluate(m, iva, dev); m = m.cpu()      # eval sur dev, mesures sur CPU
    r = dict(name=name, val_mIoU=mi, val_OA=oa, size_MB=size_mb(m), lat_ms=latency(m),
             params=nnz(m)[1], nonzero=nnz(m)[0], note=note)
    rows.append(r); VAL[name] = mi; print(json.dumps(r), flush=True); return r

record("PointNet float32 (BN fusionnee)", fused)

# ---------- 2. Pruning non structure + fine-tuning (GPU) ----------
def prune_ft(m, sp):
    m = copy.deepcopy(m).to(GPU)
    lins = [(mod, "weight") for mod in m.modules() if isinstance(mod, nn.Linear)]
    prune.global_unstructured(lins, pruning_method=prune.L1Unstructured, amount=sp)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4); ce = nn.CrossEntropyLoss()
    for ep in range(a.ft_epochs):
        m.train()
        for x, y in batches(itr, a.bs, True):
            x, y = x.to(GPU), y.to(GPU)
            opt.zero_grad(); ce(m(x).reshape(-1, NUM_CLASSES), y.reshape(-1)).backward(); opt.step()
        print(f"  sp={sp} ep{ep+1} val mIoU={evaluate(m, iva, GPU)[1]:.4f}", flush=True)
    for mod, n in lins: prune.remove(mod, n)       # fige les zeros dans le poids
    return m.cpu().eval()

pruned = {}
for sp in [float(s) for s in a.sparsities.split(",")]:
    pruned[sp] = prune_ft(fused, sp)
    record(f"PointNet pruning {int(sp*100)}%", pruned[sp], note="zeros non exploites par CPU dense : pas de gain attendu en taille/latence")

# ---------- 3. INT8 dynamique (poids int8, activations quantifiees a la volee) ----------
def q8(m):
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(m).cpu(), {nn.Linear}, dtype=torch.qint8)
record("PointNet INT8 dynamique", q8(fused))
best_sp = max(pruned, key=lambda s: VAL[f"PointNet pruning {int(s*100)}%"])   # choix sur VAL
record(f"PointNet pruning {int(best_sp*100)}% + INT8", q8(pruned[best_sp]))

# ---------- 4. Teacher (reference) ----------
try:
    T = DGCNNSeg(); T.load_state_dict(load_sd(a.teacher)); T.eval()
    record("DGCNN Teacher float32", T, dev=GPU)
except Exception as e:
    print("Teacher non evalue :", repr(e), flush=True)

# ---------- 5. Sortie : tableau + TEST une seule fois ----------
print("\n=== RESULTATS (VAL) ===")
for r in rows:
    print(f"{r['name']:40s} mIoU={r['val_mIoU']:.4f} taille={r['size_MB']:.2f}MB lat={r['lat_ms']:.1f}ms params={r['params']} nnz={r['nonzero']}")
print("\n=== TEST (modeles retenus) ===", flush=True)
test = {}
for n, m in [("fused", fused), (f"pruned{int(best_sp*100)}", pruned[best_sp]),
             ("int8", q8(fused)), (f"pruned{int(best_sp*100)}+int8", q8(pruned[best_sp]))]:
    oa, mi = evaluate(m, ite); test[n] = dict(OA=oa, mIoU=mi); print(n, f"OA={oa:.4f} mIoU={mi:.4f}", flush=True)
json.dump(dict(val=rows, test=test), open("compress_results.json", "w"), indent=1)