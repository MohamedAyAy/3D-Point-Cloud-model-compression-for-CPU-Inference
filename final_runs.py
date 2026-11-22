"""final_runs.py - Experience FINALE : pruning structure + re-entrainement CE doux + INT8, sur les 3 checkpoints de base.
Usage : python final_runs.py                    (CE corrigee, defaut)
        python final_runs.py --loss standard    (CE PyTorch standard, pour comparer)
        (attend base_s0_best.pt, base_s1_best.pt, base_s2_best.pt dans le dossier et fixed_ce.py)
Variabilite rapportee = entre les 3 entrainements de base (seeds 0,1,2). Epoque choisie sur VAL ; TEST lu une fois par modele.
Sorties : tableau, front de Pareto (non domines), final_results_<loss>.json, pareto_<loss>.png (si matplotlib).
"""
import argparse, copy, io, json, os, time
import numpy as np, torch, torch.nn as nn
from teeth_kd import PointNetSeg, patient_split, NUM_CLASSES
from fixed_ce import fixed_ce

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache_4096.npz")
ap.add_argument("--bases", default="base_s0_best.pt,base_s1_best.pt,base_s2_best.pt")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--loss", choices=["fixed", "standard"], default="fixed")
a = ap.parse_args()
GPU = "cuda" if torch.cuda.is_available() else "cpu"
C = NUM_CLASSES
for f in a.bases.split(","): assert os.path.exists(f), f"checkpoint introuvable : {f}"

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
def evaluate(m, idx, dev):
    m.eval(); cm = torch.zeros(C, C, dtype=torch.long)
    for x, y in batches(idx, 8):
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

def finetune(m):
    torch.manual_seed(0); np.random.seed(0)
    m = m.to(GPU); ce = fixed_ce if a.loss == "fixed" else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * ((len(itr) + a.bs - 1) // a.bs))
    best, state = -1.0, None
    for ep in range(a.epochs):
        m.train()
        for x, y in batches(itr, a.bs, True):
            x, y = x.to(GPU), y.to(GPU)
            opt.zero_grad(); ce(m(x).reshape(-1, C), y.reshape(-1)).backward(); opt.step(); sched.step()
        v = evaluate(m, iva, GPU)
        if v > best: best, state = v, copy.deepcopy(m.state_dict())
    m.load_state_dict(state)
    return m.cpu().eval()

def q8(m): return torch.ao.quantization.quantize_dynamic(copy.deepcopy(m).cpu(), {nn.Linear}, dtype=torch.qint8)
def size_mb(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell()/1e6
def mmac(m): return sum(l.in_features*l.out_features for l in m.modules() if isinstance(l, nn.Linear))/1e6
@torch.no_grad()
def latency(m, runs=50):
    torch.set_num_threads(1); x = torch.from_numpy(P[ite[:1]]).float(); m.eval()
    for _ in range(5): m(x)
    t = []
    for _ in range(runs):
        s = time.perf_counter(); m(x); t.append((time.perf_counter()-s)*1000)
    return float(np.median(t))

CFG = [(1024, 256), (512, 256), (256, 128), (128, 64)]
R = {c: dict(val=[], test=[], val8=[], test8=[]) for c in CFG}
META = {}; t0 = time.time()
for bi, bf in enumerate(a.bases.split(",")):
    base = PointNetSeg(); sd = torch.load(bf, map_location="cpu")
    base.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd); base.eval()
    for c in CFG:
        m = finetune(from_base(base, *c)); q = q8(m)
        R[c]["val"].append(evaluate(m, iva, "cpu")); R[c]["test"].append(evaluate(m, ite, "cpu"))
        R[c]["val8"].append(evaluate(q, iva, "cpu")); R[c]["test8"].append(evaluate(q, ite, "cpu"))
        if bi == 0:
            META[c] = dict(mmac=mmac(m), params=sum(p.numel() for p in m.parameters()), size=size_mb(m), size8=size_mb(q),
                           lat=latency(m), lat8=latency(q))
        print(f"base {bi} {c}: val={R[c]['val'][-1]:.4f} test={R[c]['test'][-1]:.4f} test_int8={R[c]['test8'][-1]:.4f}  [{time.time()-t0:.0f}s]", flush=True)

ms = lambda v: (float(np.mean(v)), float(np.std(v)))
rows = []
for c in CFG:
    r = R[c]; mt = META[c]
    for mode, tk, vk, sz, lt in (("fp32", "test", "val", "size", "lat"), ("int8", "test8", "val8", "size8", "lat8")):
        rows.append(dict(name=f"{c[0]}/{c[1]} {mode}", val=ms(r[vk]), test=ms(r[tk]), size=mt[sz], lat=mt[lt], mmac=mt["mmac"], params=mt["params"]))
rows.insert(0, dict(name="Teacher DGCNN (mesure anterieure, 1 seed)", val=(0.6930, 0.0), test=(0.7007, 0.0), size=2.28, lat=433.4, mmac=None, params=562833))

def dominated(i):
    ri = rows[i]
    return any(j != i and rows[j]["test"][0] >= ri["test"][0] and rows[j]["lat"] <= ri["lat"] and rows[j]["size"] <= ri["size"]
               and (rows[j]["test"][0] > ri["test"][0] or rows[j]["lat"] < ri["lat"] or rows[j]["size"] < ri["size"]) for j in range(len(rows)))

print(f"\n=== RESULTATS FINAUX (moyenne ± ecart-type sur {len(a.bases.split(','))} entrainements de base ; recette CE {a.loss} lr={a.lr}, {a.epochs} ep) ===")
print(f"{'modele':44s} {'val':>14s} {'test':>14s} {'Mo':>6s} {'ms':>7s} {'MMAC':>6s} Pareto")
for i, r in enumerate(rows):
    print(f"{r['name']:44s} {r['val'][0]:.4f}±{r['val'][1]:.4f} {r['test'][0]:.4f}±{r['test'][1]:.4f} {r['size']:6.2f} {r['lat']:7.1f} "
          f"{(r['mmac'] if r['mmac'] else 0):6.3f} {'domine' if dominated(i) else 'FRONT'}")
print("(latence : mediane de 50 passages, 1 thread, une seule serie de mesures -> ~±10% ; 'domine' est strict, donc sensible a ce bruit)")
json.dump(dict(rows=rows, raw={f"{k[0]}/{k[1]}": v for k, v in R.items()}), open(f"final_results_{a.loss}.json", "w"), indent=1)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, r in enumerate(rows):
        ax.errorbar(r["lat"], r["test"][0], yerr=r["test"][1], fmt="o" if "fp32" in r["name"] or "Teacher" in r["name"] else "s",
                    color="tab:red" if "Teacher" in r["name"] else "tab:blue", capsize=3)
        ax.annotate(r["name"].split(" (")[0].replace(" fp32", "").replace(" int8", " i8"), (r["lat"], r["test"][0]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("latence CPU 1 thread (ms, echelle log)"); ax.set_ylabel("mIoU test"); ax.grid(alpha=.3)
    ax.set_title("Compromis precision / latence (cercle = fp32, carre = INT8)"); fig.tight_layout(); fig.savefig(f"pareto_{a.loss}.png", dpi=200)
    print(f"figure : pareto_{a.loss}.png")
except Exception as e:
    print("figure non generee :", repr(e)[:80])