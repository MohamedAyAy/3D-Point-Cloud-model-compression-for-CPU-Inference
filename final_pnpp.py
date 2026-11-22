"""final_pnpp.py - Benchmark FINAL du Student PointNet++ (3 graines) : fp32, fp32 avec BN fusionnes, INT8 dynamique,
puis front de Pareto commun avec les resultats PointNet (final_results_fixed.json ou final_results.json).
Usage : python final_pnpp.py            (attend pnpp_base_s0.pt, pnpp_base_s1.pt, pnpp_base_s2.pt)
Evaluation sur CPU (les modeles quantifies ne tournent que sur CPU). Val et test lus une fois par modele.
Latence : 3 series x (5 echauffements + 50 passages), 1 thread, 4096 points ; on rapporte la mediane des 3 medianes ET l'intervalle min-max.
Sorties : tableau, final_pnpp_results.json, pareto_all.png
"""
import argparse, copy, io, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from teeth_kd import patient_split, NUM_CLASSES as C

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache_4096.npz")
ap.add_argument("--ckpts", default="pnpp_base_s0.pt,pnpp_base_s1.pt,pnpp_base_s2.pt")
ap.add_argument("--pn-json", default="")
a = ap.parse_args()
for f in a.ckpts.split(","): assert os.path.exists(f), f"checkpoint introuvable : {f}"

d = np.load(a.cache, allow_pickle=True)
P, Y, ids = d["P"], d["Y"], [str(i) for i in d["ids"]]
tr, va, te = patient_split(ids)
pos = {k: i for i, k in enumerate(ids)}
iva, ite = (np.array([pos[i] for i in s]) for s in (va, te))

# ------------------------------------------------------------------ Meme architecture que pnpp_kd.py (copie ; head aplati pour la quantification)
def gather(pts, idx):
    B = pts.shape[0]
    return pts[torch.arange(B, device=pts.device).view(B, *([1] * (idx.dim() - 1))), idx]
@torch.no_grad()
def knn(q, ref, k): return torch.cdist(q, ref).topk(k, dim=-1, largest=False)[1]
def interp(xt, xs, fs):
    with torch.no_grad():
        dd, idx = torch.cdist(xt, xs).topk(3, dim=-1, largest=False)
        w = 1.0 / (dd + 1e-8); w = w / w.sum(-1, keepdim=True)
    return (gather(fs, idx) * w.unsqueeze(-1)).sum(2)
class MLP(nn.Module):
    def __init__(self, cin, chans):
        super().__init__()
        cs = [cin] + list(chans)
        self.lins = nn.ModuleList(nn.Linear(cs[i], cs[i + 1]) for i in range(len(chans)))
        self.bns = nn.ModuleList(nn.BatchNorm1d(c) for c in chans)
    def forward(self, x):
        sh = x.shape[:-1]; x = x.reshape(-1, x.shape[-1])
        for lin, bn in zip(self.lins, self.bns): x = F.relu(bn(lin(x)))
        return x.reshape(*sh, -1)
class PN2(nn.Module):
    def __init__(self, nc=C, N=4096, n1=1024, n2=256, k=16, drop=0.2):
        super().__init__()
        self.n1, self.n2, self.k = n1, n2, k
        self.sa1, self.sa2, self.sa3 = MLP(6, [32, 32, 64]), MLP(3 + 64, [64, 64, 128]), MLP(3 + 128, [128, 256, 512])
        self.fp3, self.fp2, self.fp1 = MLP(512 + 128, [256, 256]), MLP(256 + 64, [128, 128]), MLP(128 + 3, [128, 128])
        self.h1, self.hbn, self.drop, self.out = nn.Linear(128, 128), nn.BatchNorm1d(128), nn.Dropout(drop), nn.Linear(128, nc)
        self.register_buffer("cidx_eval", torch.randperm(N, generator=torch.Generator().manual_seed(123))[:n1])
    def forward(self, x):
        B, N, _ = x.shape
        c1 = x[:, self.cidx_eval]
        idx = knn(c1, x, self.k); nb = gather(x, idx)
        f1 = self.sa1(torch.cat([nb - c1.unsqueeze(2), nb], -1)).max(2)[0]
        c2 = c1[:, :self.n2]
        idx2 = knn(c2, c1, self.k)
        f2 = self.sa2(torch.cat([gather(c1, idx2) - c2.unsqueeze(2), gather(f1, idx2)], -1)).max(2)[0]
        g = self.sa3(torch.cat([c2, f2], -1)).max(1)[0]
        u2 = self.fp3(torch.cat([g.unsqueeze(1).expand(-1, self.n2, -1), f2], -1))
        u1 = self.fp2(torch.cat([interp(c1, c2, u2), f1], -1))
        u0 = self.fp1(torch.cat([interp(x, c1, u1), x], -1))
        h = F.relu(self.hbn(self.h1(u0.reshape(-1, 128))))
        return self.out(self.drop(h)).reshape(B, N, -1)

def fuse_lb(lin, bn):                                    # Linear + BatchNorm1d -> un seul Linear
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    f = nn.Linear(lin.in_features, lin.out_features)
    f.weight.data = lin.weight.data * s[:, None]
    f.bias.data = (lin.bias.data - bn.running_mean) * s + bn.bias.data
    return f
def fuse_pn2(S):
    S = copy.deepcopy(S).eval()
    for m in [x for x in S.modules() if isinstance(x, MLP)]:
        m.lins = nn.ModuleList(fuse_lb(l, b) for l, b in zip(m.lins, m.bns))
        m.bns = nn.ModuleList(nn.Identity() for _ in m.bns)
    S.h1 = fuse_lb(S.h1, S.hbn); S.hbn = nn.Identity()
    return S
def q8(m): return torch.ao.quantization.quantize_dynamic(copy.deepcopy(m).cpu().eval(), {nn.Linear}, dtype=torch.qint8)

@torch.no_grad()
def evaluate(m, idx):
    m.eval(); cm = torch.zeros(C, C, dtype=torch.long)
    for s in range(0, len(idx), 8):
        j = np.sort(idx[s:s+8]); x = torch.from_numpy(P[j]).float(); y = torch.from_numpy(Y[j].astype(np.int64))
        cm += torch.bincount((y * C + m(x).argmax(-1)).flatten(), minlength=C * C).view(C, C)
    tp = cm.diag().float(); gt = cm.sum(1).float(); pr = cm.sum(0).float()
    return (tp / (gt + pr - tp).clamp(min=1))[gt > 0].mean().item()

def size_mb(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 1e6
@torch.no_grad()
def latency(m, reps=3, runs=50):
    nt = torch.get_num_threads(); torch.set_num_threads(1)
    x = torch.from_numpy(P[ite[:1]]).float(); m.eval(); meds = []
    for _ in range(reps):
        for _ in range(5): m(x)
        t = []
        for _ in range(runs):
            s = time.perf_counter(); m(x); t.append((time.perf_counter() - s) * 1000)
        meds.append(float(np.median(t)))
    torch.set_num_threads(nt)
    return float(np.median(meds)), min(meds), max(meds)

# ------------------------------------------------------------------ Mesures
MODES = ["fp32", "fp32 BN-fusion", "int8"]
R = {m: dict(val=[], test=[]) for m in MODES}; META = {}; t0 = time.time()
for si, cf in enumerate(a.ckpts.split(",")):
    S = PN2(); S.load_state_dict(torch.load(cf, map_location="cpu")); S.eval()
    models = {"fp32": S, "fp32 BN-fusion": fuse_pn2(S)}; models["int8"] = q8(models["fp32 BN-fusion"])
    for mode, m in models.items():
        R[mode]["val"].append(evaluate(m, iva)); R[mode]["test"].append(evaluate(m, ite))
        if si == 0:
            META[mode] = dict(params=sum(p.numel() for p in m.parameters()), size=size_mb(m), lat=latency(m))
        print(f"{cf} {mode:15s}: val={R[mode]['val'][-1]:.4f} test={R[mode]['test'][-1]:.4f}  [{time.time()-t0:.0f}s]", flush=True)

ms = lambda v: (float(np.mean(v)), float(np.std(v)))
rows = []
for mode in MODES:
    mt = META[mode]
    rows.append(dict(name=f"PointNet++ {mode}", val=ms(R[mode]["val"]), test=ms(R[mode]["test"]), size=mt["size"], lat=mt["lat"][0],
                     lat_range=(mt["lat"][1], mt["lat"][2]), mmac=None, params=mt["params"], fam="pp"))

pj = a.pn_json or next((f for f in ("final_results_fixed.json", "final_results.json") if os.path.exists(f)), "")
if pj:
    for r in json.load(open(pj))["rows"]:
        r["fam"] = "teacher" if "Teacher" in r["name"] else "pn"
        if r["fam"] == "pn": r["name"] = "PointNet " + r["name"]
        r["val"], r["test"] = tuple(r["val"]), tuple(r["test"]); rows.append(r)
    print("\nresultats PointNet lus dans :", pj)
else:
    print("\n(aucun final_results*.json trouve : Pareto PointNet++ seul)")

def dominated(i):
    ri = rows[i]
    return any(j != i and rows[j]["test"][0] >= ri["test"][0] and rows[j]["lat"] <= ri["lat"] and rows[j]["size"] <= ri["size"]
               and (rows[j]["test"][0] > ri["test"][0] or rows[j]["lat"] < ri["lat"] or rows[j]["size"] < ri["size"]) for j in range(len(rows)))

print(f"\n=== RESULTATS (moyenne ± ecart-type sur {len(R['fp32']['test'])} graines ; latence = mediane de 3 series, [min-max]) ===")
print(f"{'modele':46s} {'val':>14s} {'test':>14s} {'Mo':>6s} {'ms':>7s} Pareto")
for i, r in enumerate(rows):
    lr = f" [{r['lat_range'][0]:.0f}-{r['lat_range'][1]:.0f}]" if r.get("lat_range") else ""
    print(f"{r['name']:46s} {r['val'][0]:.4f}±{r['val'][1]:.4f} {r['test'][0]:.4f}±{r['test'][1]:.4f} {r['size']:6.2f} {r['lat']:7.1f}{lr} {'domine' if dominated(i) else 'FRONT'}")
print("('domine' est strict sur (mIoU test, latence, taille) ; la latence varie de ~10-20 % d'un lancement a l'autre -> a lire avec prudence)")
json.dump(dict(rows=rows, raw=R), open("final_pnpp_results.json", "w"), indent=1)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    col = dict(pp="tab:green", pn="tab:blue", teacher="tab:red")
    special = {"PointNet++ fp32": (-52, 9), "PointNet++ fp32 BN-fusion": (7, -3)}
    for r in rows:
        ax.errorbar(r["lat"], r["test"][0], yerr=r["test"][1], fmt="s" if "int8" in r["name"] else "o", color=col[r["fam"]], capsize=3)
        lab = r["name"].replace("PointNet++ ", "PN++ ").replace("PointNet ", "").split(" (")[0]
        off = special.get(r["name"], (-8, -13) if "int8" in r["name"] else (4, 5))
        ax.annotate(lab, (r["lat"], r["test"][0]), fontsize=6.5, xytext=off, textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("latence CPU 1 thread (ms, echelle log)"); ax.set_ylabel("mIoU test"); ax.grid(alpha=.3)
    ax.set_title("Pareto : PointNet++ (vert), PointNet elague (bleu), Teacher (rouge)\ncercle = fp32, carre = INT8", fontsize=9)
    fig.tight_layout(); fig.savefig("pareto_all.png", dpi=200); print("figure : pareto_all.png")
except Exception as e:
    print("figure non generee :", repr(e)[:80])