"""pnpp_kd.py - Distillation DGCNN (Teacher fige) -> PointNet++ SSG "CPU-friendly" (Student), PyTorch pur (sans PyG ni CUDA custom).
Usage : python pnpp_kd.py --bench                          (params, taille, latence CPU du Student, SANS entrainer)
        python pnpp_kd.py                                  (graine 0, 5 variantes)
        python pnpp_kd.py --seeds 0,1,2 --variants base,all
Variantes : base (CE seul = temoin) | logit (CE + KL logits) | cent (CE + features alignees aux centroides) |
            rel (CE + affinites relationnelles) | all (logit + cent + rel = la loss du guide)
Regle : une variante n'est retenue que si elle depasse 'base' de >= 0.005 en mIoU VAL sur TOUTES les graines. Test lu a l'epoque choisie sur VAL.
Ecarts volontaires avec le guide : echantillonnage aleatoire a la place de FPS ; rel calculee sur les FEATURES (pas les logits) ;
pas de label smoothing (meme CE pour toutes les variantes) ; ReLU (pas LeakyReLU) ; N=4096 (pas 2048) ; 17 classes.
"""
import argparse, copy, io, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from teeth_kd import DGCNNSeg, edge_feat, patient_split, NUM_CLASSES as C
from fixed_ce import fixed_ce, fixed_kl

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="teacher_dgcnn_final.pt")
ap.add_argument("--cache", default="cache_4096.npz")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--alpha", type=float, default=0.5)     # part du terme logits
ap.add_argument("--beta", type=float, default=0.2)      # poids relationnel
ap.add_argument("--gamma", type=float, default=0.2)     # poids centroides
ap.add_argument("--T", type=float, default=4.0)
ap.add_argument("--M", type=int, default=256)           # points pour la matrice d'affinite
ap.add_argument("--seeds", default="0")
ap.add_argument("--variants", default="base,logit,cent,rel,all")
ap.add_argument("--out", default="pnpp_kd_results.json")
ap.add_argument("--bench", action="store_true")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
TERMS = {"base": [], "logit": ["logit"], "cent": ["cent"], "rel": ["rel"], "all": ["logit", "cent", "rel"]}

# ------------------------------------------------------------------ Student PointNet++ SSG
def gather(pts, idx):                                    # pts [B,m,C], idx [B,...] -> [B,...,C]
    B = pts.shape[0]
    return pts[torch.arange(B, device=pts.device).view(B, *([1] * (idx.dim() - 1))), idx]

@torch.no_grad()
def knn(q, ref, k):                                      # q [B,n,3], ref [B,m,3] -> [B,n,k]
    return torch.cdist(q, ref).topk(k, dim=-1, largest=False)[1]

def interp(xt, xs, fs):                                  # interpolation 3-plus-proches-voisins : fs [B,m,C] -> [B,n,C]
    with torch.no_grad():
        d, idx = torch.cdist(xt, xs).topk(3, dim=-1, largest=False)
        w = 1.0 / (d + 1e-8); w = w / w.sum(-1, keepdim=True)
    return (gather(fs, idx) * w.unsqueeze(-1)).sum(2)

class MLP(nn.Module):                                    # (Linear + BN + ReLU) x len(chans), sur la derniere dimension
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
    def forward(self, x, cidx=None):                     # x [B,N,3]
        B, N, _ = x.shape
        if cidx is None:
            cidx = torch.randperm(N, device=x.device)[:self.n1] if self.training else self.cidx_eval
        c1 = x[:, cidx]                                  # centroides SA1 : sous-echantillon aleatoire (remplace FPS)
        idx = knn(c1, x, self.k); nb = gather(x, idx)    # [B,n1,k,3]
        f1 = self.sa1(torch.cat([nb - c1.unsqueeze(2), nb], -1)).max(2)[0]            # [B,n1,64]
        c2 = c1[:, :self.n2]                             # c1 est deja dans un ordre aleatoire
        idx2 = knn(c2, c1, self.k)
        f2 = self.sa2(torch.cat([gather(c1, idx2) - c2.unsqueeze(2), gather(f1, idx2)], -1)).max(2)[0]   # [B,n2,128]
        g = self.sa3(torch.cat([c2, f2], -1)).max(1)[0]                                # [B,512]
        u2 = self.fp3(torch.cat([g.unsqueeze(1).expand(-1, self.n2, -1), f2], -1))     # [B,n2,256]
        u1 = self.fp2(torch.cat([interp(c1, c2, u2), f1], -1))                         # [B,n1,128]
        u0 = self.fp1(torch.cat([interp(x, c1, u1), x], -1))                           # [B,N,128]
        h = F.relu(self.hbn(self.h1(u0).reshape(-1, 128))).reshape(B, N, 128)
        return dict(logit=self.out(self.drop(h)), f1=f1, feat=u0, cidx=cidx)

# ------------------------------------------------------------------ Benchmark sans entrainement
def bench():
    S = PN2().cpu().eval(); nt = torch.get_num_threads(); torch.set_num_threads(1)
    n = sum(p.numel() for p in S.parameters()); b = io.BytesIO(); torch.save(S.state_dict(), b)
    x = torch.randn(1, 4096, 3)
    with torch.no_grad():
        for _ in range(5): S(x)
        t = []
        for _ in range(50):
            s = time.perf_counter(); S(x); t.append((time.perf_counter() - s) * 1000)
    torch.set_num_threads(nt)
    print(f"PointNet++ SSG student : {n:,} params, {b.tell()/1e6:.2f} Mo, latence CPU 1 thread (4096 pts, mediane de 50) = {np.median(t):.1f} ms")
    print("references mesurees : Teacher DGCNN 433.4 ms | PointNet elague 1024/256 = 69.0 ms (51.1 INT8) | 128/64 = 5.3 ms")
if a.bench:
    bench(); raise SystemExit

# ------------------------------------------------------------------ Donnees, Teacher
d = np.load(a.cache, allow_pickle=True)
P, Y, ids = d["P"], d["Y"], [str(i) for i in d["ids"]]
tr, va, te = patient_split(ids)
pos = {k: i for i, k in enumerate(ids)}
itr, iva, ite = (np.array([pos[i] for i in s]) for s in (tr, va, te))

def batches(idx, bs, shuffle=False):
    idx = np.random.permutation(idx) if shuffle else idx
    for s in range(0, len(idx), bs):
        j = np.sort(idx[s:s+bs])
        yield torch.from_numpy(P[j]).float().to(dev), torch.from_numpy(Y[j].astype(np.int64)).to(dev)

T_ = DGCNNSeg().to(dev)
sd = torch.load(a.teacher, map_location=dev); T_.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
T_.eval()
for p in T_.parameters(): p.requires_grad_(False)

@torch.no_grad()
def teacher_forward(x):                                  # x [B,N,3]
    xt = x.transpose(1, 2)
    x1 = T_.e1(edge_feat(xt, T_.k)).max(3)[0]
    x2 = T_.e2(edge_feat(x1, T_.k)).max(3)[0]
    x3 = T_.e3(edge_feat(x2, T_.k)).max(3)[0]            # [B,64,N]  EdgeConv couche 3
    l = torch.cat([x1, x2, x3], 1)
    g = T_.mid(l).max(2, keepdim=True)[0]
    hid = T_.head[:3](torch.cat([l, g.expand(-1, -1, l.size(2))], 1))   # [B,256,N]
    return dict(x3=x3.transpose(1, 2), hid=hid.transpose(1, 2), logit=T_.head[3:](hid).transpose(1, 2))

@torch.no_grad()
def evaluate(S, idx):
    S.eval(); cm = torch.zeros(C, C, dtype=torch.long, device=dev)
    for x, y in batches(idx, 8):
        cm += torch.bincount((y * C + S(x)["logit"].argmax(-1)).flatten(), minlength=C * C).view(C, C)
    tp = cm.diag().float(); gt = cm.sum(1).float(); pr = cm.sum(0).float()
    return (tp / (gt + pr - tp).clamp(min=1))[gt > 0].mean().item()

def affinity(f):                                         # f [B,M,D] -> cosinus deux a deux [B,M,M]
    f = F.normalize(f, dim=-1); return torch.bmm(f, f.transpose(1, 2))

def run(variant, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    S = PN2().to(dev); proj = nn.Linear(64, 64).to(dev)  # projecteur SA1 (64) -> x3 Teacher (64)
    opt = torch.optim.AdamW(list(S.parameters()) + list(proj.parameters()), lr=a.lr, weight_decay=1e-4)
    warm = 3
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm else
            max(0.01, 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, a.epochs - warm)))))
    terms = TERMS[variant]; best, state, best_ep, t0 = -1.0, None, 0, time.time()
    for ep in range(a.epochs):
        S.train(); tot, nb = 0.0, 0
        for x, y in batches(itr, a.bs, True):
            o = S(x); B, N = y.shape
            ce = fixed_ce(o["logit"].reshape(-1, C), y.reshape(-1)); loss = ce
            if terms:
                t = teacher_forward(x)
                if "logit" in terms:
                    kl = fixed_kl(o["logit"].reshape(-1, C), t["logit"].reshape(-1, C), a.T)
                    loss = (1 - a.alpha) * ce + a.alpha * a.T ** 2 * kl
                if "cent" in terms:                      # Teacher x3 aux MEMES indices que les centroides du Student
                    tg = t["x3"][:, o["cidx"]]           # [B,n1,64]
                    loss = loss + a.gamma * (1 - F.cosine_similarity(proj(o["f1"]), tg, dim=-1)).mean()
                if "rel" in terms:                       # affinites entre M points : features finales Student (128) vs Teacher hid (256)
                    sub = torch.randperm(N, device=x.device)[:a.M]
                    loss = loss + a.beta * F.smooth_l1_loss(affinity(o["feat"][:, sub]), affinity(t["hid"][:, sub]))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        sched.step(); v = evaluate(S, iva)
        if v > best: best, best_ep, state = v, ep + 1, copy.deepcopy(S.state_dict())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [{variant} s{seed}] ep{ep+1}/{a.epochs} loss={tot/nb:.4f} val mIoU={v:.4f} best={best:.4f}@{best_ep} [{time.time()-t0:.0f}s]", flush=True)
    S.load_state_dict(state); torch.save(state, f"pnpp_{variant}_s{seed}.pt")      # poids de la meilleure epoque (VAL)
    return dict(val=best, test=evaluate(S, ite), best_ep=best_ep)

bench()
R = {}
for v in a.variants.split(","):
    R[v] = {}
    for s in [int(z) for z in a.seeds.split(",")]:
        R[v][s] = run(v, s); print(f"==> {v} seed{s}: val={R[v][s]['val']:.4f} test={R[v][s]['test']:.4f} (ep {R[v][s]['best_ep']})", flush=True)
        json.dump(R, open(a.out, "w"), indent=1)

print(f"\n=== RESUME PointNet++ (alpha={a.alpha} beta={a.beta} gamma={a.gamma} T={a.T}, {a.epochs} ep, graines {a.seeds}) ===")
for v, r in R.items():
    V = [x["val"] for x in r.values()]; Te = [x["test"] for x in r.values()]
    dv = "" if "base" not in R or v == "base" else "  Δval vs base par graine: " + ", ".join(f"{r[s]['val']-R['base'][s]['val']:+.4f}" for s in r if s in R["base"])
    print(f"{v:6s} val {np.mean(V):.4f}±{np.std(V):.4f}  test {np.mean(Te):.4f}±{np.std(Te):.4f}{dv}")
print("References : PointNet (CE) test 0.602-0.608 | Teacher DGCNN test 0.7007.  Regle : retenir une variante seulement si Δval >= +0.005 sur toutes les graines.")