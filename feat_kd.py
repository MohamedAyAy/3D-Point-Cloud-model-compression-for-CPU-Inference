"""feat_kd.py - Distillation sur COUCHES INTERMEDIAIRES (DGCNN Teacher -> PointNet Student), en ligne (Teacher ffige, recalcule a chaque lot).
Usage : python feat_kd.py                      (graine 0, 4 variantes ; ~10 min par run)
        python feat_kd.py --seeds 0,1,2 --variants base,hid
Variantes : base (CE seul, temoin avec CETTE boucle) | hid (couche cachee 256) | all (locale+cachee+globale) | all_logit (all + logits)
Regle : une variante n'est retenue que si elle depasse 'base' de >= 0.005 en mIoU VAL (memes graines). Test lu a l'epoque choisie sur VAL.
"""
import argparse, copy, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from teeth_kd import PointNetSeg, DGCNNSeg, edge_feat, patient_split, NUM_CLASSES as C

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="teacher_dgcnn_final.pt")
ap.add_argument("--cache", default="cache_4096.npz")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--lam", type=float, default=1.0)       # poids des termes de features
ap.add_argument("--alpha", type=float, default=0.5)     # (all_logit) part du terme logits
ap.add_argument("--T", type=float, default=4.0)
ap.add_argument("--seeds", default="0")
ap.add_argument("--variants", default="base,hid,all,all_logit")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
TERMS = {"base": [], "hid": ["hid"], "all": ["local", "hid", "glob"], "all_logit": ["local", "hid", "glob", "logit"]}

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

# ---------- Teacher fige : recalcule features et logits ----------
T_ = DGCNNSeg().to(dev)
sd = torch.load(a.teacher, map_location=dev); T_.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
T_.eval()
for p in T_.parameters(): p.requires_grad_(False)

@torch.no_grad()
def teacher_forward(x):                                    # x [B,N,3]
    xt = x.transpose(1, 2)
    x1 = T_.e1(edge_feat(xt, T_.k)).max(3)[0]
    x2 = T_.e2(edge_feat(x1, T_.k)).max(3)[0]
    x3 = T_.e3(edge_feat(x2, T_.k)).max(3)[0]
    l = torch.cat([x1, x2, x3], 1)                         # [B,192,N]  local (avec voisinage)
    g = T_.mid(l).max(2, keepdim=True)[0]                  # [B,1024,1] global
    hid = T_.head[:3](torch.cat([l, g.expand(-1, -1, l.size(2))], 1))   # [B,256,N] couche cachee avant logits
    return dict(local=l, glob=g.squeeze(2), hid=hid, logit=T_.head[3:](hid).transpose(1, 2))

def student_forward(S, x):
    l = S.f1(x.transpose(1, 2))                            # [B,128,N]
    g = S.f2(l).max(2)[0]                                  # [B,1024]
    hid = S.head[:3](torch.cat([l, g.unsqueeze(2).expand(-1, -1, l.size(2))], 1))   # [B,256,N]
    return dict(local=l, glob=g, hid=hid, logit=S.head[3:](hid).transpose(1, 2))

@torch.no_grad()
def evaluate(S, idx):
    S.eval(); cm = torch.zeros(C, C, dtype=torch.long, device=dev)
    for x, y in batches(idx, 8):
        cm += torch.bincount((y * C + student_forward(S, x)["logit"].argmax(-1)).flatten(), minlength=C * C).view(C, C)
    tp = cm.diag().float(); gt = cm.sum(1).float(); pr = cm.sum(0).float()
    return (tp / (gt + pr - tp).clamp(min=1))[gt > 0].mean().item()

def run(variant, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    S = PointNetSeg().to(dev)
    proj = nn.ModuleDict(dict(local=nn.Conv1d(128, 192, 1), hid=nn.Conv1d(256, 256, 1), glob=nn.Linear(1024, 1024))).to(dev)
    opt = torch.optim.AdamW(list(S.parameters()) + list(proj.parameters()), lr=a.lr, weight_decay=1e-4)
    warm = 3
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm else
            max(0.01, 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, a.epochs - warm)))))
    terms = TERMS[variant]; best, state, best_ep, t0 = -1.0, None, 0, time.time()
    for ep in range(a.epochs):
        S.train(); tot = 0.0; nb = 0
        for x, y in batches(itr, a.bs, True):
            o = student_forward(S, x)
            ce = F.cross_entropy(o["logit"].reshape(-1, C), y.reshape(-1))
            if terms:
                t = teacher_forward(x)
                feat = sum((1 - F.cosine_similarity(proj[k](o[k]), t[k], dim=1)).mean() for k in terms if k != "logit")
                if "logit" in terms:
                    kl = F.kl_div(F.log_softmax(o["logit"].reshape(-1, C) / a.T, -1), F.softmax(t["logit"].reshape(-1, C) / a.T, -1), reduction="batchmean")
                    loss = (1 - a.alpha) * ce + a.alpha * a.T ** 2 * kl + a.lam * feat
                else:
                    loss = ce + a.lam * feat
            else:
                loss = ce
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        sched.step(); v = evaluate(S, iva)
        if v > best: best, best_ep, state = v, ep + 1, copy.deepcopy(S.state_dict())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [{variant} s{seed}] ep{ep+1}/{a.epochs} loss={tot/nb:.4f} val mIoU={v:.4f} best={best:.4f}@{best_ep} [{time.time()-t0:.0f}s]", flush=True)
    S.load_state_dict(state)
    return dict(val=best, test=evaluate(S, ite), best_ep=best_ep)

R = {}
for v in a.variants.split(","):
    R[v] = {}
    for s in [int(z) for z in a.seeds.split(",")]:
        R[v][s] = run(v, s); print(f"==> {v} seed{s}: val={R[v][s]['val']:.4f} test={R[v][s]['test']:.4f} (ep {R[v][s]['best_ep']})", flush=True)
json.dump(R, open("feat_kd_results.json", "w"), indent=1)

print(f"\n=== RESUME (lam={a.lam}, {a.epochs} ep, graines {a.seeds}) ===")
for v, r in R.items():
    V = [x["val"] for x in r.values()]; Te = [x["test"] for x in r.values()]
    dv = "" if "base" not in R or v == "base" else "  Δval vs base par graine: " + ", ".join(f"{r[s]['val']-R['base'][s]['val']:+.4f}" for s in r if s in R["base"])
    print(f"{v:10s} val {np.mean(V):.4f}±{np.std(V):.4f}  test {np.mean(Te):.4f}±{np.std(Te):.4f}{dv}")
print("Regle : retenir une variante seulement si Δval >= +0.005 sur les memes graines que 'base'.")