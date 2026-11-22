"""check_softmax_bugs.py - Les bugs 'Softmax Collapse' et 'Softmax Cancellation' nous concernent-ils ?
Usage : python check_softmax_bugs.py --ckpt base_s0_best.pt
"""
import argparse, glob, numpy as np, torch, torch.nn.functional as F
from teeth_kd import PointNetSeg, patient_split, NUM_CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="base_s0_best.pt")
ap.add_argument("--cache", default="cache_4096.npz")
ap.add_argument("--tl", default="")
ap.add_argument("--T", type=float, default=4.0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
C = NUM_CLASSES
CLIFF = {"bf16": 5.5, "fp32": 16.6}

d = np.load(a.cache, allow_pickle=True)
P, Y, ids = d["P"], d["Y"], [str(i) for i in d["ids"]]
tr, va, te = patient_split(ids)
pos = {k: i for i, k in enumerate(ids)}
iva = np.array([pos[i] for i in va]); itr = np.array([pos[i] for i in tr])[:200]   # 200 scans train (marges plus grandes sur train)
tlf = a.tl or glob.glob("teacher_logits_*.npy")[0]
TL = np.load(tlf).astype(np.float32)

S = PointNetSeg()
sd = torch.load(a.ckpt, map_location="cpu")
S.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd); S.to(dev).eval()

@torch.no_grad()
def student_logits(idx):
    out = []
    for s in range(0, len(idx), 8):
        j = idx[s:s+8]
        out.append(S(torch.from_numpy(P[j]).float().to(dev)).cpu())
    return torch.cat(out).reshape(-1, C).float()

def margin(z, y):
    """m = z_t - logsumexp_{j != t} z_j   (en float64)"""
    z = z.double(); y = y[:, None]
    zt = z.gather(1, y)[:, 0]
    zo = z.clone(); zo.scatter_(1, y, float("-inf"))
    return zt - torch.logsumexp(zo, 1)

def report_bug1(name, z, y):
    m = margin(z, y)
    zz = z.clone().requires_grad_(True)
    F.cross_entropy(zz, y, reduction="sum").backward()          # CE standard PyTorch, fp32
    g = zz.grad
    gt = g.gather(1, y[:, None])[:, 0]
    true_gt = -torch.sigmoid(-m)                                  # vrai gradient cible = -e^-m/(1+e^-m)
    collapsed = (gt == 0) & (true_gt.abs() > 0)
    nonsum = (g.sum(1).abs() > 0).float().mean().item()
    
    lost = true_gt.abs()[collapsed].sum() / g.abs().sum()
    print(f"  part du gradient total perdue (|g| arrondis a 0 / |g| total) : {lost.item()*100:.2e} %")
    print(f"\n[BUG 1] {name}  ({len(y)} points)")
    print(f"  marge m : max={m.max():.2f}  p99.9={torch.quantile(m[:1_000_000], 0.999):.2f}  mediane={m.median():.2f}")
    print(f"  m > {CLIFF['bf16']} (falaise bf16) : {(m > CLIFF['bf16']).float().mean()*100:.3f} % des points")
    print(f"  m > {CLIFF['fp32']} (falaise fp32) : {(m > CLIFF['fp32']).float().mean()*100:.5f} % des points")
    print(f"  gradient cible arrondi a 0 alors que le vrai est non nul : {collapsed.float().mean()*100:.5f} %")
    print(f"  lignes dont la somme des gradients != 0 : {nonsum*100:.5f} %")
    return (m > CLIFF["fp32"]).float().mean().item()

def kd_grad(zs, zt, T):
    zs = zs.clone().requires_grad_(True)
    loss = T*T*F.kl_div(F.log_softmax(zs/T, -1), F.softmax(zt/T, -1), reduction="batchmean")
    loss.backward()
    return zs.grad * zs.shape[0]            # gradient par point (on annule le 1/M du batchmean)

def report_bug2(zs, zt, T):
    zs, zt = zs[:200_000], zt[:200_000]
    g_res = kd_grad(zt, zt, T).norm(dim=1)          # zS = zT exactement : le vrai gradient est 0
    g_real = kd_grad(zs, zt, T).norm(dim=1)         # Student reel
    gap = ((zs - zt) / T).abs().max(1).values
    print(f"\n[BUG 2] KD logits, T={T}")
    print(f"  ecart |zS - zT|/T par point : mediane={gap.median():.3f}  min={gap.min():.4f}")
    print(f"  gradient residuel a zS = zT  : moyenne={g_res.mean():.2e}  max={g_res.max():.2e}")
    print(f"  gradient reel (Student)      : moyenne={g_real.mean():.2e}")
    r = (g_res.mean() / g_real.mean()).item()
    print(f"  rapport residuel / reel = {r:.2e}")
    return r

zs_va, y_va = student_logits(iva), torch.from_numpy(Y[iva].astype(np.int64)).reshape(-1)
zs_tr, y_tr = student_logits(itr), torch.from_numpy(Y[itr].astype(np.int64)).reshape(-1)
zt_va = torch.from_numpy(TL[iva]).reshape(-1, C)
zt_tr = torch.from_numpy(TL[itr]).reshape(-1, C)

f = [report_bug1("Student (val)", zs_va, y_va), report_bug1("Student (train, 200 scans)", zs_tr, y_tr),
     report_bug1("Teacher (val)", zt_va, y_va), report_bug1("Teacher (train, 200 scans)", zt_tr, y_tr)]
r = report_bug2(zs_va, zt_va, a.T)

print("\n=== VERDICT ===")
print("Bug 1 :", "CONCERNE (des points depassent la falaise fp32)" if max(f) > 0 else "NE CONCERNE PAS (aucun point au-dessus de la falaise fp32)")
print("Bug 2 :", "CONCERNE (residuel non negligeable)" if r > 1e-3 else "NE CONCERNE PAS (residuel negligeable devant le gradient reel)")