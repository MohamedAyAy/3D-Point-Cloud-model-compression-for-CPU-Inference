"""structured_prune.py - pruning STRUCTURE du Student PointNet (suppression de neurones entiers).
Usage : python structured_prune.py --ckpt base_s0_best.pt
Compare, sur 3 seeds de fine-tuning : temoin (0% supprime), et 3 largeurs reduites, en fp32 et INT8.
Mesure : mIoU val/test, MMAC/point, taille, latence CPU 1 thread, surcout RAM de l'inference.
"""
import argparse, copy, io, json, os, resource, subprocess, sys, tempfile, time
import numpy as np, torch, torch.nn as nn
from teeth_kd import PointNetSeg, patient_split, NUM_CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="base_s0_best.pt")
ap.add_argument("--cache", default="cache_4096.npz")
ap.add_argument("--ft_epochs", type=int, default=8)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--ram", default="")            # usage interne (sous-processus de mesure RAM)
a = ap.parse_args()

# ---------- modele a largeurs variables (Linear, BN deja fusionnee) ----------
class SlimNet(nn.Module):
    def __init__(self, c3=1024, c4=256):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(3, 64), nn.Linear(64, 128), nn.Linear(128, c3)
        self.h1, self.h2 = nn.Linear(128 + c3, c4), nn.Linear(c4, 128)
        self.out = nn.Linear(128, NUM_CLASSES); self.r = nn.ReLU()
    def forward(self, x):                           # [B,N,3] -> [B,N,C]
        l = self.r(self.l2(self.r(self.l1(x))))
        g = self.r(self.l3(l)).max(1, keepdim=True)[0].expand(-1, l.size(1), -1)
        return self.out(self.r(self.h2(self.r(self.h1(torch.cat([l, g], -1))))))

# ---------- sous-processus : mesure du surcout memoire d'une inference ----------
if a.ram:
    torch.set_num_threads(1)
    m = torch.load(a.ram, weights_only=False).eval(); x = torch.randn(1, 4096, 3)
    r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # Ko sous Linux
    with torch.no_grad(): m(x)
    r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(process_MB=r1/1024, forward_delta_MB=(r1-r0)/1024))); sys.exit()

GPU = "cuda" if torch.cuda.is_available() else "cpu"
print("device :", GPU, flush=True)

# ---------- donnees ----------
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
    iou = tp / (gt + pr - tp).clamp(min=1)
    return (tp.sum()/cm.sum()).item(), iou[gt > 0].mean().item()      # OA, mIoU

# ---------- fusion BN + decoupe des neurones les moins importants ----------
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
    i3 = l3.weight.data.abs().sum(1).topk(c3)[1].sort()[0]            # c3 canaux de l3 les plus "forts" (norme L1)
    cols = torch.cat([torch.arange(128), 128 + i3])                   # entrees de h1 : 128 locaux + canaux globaux gardes
    h1w = h1.weight.data[:, cols]
    i4 = h1w.abs().sum(1).topk(c4)[1].sort()[0]                       # c4 neurones de h1 les plus "forts"
    m = SlimNet(c3, c4)
    m.l1.load_state_dict(l1.state_dict()); m.l2.load_state_dict(l2.state_dict())
    m.l3.weight.data, m.l3.bias.data = l3.weight.data[i3].clone(), l3.bias.data[i3].clone()
    m.h1.weight.data, m.h1.bias.data = h1w[i4].clone(), h1.bias.data[i4].clone()
    m.h2.weight.data, m.h2.bias.data = h2.weight.data[:, i4].clone(), h2.bias.data.clone()
    m.out.weight.data, m.out.bias.data = h[6].weight.data[:, :, 0].clone(), h[6].bias.data.clone()
    return m

def finetune(m, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = m.to(GPU); ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = a.ft_epochs * ((len(itr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for ep in range(a.ft_epochs):
        m.train()
        for x, y in batches(itr, a.bs, True):
            x, y = x.to(GPU), y.to(GPU)
            opt.zero_grad(); ce(m(x).reshape(-1, NUM_CLASSES), y.reshape(-1)).backward(); opt.step(); sched.step()
        print(f"    seed{seed} ep{ep+1} val mIoU={evaluate(m, iva, GPU)[1]:.4f}", flush=True)
    return m

# ---------- mesures ----------
def size_mb(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell()/1e6
def mmac_per_point(m):
    return sum(l.in_features*l.out_features for l in m.modules() if isinstance(l, nn.Linear))/1e6
@torch.no_grad()
def latency(m, runs=30):
    torch.set_num_threads(1); x = torch.from_numpy(P[ite[:1]]).float(); m.eval()
    for _ in range(3): m(x)
    t = []
    for _ in range(runs):
        s = time.perf_counter(); m(x); t.append((time.perf_counter()-s)*1000)
    return float(np.median(t))
def ram(m):
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False); f.close()
    try:
        torch.save(m, f.name)
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--ram", f.name],
                             capture_output=True, text=True, timeout=300).stdout.strip().splitlines()[-1]
        return json.loads(out)
    except Exception as e:
        return dict(error=repr(e)[:80])
    finally: os.remove(f.name)
def q8(m):
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(m).cpu(), {nn.Linear}, dtype=torch.qint8)

# ---------- experience ----------
base = PointNetSeg(); sd = torch.load(a.ckpt, map_location="cpu")
base.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd); base.eval()

CFG = [(1024, 256), (512, 256), (256, 128), (128, 64)]       # (largeur l3, largeur h1) ; le premier = TEMOIN
results = []
for c3, c4 in CFG:
    name = f"c3={c3},c4={c4}" + ("  (temoin)" if (c3, c4) == (1024, 256) else "")
    print(f"\n=== {name} ===", flush=True)
    V, T, VQ, TQ = [], [], [], []
    for s in range(a.seeds):
        m = from_base(base, c3, c4)
        if s == 0:
            print(f"  avant fine-tune : val mIoU={evaluate(m.to(GPU), iva, GPU)[1]:.4f}", flush=True)
        m = finetune(m, s)
        V.append(evaluate(m, iva, GPU)[1]); T.append(evaluate(m, ite, GPU)[1])
        m = m.cpu().eval(); q = q8(m)
        VQ.append(evaluate(q, iva)[1]); TQ.append(evaluate(q, ite)[1])
    r = dict(name=name, mmac=mmac_per_point(m), params=sum(p.numel() for p in m.parameters()),
             size_fp32=size_mb(m), size_int8=size_mb(q), lat_fp32=latency(m), lat_int8=latency(q),
             ram_fp32=ram(m), ram_int8=ram(q),
             val=(float(np.mean(V)), float(np.std(V))), test=(float(np.mean(T)), float(np.std(T))),
             val_int8=(float(np.mean(VQ)), float(np.std(VQ))), test_int8=(float(np.mean(TQ)), float(np.std(TQ))))
    results.append(r); print(json.dumps(r), flush=True)

print("\n=== RESUME (moyenne ± ecart-type sur", a.seeds, "seeds de fine-tuning ; meme checkpoint de base) ===")
print(f"{'config':26s} {'MMAC/pt':>7s} {'val fp32':>14s} {'test fp32':>14s} {'Mo':>5s} {'ms':>6s} | {'val int8':>14s} {'test int8':>14s} {'Mo':>5s} {'ms':>6s}")
for r in results:
    print(f"{r['name']:26s} {r['mmac']:7.3f} {r['val'][0]:.4f}±{r['val'][1]:.4f} {r['test'][0]:.4f}±{r['test'][1]:.4f} "
          f"{r['size_fp32']:5.2f} {r['lat_fp32']:6.1f} | {r['val_int8'][0]:.4f}±{r['val_int8'][1]:.4f} "
          f"{r['test_int8'][0]:.4f}±{r['test_int8'][1]:.4f} {r['size_int8']:5.2f} {r['lat_int8']:6.1f}")
print("\nRAM (surcout d'une inference, Mo) :")
for r in results: print(f"  {r['name']:26s} fp32={r['ram_fp32']}  int8={r['ram_int8']}")
json.dump(results, open("structured_results.json", "w"), indent=1)