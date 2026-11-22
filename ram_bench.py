"""ram_bench.py - surcout memoire d'UNE inference (1 nuage, 4096 pts, 1 thread) pour chaque largeur du Student.
Autonome (poids aleatoires : la memoire ne depend pas des valeurs). Usage : python ram_bench.py
Chaque mesure tourne dans un processus neuf ; on echantillonne le RSS COURANT (pas le pic historique).
"""
import json, os, subprocess, sys, threading, time
CFG = [(1024, 256), (512, 256), (256, 128), (128, 64)]

def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6

if len(sys.argv) > 1 and sys.argv[1] == "--child":
    c3, c4, q = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "int8"
    import torch, torch.nn as nn
    torch.set_num_threads(1)
    class SlimNet(nn.Module):
        def __init__(self, c3, c4):
            super().__init__()
            self.l1, self.l2, self.l3 = nn.Linear(3, 64), nn.Linear(64, 128), nn.Linear(128, c3)
            self.h1, self.h2 = nn.Linear(128 + c3, c4), nn.Linear(c4, 128)
            self.out = nn.Linear(128, 17); self.r = nn.ReLU()
        def forward(self, x):
            l = self.r(self.l2(self.r(self.l1(x))))
            g = self.r(self.l3(l)).max(1, keepdim=True)[0].expand(-1, l.size(1), -1)
            return self.out(self.r(self.h2(self.r(self.h1(torch.cat([l, g], -1))))))
    m = SlimNet(c3, c4).eval()
    if q: m = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    with torch.no_grad(): m(torch.randn(1, 16, 3))          # echauffement sur 16 points (charge les libs, pas les gros tenseurs)
    x = torch.randn(1, 4096, 3)
    base = rss_mb(); peak = [base]; stop = [False]
    def sample():
        while not stop[0]:
            peak[0] = max(peak[0], rss_mb()); time.sleep(0.0005)
    th = threading.Thread(target=sample); th.start()
    with torch.no_grad(): m(x)
    stop[0] = True; th.join()
    print(json.dumps(dict(rss_before_MB=round(base, 1), extra_during_forward_MB=round(peak[0] - base, 1))))
    sys.exit()

print(f"{'config':14s} {'mode':5s} {'RSS avant':>10s} {'surcout mesure':>15s} {'estimation h1 (Mo)':>19s}")
for c3, c4 in CFG:
    est = 4096 * 4 * (128 + c3 + c4) / 1e6        # entree concat + sortie de h1, en float32
    for mode in ("fp32", "int8"):
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", str(c3), str(c4), mode],
                             capture_output=True, text=True).stdout.strip().splitlines()
        try: r = json.loads(out[-1])
        except Exception: r = dict(rss_before_MB="?", extra_during_forward_MB="erreur")
        print(f"{c3:>5d}/{c4:<8d} {mode:5s} {r['rss_before_MB']:>10} {r['extra_during_forward_MB']:>15} {est:>19.1f}")