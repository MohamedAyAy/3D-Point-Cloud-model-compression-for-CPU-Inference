import re, sys
import matplotlib.pyplot as plt

files = sys.argv[1:] or ["log_base_s0.txt"]
pat = re.compile(r"\[(\S+)\] ep(\d+)/(\d+) loss=([\d.]+) val OA=([\d.]+) mIoU=([\d.]+)")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for f in files:
    ep, loss, miou = [], [], []
    for line in open(f):
        m = pat.search(line)
        if m:
            ep.append(int(m[2])); loss.append(float(m[4])); miou.append(float(m[6]))
    if not ep:
        print("aucune ligne reconnue dans", f); continue
    a1.plot(ep, loss, label=f); a2.plot(ep, miou, label=f)
a1.set(xlabel="époque", ylabel="loss d'entraînement", title="Loss d'entraînement"); a1.grid(alpha=.3); a1.legend()
a2.set(xlabel="époque", ylabel="mIoU validation", title="mIoU de validation"); a2.grid(alpha=.3); a2.legend()
fig.tight_layout(); fig.savefig("courbes_logs.png", dpi=200); print("courbes_logs.png")