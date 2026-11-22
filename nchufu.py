import json, numpy as np
old = json.load(open("final_results.json"))["raw"]
new = json.load(open("final_results_fixed.json"))["raw"]
print(f"{'config':10s} {'mIoU test (std)':>16s} {'mIoU test (fixed)':>18s} {'Δ moyen':>9s}  Δ par seed")
for k in old:
    o, n = np.array(old[k]["test"]), np.array(new[k]["test"])
    dl = n - o
    print(f"{k:10s} {o.mean():16.4f} {n.mean():18.4f} {dl.mean():+9.4f}  {np.round(dl, 4)}")