import os
import re
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))

folder_to_p = {
    "p~1%":    1.0,
    "p~2_5%":  2.5,
    "p~5%":    5.0,
    "p~10%":  10.0,
    "p~20%":  20.0,
}

# data[nsub][p_pct] = gmres_iters
data = {}

for folder, p_val in folder_to_p.items():
    folder_path = os.path.join(base, folder)
    for fname in os.listdir(folder_path):
        if not fname.endswith(".out"):
            continue
        m = re.search(r"_(\d+)ranks\.out$", fname)
        if not m:
            continue
        nsub = int(m.group(1))

        fpath = os.path.join(folder_path, fname)
        with open(fpath) as f:
            content = f.read()
        iters_match = re.search(r"iterations\s+(\d+)", content)
        if not iters_match:
            print(f"WARNING: no iteration count in {fname}")
            continue
        iters = int(iters_match.group(1))

        data.setdefault(nsub, {})[p_val] = iters

p_vals = sorted(folder_to_p.values())
nsub_list = sorted(data.keys())

colors = ["steelblue", "firebrick", "seagreen", "darkorange", "mediumpurple"]

fig = plt.figure(figsize=(13, 8))
positions = [
    (2, 3, 1), (2, 3, 2), (2, 3, 3),
    (2, 3, 4), (2, 3, 5),
]

for (row, col, pos), nsub, color in zip(positions, nsub_list, colors):
    ax = fig.add_subplot(row, col, pos)
    xs = []
    ys = []
    for p in p_vals:
        if p in data[nsub] and not (nsub == 20 and p == 20.0):
            xs.append(p)
            ys.append(data[nsub][p])
    ax.plot(xs, ys, marker="o", color=color)
    if nsub == 20:
        ax.text(0.97, 0.95, "$p$ = 20% diverged\n(10000 iters)",
                transform=ax.transAxes, fontsize=9,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))
    ax.set_title(f"$N_{{sub}} = {nsub}$", fontsize=12)
    ax.set_xlabel("$\%p = 2(N_{{sub}}-1)/n_t * 100\%$", fontsize=11)
    ax.set_ylabel("GMRES Iterations", fontsize=11)
    ax.set_xticks(p_vals)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)

fig.suptitle("GMRES Iterations vs. Percent of Polluted Eigenvalues", fontsize=13)
fig.tight_layout()

out_path = os.path.join(base, "gmres_vs_p.jpg")
fig.savefig(out_path)
print(f"Saved {out_path}")
plt.show()
