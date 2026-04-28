import os
import re
import matplotlib.pyplot as plt

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

nx_dirs = {
    50:  os.path.join(results_dir, "pollution_percent_nx-50"),
    100: os.path.join(results_dir, "pollution_percent_nx-100"),
    200: os.path.join(results_dir, "pollution_percent_nx-200"),
}

folder_to_p = {
    "p~1%":    1.0,
    "p~2_5%":  2.5,
    "p~2.5%":  2.5,
    "p~5%":    5.0,
    "p~10%":  10.0,
    "p~20%":  20.0,
}

# data[nx][nsub] = sorted list of (nt_nx_ratio, iters)
data = {}

for nx, base in nx_dirs.items():
    data[nx] = {}
    for folder, p_val in folder_to_p.items():
        folder_path = os.path.join(base, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".out"):
                continue
            m = re.search(r"_(\d+)x(\d+)_RAS_(\d+)ranks\.out$", fname)
            if not m:
                continue
            nx_file = int(m.group(1))
            nt      = int(m.group(2))
            nsub    = int(m.group(3))

            diverged = (nsub == 20 and p_val == 20.0 and nx in (100, 200)) or \
                       (nsub == 20 and p_val == 10.0 and nx == 200)
            if diverged:
                continue

            fpath = os.path.join(folder_path, fname)
            with open(fpath) as f:
                content = f.read()
            iters_match = re.search(r"iterations\s+(\d+)", content)
            if not iters_match:
                print(f"WARNING: no iteration count in {fname}")
                continue
            iters = int(iters_match.group(1))

            ratio = nx_file / nt
            data[nx].setdefault(nsub, []).append((ratio, iters))

# sort each list by nt/nx ratio
for nx in data:
    for nsub in data[nx]:
        data[nx][nsub].sort(key=lambda x: x[0])

nsub_list = sorted({nsub for nx_data in data.values() for nsub in nx_data})

nx_styles = {
    50:  dict(color="steelblue", linestyle="-",  marker="o", label="$n_x = 50$"),
    100: dict(color="firebrick", linestyle="--", marker="s", label="$n_x = 100$"),
    200: dict(color="seagreen",  linestyle="-.", marker="^", label="$n_x = 200$"),
}

fig = plt.figure(figsize=(13, 8))
positions = [
    (2, 3, 1), (2, 3, 2), (2, 3, 3),
    (2, 3, 4), (2, 3, 5),
]

for (row, col, pos), nsub in zip(positions, nsub_list):
    ax = fig.add_subplot(row, col, pos)
    for nx, style in nx_styles.items():
        if nsub not in data[nx]:
            continue
        pts = data[nx][nsub]
        xs = [r for r, _ in pts]
        ys = [i for _, i in pts]
        ax.plot(xs, ys, **style)
    if nsub == 20:
        ax.text(0.97, 0.95, "Diverged (10000 iters):\n$n_x$=100,200: $p$=20%\n$n_x$=200: $p$=10%",
                transform=ax.transAxes, fontsize=9,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))
    ax.set_title(f"$N_{{sub}} = {nsub}$", fontsize=12)
    ax.set_xlabel("$n_x / n_t$  $(\propto$ CFL$)$", fontsize=11)
    ax.set_ylabel("GMRES Iterations", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

fig.suptitle("GMRES Iterations vs. $n_x / n_t$  $(\propto$ CFL$)$", fontsize=13)
fig.tight_layout()

out_path = os.path.join(results_dir, "pollution_percent_nx-100", "gmres_vs_nx_nt.jpg")
fig.savefig(out_path)
print(f"Saved {out_path}")
plt.show()
