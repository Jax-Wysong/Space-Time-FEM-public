import os
import re
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))

# (nx, nsub, nt) triples known to have diverged (10000 iter limit hit)
diverged_runs = {
    (100, 20, 200),
    (200, 20, 200),
    (200, 20, 380),
}

nx_styles = {
    50:  dict(color="steelblue", linestyle="-",  marker="o", label="$n_x = 50$"),
    100: dict(color="firebrick", linestyle="--", marker="s", label="$n_x = 100$"),
    200: dict(color="seagreen",  linestyle="-.", marker="^", label="$n_x = 200$"),
}

# data[nsub][nx] = sorted list of (p_pct, iters)
data = {}

for nsub_dir in os.listdir(base):
    m = re.fullmatch(r"Nsub-(\d+)", nsub_dir)
    if not m:
        continue
    nsub = int(m.group(1))
    for nx_dir in os.listdir(os.path.join(base, nsub_dir)):
        m2 = re.fullmatch(r"nx-(\d+)", nx_dir)
        if not m2:
            continue
        nx = int(m2.group(1))
        folder = os.path.join(base, nsub_dir, nx_dir)
        for fname in os.listdir(folder):
            if not fname.endswith(".out"):
                continue
            mf = re.search(r"_(\d+)x(\d+)_RAS_(\d+)ranks\.out$", fname)
            if not mf:
                continue
            nx_f   = int(mf.group(1))
            nt     = int(mf.group(2))
            nsub_f = int(mf.group(3))

            if (nx_f, nsub_f, nt) in diverged_runs:
                continue

            fpath = os.path.join(folder, fname)
            with open(fpath) as f:
                content = f.read()
            iters_match = re.search(r"iterations\s+(\d+)", content)
            if not iters_match:
                print(f"WARNING: no iteration count in {fname}")
                continue
            iters = int(iters_match.group(1))

            p_pct = 2 * (nsub_f - 1) / nt * 100
            data.setdefault(nsub, {}).setdefault(nx, []).append((p_pct, iters))

for nsub in data:
    for nx in data[nsub]:
        data[nsub][nx].sort(key=lambda x: x[0])

nsub_list = sorted(data.keys())

fig = plt.figure(figsize=(13, 8))
positions = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5)]

for (row, col, pos), nsub in zip(positions, nsub_list):
    ax = fig.add_subplot(row, col, pos)
    for nx, style in nx_styles.items():
        if nx not in data.get(nsub, {}):
            continue
        pts = data[nsub][nx]
        ax.plot([p for p, _ in pts], [i for _, i in pts], **style)
    if nsub == 20:
        ax.text(0.97, 0.95, "Diverged (10000 iters):\n$n_x$=100,200: $p$=20%\n$n_x$=200: $p$=10%",
                transform=ax.transAxes, fontsize=9,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))
    ax.set_title(f"$N_{{sub}} = {nsub}$", fontsize=12)
    ax.set_xlabel("$\\%p = 2(N_{sub}-1)/n_t \\times 100\\%$", fontsize=11)
    ax.set_ylabel("GMRES Iterations", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

fig.suptitle("GMRES Iterations vs. Percent of Polluted Eigenvalues", fontsize=13)
fig.tight_layout()

out_path = os.path.join(base, "gmres_vs_p.jpg")
fig.savefig(out_path)
print(f"Saved {out_path}")
plt.show()
