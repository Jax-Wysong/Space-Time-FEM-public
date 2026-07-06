import os
import re
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))

# (nx, nsub, nt) triples known to have diverged (10000 iter limit hit)
diverged_runs = {
}

nx_styles = {
    50:  dict(color="steelblue", linestyle="-",  marker="o", label="$n_x = 50$"),
    100: dict(color="firebrick", linestyle="--", marker="s", label="$n_x = 100$"),
    200: dict(color="seagreen",  linestyle="-.", marker="^", label="$n_x = 200$"),
}

# data[nsub][nx] = sorted list of (p_pct, iters)
data = {}
# nt_for_p[nsub][p_pct] = nt  (for top axis labels)
nt_for_p = {}

FILTER_NSUB = {}          # temporary: remove to plot all Nsub values
FILTER_NX   = {}   # temporary: remove to plot all nx values

for nsub_dir in os.listdir(base):
    m = re.fullmatch(r"Nsub-(\d+)", nsub_dir)
    if not m:
        continue
    nsub = int(m.group(1))
    if FILTER_NSUB and nsub not in FILTER_NSUB:
        continue
    for nx_dir in os.listdir(os.path.join(base, nsub_dir)):
        m2 = re.fullmatch(r"nx-(\d+)", nx_dir)
        if not m2:
            continue
        nx = int(m2.group(1))
        if FILTER_NX and nx not in FILTER_NX:
            continue
        folder = os.path.join(base, nsub_dir, nx_dir)
        for fname in os.listdir(folder):
            if not fname.endswith(".out"):
                continue
            mf = re.search(r"RAS_(\d+)x(\d+)_(\d+)ranks\.out$", fname)
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
            nt_for_p.setdefault(nsub_f, {})[p_pct] = nt

for nsub in data:
    for nx in data[nsub]:
        data[nsub][nx].sort(key=lambda x: x[0])

nsub_list = sorted(data.keys())

def pick_ticks(vals, n=5):
    """Return n values from sorted list, always including first and last."""
    if len(vals) <= n:
        return list(vals)
    indices = sorted({0, len(vals) - 1} |
                     {round(i * (len(vals) - 1) / (n - 1)) for i in range(1, n - 1)})
    return [vals[i] for i in indices]

fig = plt.figure(figsize=(13, 8))

positions = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5)]

for (row, col, pos), nsub in zip(positions, nsub_list):
    ax = fig.add_subplot(row, col, pos)
    for nx, style in nx_styles.items():
        if nx not in data.get(nsub, {}):
            continue
        pts = data[nsub][nx]
        ax.plot([p for p, _ in pts], [i for _, i in pts], **style)
    # if nsub == 20:
    #     ax.text(0.97, 0.95, "Diverged (10000 iters):\n$n_x$=100,200: $p$=20%\n$n_x$=200: $p$=10%",
    #             transform=ax.transAxes, fontsize=9,
    #             ha="right", va="top",
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))
    ax.set_title(f"$N_{{sub}} = {nsub}$", fontsize=12)
    ax.set_xlabel("$\\%p$", fontsize=11)
    ax.set_ylabel("GMRES Iterations", fontsize=11)

    # set bottom ticks to a reduced subset of the actual %p values
    p_ticks_all = sorted(nt_for_p.get(nsub, {}).keys())
    p_ticks = pick_ticks(p_ticks_all)
    ax.set_xticks(p_ticks)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    # top axis: nt values corresponding to the same reduced tick set
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(p_ticks)
    ax2.set_xticklabels([str(nt_for_p[nsub][p]) for p in p_ticks], rotation=45, fontsize=8)
    ax2.set_xlabel("$n_t$", fontsize=11)

fig.suptitle("GMRES Iterations vs. Percent of Polluted Eigenvalues", fontsize=13)
fig.tight_layout()

out_path = os.path.join(base, "gmres_vs_p.jpg")
fig.savefig(out_path)
print(f"Saved {out_path}")
plt.show()
