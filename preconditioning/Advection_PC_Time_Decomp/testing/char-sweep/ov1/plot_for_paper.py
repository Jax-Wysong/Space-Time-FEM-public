import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LEGEND_FS      = 11
SUPER_TITLE_FS = 16

nsub = [2, 4, 8, 10, 20, 40]

data = {
    '$n_x=100,\\ n_t=200$': {'base': [6, 15, 43, 57, 138, 376], 'char': [5, 13, 33, 44, 106, 169]},
    '$n_x=200,\\ n_t=400$': {'base': [4, 12, 39, 57, 137, 376], 'char': [4,  9, 30, 42,  94, 205]},
    '$n_x=300,\\ n_t=600$': {'base': [4, 12, 38, 53, 148, 470], 'char': [4,  9, 27, 39, 104, 254]},
}

nsub_400 = [2, 4, 8, 10, 20, 40, 80, 100, 160]
base_400 = [4, 11, 36, 52, 159, 406, 870, 1046, 1516]
char_400 = [3,  8, 27, 37, 105, 206, 573,  747, 1094]

fig, axes = plt.subplots(1, 4, figsize=(15, 5), constrained_layout=True)
fig.suptitle('GMRES Iterations: Baseline vs. Characteristic IC  (overlap=1)', fontsize=SUPER_TITLE_FS)

for ax, (label, d) in zip(axes[:3], data.items()):
    ax.plot(nsub, d['base'], color="#002E79", linestyle='--', marker='o', label='Baseline')
    ax.plot(nsub, d['char'], color="#F80828", linestyle='-',  marker='s', label='Characteristic IC')
    ax.set_title(label, fontsize=11)
    ax.set_xlabel('$N_{sub}$', fontsize=12)
    ax.set_ylabel('GMRES Iterations', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FS)

ax4 = axes[3]
ax4.plot(nsub_400, base_400, color="#002E79", linestyle='--', marker='o', label='Baseline')
ax4.plot(nsub_400, char_400, color="#F80828", linestyle='-',  marker='s', label='Characteristic IC')
ax4.set_title('$n_x=400,\\ n_t=800$  (log–log)', fontsize=11)
ax4.set_xlabel('$N_{sub}$', fontsize=12)
ax4.set_ylabel('GMRES Iterations', fontsize=12)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3, which='both')
ax4.legend(fontsize=LEGEND_FS)

fig.savefig('gmres_char_ic.jpg', dpi=150)
print('Saved gmres_char_ic.jpg')
