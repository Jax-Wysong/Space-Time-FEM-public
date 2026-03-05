"""
Plot rewritten from MATLAB to matplotlib.
Saves figure to IC_5_Lam66_tF_vs_A.png (dpi=200) and shows plot.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Data
t_F = np.array([336, 233, 182, 160, 142, 142, 150, 141, 126, 124, 125, 117, 114, 127, 132, 162, 224, 224, 151, 225, 94, 26, 15, 8, 8, 5, 4, 4, 3, 3, 2])
A = np.array([0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.705, 0.71, 0.75, 0.8, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.275, 1.3, 1.325, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])


def plot_tF_vs_A(save_path='IC_5_Lam66_tF_vs_A.png', show=True):
	fig, ax = plt.subplots(figsize=(8, 6))

	ax.plot(A, t_F, '-o', color='tab:blue', linewidth=1.2,
			markersize=5, markerfacecolor='none', label=r'$t_F$')

	ax.axvline(1 / np.sqrt(2), color='red', linestyle=':', linewidth=1.5,
			   label=r'$A = 1/\sqrt{2}$')

	ax.set_xlabel('Amplitude (A)')
	ax.set_ylabel('Final Time Survived ' + r'($t_F$)')
	ax.set_title('Time of Instability vs Initial Amplitude')
	ax.grid(True)
	ax.legend(loc='best', fontsize='medium')
	fig.tight_layout()

	fig.savefig(save_path, dpi=200)

	if show:
		plt.show()

def set_plot_style(font="DejaVu Serif", fontsize=16):
    mpl.rcParams["font.family"] = font
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["axes.titlesize"] = fontsize
    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["xtick.labelsize"] = int(max(8, fontsize * 0.9))
    mpl.rcParams["ytick.labelsize"] = int(max(8, fontsize * 0.9))
    mpl.rcParams["legend.fontsize"] = int(max(8, fontsize * 0.9))

# Apply global matplotlib style before creating any figures
# Font Type: DejaVu Serif
# Font Size: 16 pt
set_plot_style()

if __name__ == '__main__':
	plot_tF_vs_A()
