import matplotlib.pyplot as plt

def plot_sy(sy, boundaries=None):
    labels = ['stim cos', 'stim sin', 'fixation']
    _, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes = axes.ravel()
    for i in range(3):
        axes[i].plot(sy['s'][:, i], label='s')
        axes[i].plot(sy['y'][:, i], label='y')
        axes[i].set_ylim(-1.2, 1.2)

        if boundaries is not None:
            for b in boundaries:
                axes[i].axvline(b, color='k', linestyle='--')
        axes[i].set_ylabel(labels[i])
    axes[0].legend()
    plt.xlabel('timesteps')


def set_rcparams():
    plt.rcParams['figure.figsize'] = [2, 1.5]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'sans-serif']
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.25
    plt.rcParams['xtick.major.width'] = 0.25
    plt.rcParams['ytick.major.width'] = 0.25
    plt.rcParams['xtick.minor.width'] = 0.25
    plt.rcParams['ytick.minor.width'] = 0.25
    plt.rcParams['grid.linewidth'] = 0.25
    plt.rcParams['font.size'] = 8
    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['ytick.labelsize'] = 5
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8