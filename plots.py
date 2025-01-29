import matplotlib.pyplot as plt

def plot_sy(sy, boundaries=None):
    labels = ['stim cos', 'stim sin', 'fixation']
    _, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes = axes.ravel()
    for i in range(3):
        axes[i].plot(sy['s'][:, i], label='s')
        axes[i].plot(sy['y'][:, i], label='y')
        axes[i].set_ylim(-0.2, 1.2)

        if boundaries is not None:
            for b in boundaries:
                axes[i].axvline(b, color='k', linestyle='--')
        axes[i].set_ylabel(labels[i])
    axes[0].legend()
    plt.xlabel('timesteps')