import numpy as np
import matplotlib.pyplot as plt


def two_dim_heatmap(data, symbol, index1, index2, symbol_name, title):
    """
    data: dataset to include 3 columns: symbol, index1, index2
    symbol: variables to be analyzed
    index1: numeric value, heatmap 1
    index2: numeric value, heatmap 2
    symbol_name: the name of variable symbol shown on the plot
    REFERENCE: https://www.tutorialguruji.com/python/how-to-create-a-heatmap-for-2-columns-at-2-different-scales-in-python/
    """
    fig, ax = plt.subplots()
    N = data.index.size

    # first heatmap
    im1 = ax.imshow(np.vstack([data[index1],data[index1]]).T,
                    aspect='auto', extent=[-0.5,0.5,-0.5,N-0.5], origin='lower', cmap='magma')
    # second heatmap
    im2 = ax.imshow(np.vstack([data[index2],data[index2]]).T,
                    aspect='auto', extent=[0.5,1.5,-0.5,N-0.5], origin='lower', cmap='Blues')

    cbar1 = fig.colorbar(im1, ax=ax, label=index1)
    cbar2 = fig.colorbar(im2, ax=ax, label=index2)


    ax.set_xlim(-0.5,1.5)
    ax.set_xticks([0,1])
    ax.set_xticklabels([index1,index2])
    ax.set_yticks(range(N))
    ax.set_yticklabels(data[symbol])
    ax.set_ylabel(symbol_name)
    plt.title(title)
    fig.tight_layout()
    plt.show()