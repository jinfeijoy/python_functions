import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(data, x, y, title='', figsize = (15,6)):
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, cmap='YlOrBr', annot=False)
    ax.set_ylabel(y, fontsize=15)
    ax.set_xlabel(x, fontsize=15)
    ax.set_title(title, fontsize=20, weight='bold')
    plt.show()
