import matplotlib.pyplot as plt
import seaborn as sb

def plot_count_dist(feature, title, df, size=1, ordered=True):
    """
    :param feature: the feature we want to plot (distribution)
    :param title: the feature name shown in the title
    :param df: dataset
    :param size: if feature values are more than 1 then input numbers e.g.4 to make the plot wider
    :param ordered: to make distribution in ordered
    :return:
    """
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    if ordered:
        g = sb.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
    else:
        g = sb.countplot(df[feature], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height,
                '{:1.2f}%'.format(100*height/total),
                ha="center")
    plt.show()