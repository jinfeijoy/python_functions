import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def plot_count_dist(feature, title, df, size=1, ordered=True, horizontally=False):
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
    if horizontally:
        if ordered:
            g = sb.countplot(y=df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
        else:
            g = sb.countplot(y=df[feature], palette='Set3')
        g.set_title("Number and percentage of {}".format(title))
        adj_ratio = 2
        if (size > 2):
            plt.xticks(rotation=90, size=8)
            adj_ratio = 1
        for p in ax.patches:
            width = p.get_width()
            ax.text(width+50*adj_ratio,
                    p.get_y() + p.get_height() / 2.,
                    '{:1.2f}%'.format(100 * width / total),
                    ha="center")
    else:
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


def generate_bar_proportion(data, feature, group_var, color=0, order=False, ascending=False, topn=10, subtitle = None):
    """

    :param data: dataset
    :param feature: feature to be plot as bar proportion
    :param group_var: group by a variable, which is X in plot
    :param color: integer, color set (from 0 to 12)
    :param order: default is False, if order is true, then the X will be ordered by total records in each group
    :param ascending: particular for order, False: total volume from large to small, activated once order = True
    :param topn: top n gorups will shows in plot, activated once order = True
    :return:
    """
    background_color = '#f5f8fa'
    fig = plt.figure(figsize=(15, 3), dpi=150, facecolor=background_color)
    gs = fig.add_gridspec(1, 1)
    gs.update(wspace=0, hspace=0)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(background_color)
    color_list = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                  'tab20c', 'flag']

    feature_n_cat = len(data[feature].unique())
    color = plt.cm.get_cmap(color_list[color])(np.arange(feature_n_cat))

    col_index = data.groupby(feature)[group_var].value_counts().unstack().T.agg('mean').sort_values().index
    astra_temp = data.groupby(feature)[group_var].value_counts().unstack().fillna(0).loc[col_index].T
    astra_all = astra_temp.sum(axis=1)
    astra_temp = (astra_temp.T / astra_all).cumsum().T

    if order == True:
        selected_list = astra_all.sort_values(ascending=ascending).index[0:topn]
        astra_temp = astra_temp[astra_temp.index.isin(selected_list)]
        astra_temp.index = astra_temp.index.str.strip()
        astra_temp = astra_temp.reindex(selected_list)
    if order == False:
        astra_temp = astra_temp

    for i, sents in enumerate(astra_temp.columns[::-1]):
        if sents in list(astra_temp.columns):
            sentims = astra_temp[sents]
            ax0.bar(sentims.index, sentims, color=color[i], label=sents)
    for s in ['top', 'right', 'left']:
        ax0.spines[s].set_visible(False)

    ax0.set_yticks([])
    Xstart, Xend = ax0.get_xlim()
    Ystart, Yend = ax0.get_ylim()

    ax0.set_ylabel(" ", fontsize=8, loc='top', fontfamily='monospace')
    ax0.set_xlabel(" ", fontsize=8, loc='left', fontfamily='arial')
    ax0.tick_params(axis="both", which="both", left=False, bottom=False)
    ax0.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax0.text(Xstart, 1.26, feature.title() + ' Over Different ' + group_var.title(), fontweight='bold', fontsize=16,
             zorder=20)
    ax0.text(Xstart,1.125,subtitle,fontweight='light', fontsize=14, zorder=20)
    ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax0.grid()
    plt.show()
