import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

class VariableImportancePlot(object):
    def __init__(self):
        pass

    def plot_var_imp(feature_importances, column_name, topn = 20):
        """
        :param feature_importances: feature importance from model
        :param column_name: column name in dataset: e.g. data.columns
        :param topn: top n important variables
        :return:
        """
        y_axis = [i for i in column_name]
        x_axis = feature_importances
        xy_sorted = list(sorted([(i, j) for i, j in zip(x_axis, y_axis)], key=lambda x: x[0], reverse=True))
        x_axis_new = [i[0] for i in xy_sorted[:topn]]
        y_axis_new = [i[1] for i in xy_sorted[:topn]]
        plt.figure(figsize=(18, 10))
        sb.barplot(x=x_axis_new, y=y_axis_new)
        plt.title("Feature Importance")
        plt.ylabel("Features")
        plt.yticks(fontsize=16)

    def plot_var_imp_both_side(X, Y, column_name, raw_data, feature_importances, topn = 20):
        """
        :param X: input variables data
        :param Y: target variables data
        :param column_name: column name in dataset: e.g. data.columns
        :param raw_data: raw_data that include all variables and all records
        :param feature_importances: feature importance from model
        :param topn: top n importance variables
        :return:
        """
        M, N = X.shape
        print(N, ' features')
        features = {}
        for c in set(Y):
            features[c] = dict(
                zip(range(N), np.mean(X[raw_data['label'] == c], axis=0) * feature_importances)
            )
        y_axis = [i for i in column_name]
        x_axis = feature_importances
        xy_sorted = list(sorted([(i, j) for i, j in zip(x_axis, y_axis)], key=lambda x: x[0], reverse=True))
        features[0] = {xy_sorted[k][1]: v for k, v in features[0].items()}
        features[1] = {xy_sorted[k][1]: v for k, v in features[1].items()}
        f, axes = plt.subplots(1, 2)
        f.set_figheight(15)
        f.set_figwidth(20)
        # Positive
        y_axis_pos = list(features[1].keys())[:topn]
        x_axis_pos = list(features[1].values())[:topn]
        axes[0].set_title("Feature Importance for the Positive Target", size=25)
        axes[0].set_ylabel("Features/Words", fontsize=20)
        axes[0].set_yticklabels(labels=y_axis_pos, fontsize=20)
        # Negative
        axes[1].set_title("Feature Importance for the Negative Target", size=25)
        axes[1].set_yticklabels(labels=y_axis_pos, fontsize=20)
        y_axis_neg = list(features[0].keys())[:topn]
        x_axis_neg = list(features[0].values())[:topn]
        sb.barplot(x=x_axis_pos, y=y_axis_pos, ax=axes[0])
        sb.barplot(x=x_axis_neg, y=y_axis_neg, ax=axes[1])

        f.tight_layout()