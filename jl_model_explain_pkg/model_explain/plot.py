import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve

def cf_matrix_heatmap(cf_matrix):
    """
    :param cf_matrix: confusion matrix
    :return:
    """
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sb.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', annot_kws={'fontsize': 16})

def plot_randomfp_roc(x_train, y_train, x_val, y_val, pred_val_prob, model_label = 'Model'):
    random_predictions = DummyClassifier(strategy = 'stratified')
    random_predictions.fit(x_train, y_train)
    random_fpr, random_tpr, _ = roc_curve(y_val, random_predictions.predict(x_val))
    model_fpr, model_tpr, _ = roc_curve(y_val, pred_val_prob)
    # plot the roc curve for the model
    plt.figure(figsize=(10,6), dpi=100)
    plt.plot(random_fpr, random_tpr, linestyle = '--', label = 'RandomFlip')
    plt.plot(model_fpr, model_tpr, marker = '.', label = model_label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

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