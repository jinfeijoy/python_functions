import pandas as pd
import shap
import matplotlib.pyplot as plt

shap.initjs()

def get_SHAP_plot(model, X, define_index = False, dataset = None, index_var = None, index_value = 0, savefig = False, path = ''):
    """

    :param model: model to be explained
    :param X: a dataframe with all input variables
    :param define_index: True or False, whether to define a index variable
    :param dataset: if defind_index is True, this dataset is a dataset with X and index variable
    :param index_var: string: index variable
    :param index_value: if define_index is False, this is the index of X, if define_index is True, this is the index value
    :param savefig: True or False, whether to save plot in a png file
    :param path: path to save png file
    :return: SHAP plot and SHAP value data frame
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if define_index == True:
        id = dataset[dataset[index_var] == index_value].index
    if define_index == False:
        id = [index_value]
    if savefig == True:
        plot = shap.force_plot(explainer.expected_value[0], shap_values[0][id], X.iloc[id,:], show = False, matplotlib=True)
        plt.text(.5, .5, 'IndexVar {0} SHAP Detail'.format(index_value), ha='center', fontsize=15)
        plot.savefig(path + 'SHAP_plot_' + str(index_value) + '.png', bbox_inches='tight')
    if savefig == False:
        plot = shap.force_plot(explainer.expected_value[0], shap_values[0][id], X.iloc[id,:])


    sample_shap_value = pd.DataFrame(shap_values[0][id], columns = X.columns.values)
    sample_shap_value = sample_shap_value.append(X.iloc[id,:][X.columns.values])
    sample_shap_value = sample_shap_value.T
    sample_shap_value.columns = ['SHAP_Value', 'Sample_value']
    sample_shap_value['index_var'] = index_value
    sample_shap_value = sample_shap_value.sort_values(by=['SHAP_Value'], ascending=False)
    return plot, sample_shap_value