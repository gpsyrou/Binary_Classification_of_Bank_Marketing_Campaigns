"""
"""

# Import dependencies
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Evaluation metrics
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

sns.set_style("dark")


def visualizeCategoricalVar(input_df: pd.core.frame.DataFrame, col: str,
                            hue: str, figsize=(10, 8)) -> None:
    """
    Plot the number of subscribed clients for a specified categorical column
    of a dataframe.

    Input:
    ----
        input_df: Dataframe that contains the dataset
        col: Column of the dataframe that will be used by the countplot
        hue: Column for which to split the dataset in terms of color
    Returns:
    ------
        Countplot of the specified variable
    """
    size = float(input_df.shape[0])

    plt.figure(figsize=figsize)
    ax = sns.countplot(x=col, data=input_df, hue=hue)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2., height + 4, '{:1.2f}%'.format(
                100 * height/size), ha="center")
    if hue is None:
        plt.title('Number of Subscribed Clients', fontweight='bold')
    else:
        plt.title(f'Number of Subscribed Clients per \'{col}\'',
                  fontweight='bold')
    plt.grid(True, alpha=0.3, color='black')
    plt.legend(loc='best')
    plt.show()


def visualizeNumericalVar(input_df: pd.core.frame.DataFrame,
                          col: str,
                          hue: str,
                          figsize=(10, 8)) -> None:
    """
    Generate a Boxplot and a Histogram for a numerical variable.

    Input:
    ----
        input_df: Dataframe that contains the dataset
        col: Column of the dataframe that will be used by the boxplot/histogram
        hue: Column for which to split the dataset in terms of color
    Returns:
    ------
        Boxplot and overlapping histograms of the specified variable
    """
    box_pal = {"no": "darkcyan", "yes": "firebrick"}

    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    # Boxplot
    sns.boxplot(data=input_df, x=hue, y=col, palette=box_pal,
                ax=ax[0]).set(title=f'Boxplot for \'{col}\'')
    # First Histogram
    sns.distplot(input_df[input_df[hue] == 'no'][col], color='darkcyan',
                 fit=norm, fit_kws={"color": "darkcyan"},  kde=False,
                 ax=ax[1]).set(title=f'Histogram of \'{col}\'')
    # Second Histogram
    sns.distplot(input_df[input_df[hue] == 'yes'][col], color='firebrick',
                 fit=norm, fit_kws={"color": "indianred"}, kde=False,
                 ax=ax[1]).set(title=f'Histogram of \'{col}\'')

    ax[0].grid(True, alpha=0.1, color='black')
    ax[1].grid(True, alpha=0.1, color='black')
    fig.show()


def bucketPday(pday_val):
    """
    Split the pday feature into 5 different buckets, depending on how many days
    (if any) have passed since they last contacted from a previous campaign.
    The bucket measures the number of weeks passed since the last contact.
    """
    if pday_val == 999:
        return 0
    elif (pday_val >= 0) and (pday_val <= 7):
        return 1
    elif (pday_val >= 8) and (pday_val <= 14):
        return 2
    elif (pday_val >= 15) and (pday_val <= 21):
        return 3
    else:
        return 4


def getModelEvaluationMetrics(classifier, model_name: str,
                              x_test: pd.core.frame.DataFrame,
                              y_test: pd.core.frame.DataFrame, y_predicted,
                              plot_confusion_matrix=False,
                              figsize=(10, 8)) -> np.ndarray:
    """
    Calculate the Precision and Recall of a classifier. Return the results
    as a matrix with the scores, as well as a Confusion Matrix plot.
    """
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_predicted)
    print('Confusion matrix:\n\n {0}'.format(conf_mat))

    if plot_confusion_matrix:
        labels = ['no', 'yes']
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        cax = ax.matshow(conf_mat, cmap=plt.cm.Reds)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('Expected')
        plt.title(f'Confusion Matrix for {model_name}', fontweight='bold')
        plt.show()

    # Calculating the precision (tp/tp+fp)
    precision = str(np.round((conf_mat[1][1] / (conf_mat[1][1] +
                              conf_mat[0][1])) * 100, 2))
    print('The precision is: {0} %'.format(precision))

    # Calculating the recall (tp/tp+fn)
    recall = str(np.round((conf_mat[1][1] / (conf_mat[1][1] +
                           conf_mat[1][0])) * 100, 2))
    print('The recall is: {0} %'.format(recall))

    return conf_mat


def createROCAnalysis(classifier, model_name: str,
                      y_test: pd.core.series.Series, pred_probs: np.ndarray,
                      plot_ROC_Curve=False, figsize=(10, 8)) -> int:
    """
    Perform a ROC-AUC analysis for a specified classifier.

    Args:
    -----
        classifier: Model based on which we perform the ROC analysis.
        model_name: Name of the model (e.g. 'Logistic Regression')
        pred_probs: Predicted probabilites for each instance/class
        plot_ROC_Curve: Plot the ROC Curve against a random model

    Returns:
        The Area Under Curve (AUC) score for the specific classifier
    """
    if plot_ROC_Curve:
        plt.figure(figsize=figsize)
        plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill Classifier')
        fp_rate, tp_rate, _ = roc_curve(y_test, pred_probs[:, 1])
        plt.plot(fp_rate, tp_rate, marker='.', label=model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}', fontweight='bold')
        plt.grid(True, alpha=0.1, color='black')
        plt.legend(loc='lower right')
        plt.show()

    # Calculate Area Under Curve (AUC) for the Receiver Operating
    # Characteristics Curve (ROC)
    auc_score = np.round(roc_auc_score(y_test, pred_probs[:, 1]), 4)
    print(f'{model_name} - ROC AUC score: {auc_score}')

    return auc_score
