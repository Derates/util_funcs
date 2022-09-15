import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    explained_variance_score, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.metrics import roc_curve, auc, roc_auc_score, DetCurveDisplay, RocCurveDisplay, \
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix, classification_report, log_loss, \
    matthews_corrcoef, accuracy_score, f1_score, hamming_loss, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns

# TODO: draw DecisionBoundaryDisplay
#   https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html#sklearn.inspection.DecisionBoundaryDisplay.from_estimator

def regression_evaluation(model_name, y_test, y_test_pred, y_train=None, y_train_pred=None):

    # TODO: add func docstring

    print('----------------------------------------',model_name,'----------------------------------------')
    if y_train is not None and y_train_pred is not None:
        print('-----Train Set Metrics-----')
        r2 = r2_score(y_train, y_train_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        evs = explained_variance_score(y_train, y_train_pred)
        mae = mean_absolute_error(y_train, y_train_pred)
        mse = mean_squared_error(y_train, y_train_pred)
        mape = mean_absolute_percentage_error(y_train, y_train_pred)

        print('R-squared                         : ', r2)
        print('Root Mean Squared Error           : ', rmse)
        print('Explained Variance Score          : ', evs)
        print('Mean Absolute Error               : ', mae)
        print('Mean Squared Error                : ', mse)
        print('Mean Absolute Percentage Error    : ', mape)
        if min(y_train)>=0 and min(y_train_pred)>=0:
            msle = mean_squared_log_error(y_train, y_train_pred)
            print('Mean Squared Logarithmic Error    : ', msle, '\n')
        else:
            msle = None

        score_df = pd.DataFrame({
            'train_R2': r2,
            'train_RMSE': rmse,
            'train_EVS': evs,
            'train_MAE': mae,
            'train_MSE': mse,
            'train_MAPE': mape,
            'train_MSLE': msle
        }, index=[model_name])

        print('-----Train Set Plots-----')
        plt.figure(figsize=[15, 4])

        plt.subplot(1, 2, 1)
        sns.distplot((y_train - y_train_pred))
        plt.title('Error Terms')
        plt.xlabel('Errors')

        plt.subplot(1, 2, 2)
        plt.scatter(y_train, y_train_pred)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
        plt.title('Train vs Prediction')
        plt.xlabel('y_test')
        plt.ylabel('y_pred')
        plt.show()

    print('-----Test Set Metrics-----')

    # R-squared
    r2 = r2_score(y_test, y_test_pred)
    print('R-squared                         : ', r2)

    #RMSE
    rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print('Root Mean Squared Error           : ', np.sqrt(mean_squared_error(y_train, y_train_pred)))

    # Explained Variance Score
    evs = explained_variance_score(y_test, y_test_pred)
    print('Explained Variance Score          : ', evs)

    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_test_pred)
    print('Mean Absolute Error               : ', mae)

    # Mean Squared Error
    mse = mean_squared_error(y_test, y_test_pred)
    print('Mean Squared Error                : ', mse)

    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_test, y_test_pred)
    print('Mean Absolute Percentage Error    : ', mape)

    # Mean Squared Logarithmic Error
    if min(y_test) >= 0 and min(y_test_pred) >= 0:
        msle = mean_squared_log_error(y_test, y_test_pred)
        print('Mean Squared Logarithmic Error    : ', msle, '\n')
    else:
        msle = None

    score_df['test_R2'] = r2
    score_df['test_RMSE'] = rmse
    score_df['test_EVS'] = evs
    score_df['test_MAE'] = mae
    score_df['test_MSE'] = mse
    score_df['test_MAPE'] = mape
    score_df['test_MSLE'] = msle

    print('-----Test Set Plots-----')
    plt.figure(figsize=[15, 4])

    plt.subplot(1, 2, 1)
    sns.distplot((y_test - y_test_pred))
    plt.title('Error Terms')
    plt.xlabel('Errors')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Test vs Prediction')
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.show()

    ##Dummy Regressor R-squared with median strategy
    #rgs = DummyRegressor(strategy='median', random_state=0)
    #rgs.fit(X_train, y_train)
    #rgs.score(X_test, y_test)

    return score_df

def regression_roc_auc_score(y_true, y_pred, num_rounds=10000):
    """
    Computes Regression-ROC-AUC-score.
    Code source: https://towardsdatascience.com/how-to-calculate-roc-auc-score-for-regression-models-c0be4fdf76bb

    Parameters:
    ----------
    y_true: array-like of shape (n_samples,). Binary or continuous target variable.
    y_pred: array-like of shape (n_samples,). Target scores.
    num_rounds: int or string. If integer, number of random pairs of observations.
                If string, 'exact', all possible pairs of observations will be evaluated.

    Returns:
    -------
    rroc: float. Regression-ROC-AUC-score.
    """

    def _yield_pairs(y_true, num_rounds):
        """
        Returns pairs of valid indices. Indices must belong to observations having different values.

        Parameters:
        ----------
        y_true: array-like of shape (n_samples,). Binary or continuous target variable.
        num_rounds: int or string. If integer, number of random pairs of observations to return.
                    If string, 'exact', all possible pairs of observations will be returned.

        Yields:
        -------
        i, j: tuple of int of shape (2,). Indices referred to a pair of samples.

        """

        if num_rounds == 'exact':
            for i in range(len(y_true)):
                for j in np.where((y_true != y_true[i]) & (np.arange(len(y_true)) > i))[0]:
                    yield i, j
        else:
            for r in range(num_rounds):
                i = np.random.choice(range(len(y_true)))
                j = np.random.choice(np.where(y_true != y_true[i])[0])
                yield i, j

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_pairs = 0
    num_same_sign = 0

    for i, j in _yield_pairs(y_true, num_rounds):
        diff_true = y_true[i] - y_true[j]
        diff_score = y_pred[i] - y_pred[j]
        if diff_true * diff_score > 0:
            num_same_sign += 1
        elif diff_score == 0:
            num_same_sign += .5
        num_pairs += 1

    return num_same_sign / num_pairs

def cross_validation(model, X_train, y_train, cv=5, scoring=None, n_permutations=1000):
    # TODO: add func docstring & add available scoring metrics
    #   https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #   https://scikit-learn.org/stable/auto_examples/model_selection/plot_permutation_tests_for_classification.html#sphx-glr-auto-examples-model-selection-plot-permutation-tests-for-classification-py

    scores = cross_val_score(model, X_train, y_train, cv, scoring=scoring)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    # TODO: add comparison with dummy regressor

    score_cv, perm_scores, pvalue = permutation_test_score(
        model, X_train, y_train, scoring=scoring, cv=cv, n_permutations=n_permutations
    )

    fig, ax = plt.subplots()

    ax.hist(perm_scores, bins=20, density=True)
    ax.axvline(score_cv, ls="--", color="r")
    score_label = f"Score on original\ndata: {score_cv:.2f}\n(p-value: {pvalue:.3f})"
    ax.text(0.7, 10, score_label, fontsize=12)
    ax.set_xlabel("Score")
    _ = ax.set_ylabel("Probability")

    # TODO : add nested CV option
    #   https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py

def classification_roc_auc(y_test, y_score):
    """
    Draws ROC Curve of a single or multiple model results of binary classification.

    :param y_test: ture labels
    :type y_test: np.array or pd.Series
_
    :param y_score: dictionary containing the name of the model as key and predictions
    (probabilities or decision function) as value.
    :type y_score: dict

    Returns: None

    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure()

    for i,key in enumerate(list(y_score.keys())):
        fpr[key], tpr[key], _ = roc_curve(y_test, y_score[key])
        roc_auc[key] = roc_auc_score(y_test, y_score[key])

        plt.plot(fpr[key], tpr[key],
                 label='{}: {:.4f}'.format(key, roc_auc[key]))

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic ")
    plt.legend(loc="lower right")
    plt.show()

def classification_roc_det_curves(classifiers, X, y) -> None:
    """
    Draws Receiver Operation Characteristic and Detection Error Tradeoff Curves using make_classification

    Args:
        classifiers (dict): Dictionary with name as key and classifier object as value
        X (object): make_classification return object
        y (object): make_classification return object
    """
    #N_SAMPLES = 1000
    #classifiers = {
    #    "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.025)),
    #    "Random Forest": RandomForestClassifier(
    #        max_depth=5, n_estimators=10, max_features=1
    #    ),
    #}
    #X, y = make_classification(
    #    n_samples=N_SAMPLES,
    #    n_features=2,
    #    n_redundant=0,
    #    n_informative=2,
    #    random_state=1,
    #    n_clusters_per_class=1,
    #)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    # prepare plots
    fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)

        RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=name)
        DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name)

    ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
    ax_det.set_title("Detection Error Tradeoff (DET) curves")

    ax_roc.grid(linestyle="--")
    ax_det.grid(linestyle="--")

    plt.legend()
    plt.show()

def classification_evaluation(model_name,y_true, y_pred, sample_weight=None):
    '''
    precision: tp / (tp + fp)  ability of the classifier not to label a negative sample as positive.
    recall: tp / (tp + fn) ability of the classifier to find all the positive samples.
    balanced accuracy score: [0,1] average of recall obtained on each class; best value is 1 and the worst value is 0
    accuracy score: [0,1] fraction of correct predictions.
    cohen-kappa: [-1,1] the level of agreement between two annotators; 1 means complete agreement and zero or lower means chance agreement
    MCC: [-1,1] takes into account true and false positives and negatives and is a balanced measure for imbalance cases. The MCC is in essence a correlation coefficient value.
    F1 Score: [-1,1] harmonic mean of the precision and recall.
    Hamming Loss: [0,1] fraction of labels that are incorrectly predicted.

    Args:
        model_name (str): name of the model
        y_true (np.array or pd.Series): true labels
        y_pred (np.array or pd.Series): predicted labels
        sample_weight (np.array): sample weights

    Returns:
        evaluation_dict (dict): dict of dict with model name as key and a dict of model scores as value.

    '''
    # TODO: add precision recall curve
    #   https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-precision-recall-curve-for-each-class-and-iso-f1-curves
    # TODO: add average precision recall and maybe draw the curve

    print(f'------------------------------------------{model_name}------------------------------------------')

    print('Confusion Matrix: ')
    print(confusion_matrix(y_true, y_pred, sample_weight=sample_weight))
    print('\n\nClassification Report: ')
    print(classification_report(y_true, y_pred, sample_weight=sample_weight))
    print('\n\nBalanced Accuracy Score: ', balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight))
    print('Accuracy Score: ', accuracy_score(y_true, y_pred, sample_weight=sample_weight))
    print('Cohen-Kappa Score: ', cohen_kappa_score(y_true, y_pred, sample_weight=sample_weight))
    #print('Log Loss: ', log_loss(y_true, y_score, sample_weight=sample_weight))
    print('Matthews Correlation Coefficient: ', matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight))
    print('F1 Score: ', f1_score(y_true, y_pred, sample_weight=sample_weight, average='weighted'))
    print('Hamming Loss: ', hamming_loss(y_true, y_pred, sample_weight=sample_weight))

    evaluation_dict = {
        model_name: {
        'Balanced Accuracy Score': balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        'Accuracy Score': accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        'Cohen Kappa Score': cohen_kappa_score(y_true, y_pred, sample_weight=sample_weight),
        'Matthews Corrcoef': matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight),
        'F1 Score': f1_score(y_true, y_pred, sample_weight=sample_weight, average='weighted'),
        'Hamming Loss': hamming_loss(y_true, y_pred, sample_weight=sample_weight),
        'Precision': precision_score(y_true, y_pred, sample_weight=sample_weight),
        'Recall': recall_score(y_true, y_pred, sample_weight=sample_weight)
        }
    }

    return evaluation_dict



