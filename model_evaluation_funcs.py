import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import permutation_test_score
from sklearn.dummy import DummyRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    explained_variance_score, mean_absolute_percentage_error, mean_squared_log_error

import matplotlib.pyplot as plt
import seaborn as sns

def regression_evaluation(model_name, y_test, y_test_pred, y_train=None, y_train_pred=None):

    # TODO: add func docstring

    print('----------------------------------------',model_name,'----------------------------------------')
    if y_train is not None and y_train_pred is not None:
        print('-----Train Set Metrics-----')
        print('R-squared                         : ', r2_score(y_train, y_train_pred))
        print('Root Mean Squared Error                : ', np.sqrt(mean_squared_error(y_train, y_train_pred)))
        print('Explained Variance Score          : ', explained_variance_score(y_train, y_train_pred))
        print('Mean Absolute Error               : ', mean_absolute_error(y_train, y_train_pred))
        print('Mean Squared Error                : ', mean_squared_error(y_train, y_train_pred))
        print('Mean Absolute Percentage Error    : ', mean_absolute_percentage_error(y_train, y_train_pred))
        print('Mean Squared Logarithmic Error    : ', mean_squared_log_error(y_train, y_train_pred), '\n')

        score_df = pd.DataFrame({
            'train_R2': r2_score(y_train, y_train_pred),
            'train_EVS': explained_variance_score(y_train, y_train_pred),
            'train_MAE': mean_absolute_error(y_train, y_train_pred),
            'train_MSE': mean_squared_error(y_train, y_train_pred),
            'train_MAPE': mean_absolute_percentage_error(y_train, y_train_pred),
            'train_MSLE': mean_squared_log_error(y_train, y_train_pred)
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
        plt.title('Test vs Prediction')
        plt.xlabel('y_test')
        plt.ylabel('y_pred')
        plt.show()

    print('-----Test Set Metrics-----')

    # R-squared
    r2 = r2_score(y_test, y_test_pred)
    print('R-squared                            : ', r2)

    # Explained Variance Score
    evs = explained_variance_score(y_test, y_test_pred)
    print('Explained Variance Score             : ', evs)

    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_test_pred)
    print('Mean Absolute Error                  : ', mae)

    # Mean Squared Error
    mse = mean_squared_error(y_test, y_test_pred)
    print('Mean Squared Error                   : ', mse)

    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_test, y_test_pred)
    print('Mean Absolute Percentage Error       : ', mape)

    # Mean Squared Logarithmic Error
    msle = mean_squared_log_error(y_test, y_test_pred)
    print('Mean Squared Logarithmic Error       : ', msle, '\n')

    score_df['test_R2'] = r2
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
