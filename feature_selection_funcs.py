import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score


import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist_subplot(df, ncols=2, figsize=[10,8]):
    """
    Plots histograms of df columns in subplot format.

    :param df: DataFrame of columns to be plotted
    :type df: pd.DataFrame
    :param ncols: number of columns in the plot, default is 2
    :type ncols: int
    :param figsize: figure size of the plot, default is [10,8]
    :type figsize: np.array
    """
    num_features=df.columns
    fig, ax = plt.subplots(nrows=int(np.ceil(len(num_features) / ncols)), ncols=ncols, figsize=figsize)
    for i in range(len(num_features)):
        sns.distplot(x=df[num_features[i]], ax=ax.flat[i])
    plt.show()

def vif_for_regression(train_X, test_X, train_y, test_y, vif_score=5):
    """
    Used for Feature Seletion.
    Calculates VIF for each column in feature space using Linear Regression.
    Removes features with VIF higher than 5, one by one and plots all the RMSE values obtained in each iteration.

    :param train_X: standardized train set feature space
    :type train_X: pd.DataFrame
    :param test_X: standardized test set feature space
    :type test_X: pd.DataFrame
    :param train_y: target training values
    :type train_y: pd.Series
    :param test_y: target test values
    :type test_y: pd.Series

    :return: list of columns to be dropped
    :rtype: list
    """
    train_rmse = [];
    test_rmse = [];

    DROP = [];

    for i in range(len(train_X.columns)):
        vif = pd.DataFrame()
        X = train_X.drop(DROP, axis=1)
        vif['Features'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['VIF'] = round(vif['VIF'], 2)
        vif = vif.sort_values(by="VIF", ascending=False)
        vif.reset_index(drop=True, inplace=True)
        if vif.loc[0][1] > vif_score:
            DROP.append(vif.loc[0][0])
            LR = LinearRegression()
            LR.fit(train_X.drop(DROP, axis=1), train_y)

            pred1 = LR.predict(train_X.drop(DROP, axis=1))
            pred2 = LR.predict(test_X.drop(DROP, axis=1))

            train_rmse.append(np.sqrt(mean_squared_error(train_y, pred1)))
            test_rmse.append(np.sqrt(mean_squared_error(test_y, pred2)))

    print('Dropped Features --> ', DROP)

    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(test_rmse, label='Test RMSE')
    plt.xlabel('Number of Dropped Features')
    plt.ylabel('RMSE')
    # plt.ylim([19.75,20.75])
    plt.legend()
    plt.grid()
    plt.show()

    return DROP

def rfe_for_regression(X_train_std, X_test_std, y_train, y_test):
    """
    Used for Feature Selection.
    Performs Recursive Feature Elimination on the dataset using Linear Regression
    and plots RMSE values obtained in each iteration.

    :param X_train_std: standardized train set feature space
    :type X_train_std: pd.DataFrame
    :param X_test_std: standardized test set feature space
    :type X_test_std: pd.DataFrame
    :param y_train: training target values
    :type y_train: pd.Series
    :param y_test: test target values
    :type y_test: pd.Series
    :return: DF containing feature set combinations and respective train RMSE and test RMSE values
    :rtype: pd.DataFrame
    """
    Trr = [];
    Tss = [];

    feature_space_df = pd.DataFrame(columns=['features', 'RMSE_train', 'RMSE_test'])

    m = X_train_std.shape[1] - 2
    for i in range(m):
        lm = LinearRegression()
        rfe = RFE(lm, n_features_to_select=X_train_std.shape[1] - i)  # running RFE
        rfe = rfe.fit(X_train_std, y_train)

        LR = LinearRegression()
        LR.fit(X_train_std.loc[:, rfe.support_], y_train)

        pred1 = LR.predict(X_train_std.loc[:, rfe.support_])
        pred2 = LR.predict(X_test_std.loc[:, rfe.support_])

        Trr.append(np.sqrt(mean_squared_error(y_train, pred1)))
        Tss.append(np.sqrt(mean_squared_error(y_test, pred2)))

        feature_space_df = feature_space_df.append(
            {'features': X_train_std.columns[rfe.support_].values,
             'RMSE_train': np.sqrt(mean_squared_error(y_train, pred1)),
             'RMSE_test': np.sqrt(mean_squared_error(y_test, pred2))},
            ignore_index=True)

    plt.plot(Trr, label='Train RMSE')
    plt.plot(Tss, label='Test RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    # plt.ylim([19.75,20.75])
    plt.legend()
    plt.grid()
    plt.show()

    return feature_space_df

def pca_for_regression(X_train_std, X_test_std, y_train, y_test):
    """
    Used for Feature Selection/Reduction.
    Iteratively reduces the dimension and fits a Linear Regression Model to the PCA feature set.

    :param X_train_std: standardized train set feature space
    :type X_train_std: pd.DataFrame
    :param X_test_std: standardized test set feature space
    :type X_test_std: pd.DataFrame
    :param y_train: training target values
    :type y_train: pd.Series
    :param y_test: test target values
    :type y_test: pd.Series
    """
    Trr = [];
    Tss = [];

    m = X_train_std.shape[1] - 1

    for i in range(m):
        pca = PCA(n_components=X_train_std.shape[1] - i)
        X_train_std_pca = pca.fit_transform(X_train_std)
        X_test_std_pca = pca.fit_transform(X_test_std)

        LR = LinearRegression()
        LR.fit(X_train_std_pca, y_train)

        pred1 = LR.predict(X_train_std_pca)
        pred2 = LR.predict(X_test_std_pca)

        Trr.append(round(np.sqrt(mean_squared_error(y_train, pred1)), 2))
        Tss.append(round(np.sqrt(mean_squared_error(y_test, pred2)), 2))

    plt.plot(Trr, label='Train RMSE')
    plt.plot(Tss, label='Test RMSE')
    plt.xlabel('Number of components')
    plt.ylabel('RMSE')
    # plt.ylim([19.5,20.75])
    plt.legend()
    plt.grid()
    plt.show()

def rfeCV_for_regression(model, X_train, y_train, cv=5, scoring=None):
    # TODO: add func docstring

    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(X_train, y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(min_features_to_select, len(rfecv.cv_results_) + min_features_to_select),
        rfecv.grid_scores_,
    )
    plt.show()

    # TODO: add plot of feature importance with sorted rfecv.ranking_ (greater the number less important the feature)
    #   https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV
    #   https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

    return rfecv


