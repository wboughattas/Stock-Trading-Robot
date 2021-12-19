from sklearn.metrics import make_scorer, f1_score
from itertools import islice
import pandas as pd
import itertools
import sys


def sklearn_grid_search(X, y, search_type, param_grid, scoring, **kwargs):
    """
    Search over specified parameter values for every estimator. It runs either GridSearchCV or RandomizedSearchCV for
    every estimator
    :param scoring: a sklearn scoring metric (e.g accuracy_score)
    :param X: X_train
    :param y: y_train
    :param search_type: GridSearchCV or RandomizedSearchCV
    :param param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values
    :param kwargs: attributes corresponding to search_type
    :return: a dataframe with scores and params per model
    """

    dfs = []

    print('Running sklearn_grid_search')

    # f1_score is excluded because we cannot set average = 'weighted'
    estimator_scoring = {i.__name__: make_scorer(i) for i in scoring if i is not f1_score}

    # run a GridSearchCV or RandomizedSearchCV for every model class
    for model_name, val in param_grid.items():
        clf = search_type(val['model'], val['params'], refit=False, return_train_score=True, scoring=estimator_scoring,
                          **kwargs)
        clf.fit(X, y)

        # turn np array to pd.Dataframe
        results = pd.DataFrame(clf.cv_results_)
        # keep the columns with keywords 'mean' and 'score'
        scores_columns = [i for i in list(clf.cv_results_.keys()) if 'mean' in i and 'score' in i and 'time' not in i]
        # Since gridSearch params columns includes all parameters that we don't need for analysis (with initialized
        # values by sklearn), a more condensed column of params in scope is made
        results = results[['params', *scores_columns]]
        results = results.astype({'params': str})
        results['params'] = results['params'].replace({'{': '', '}': '', "'": ''}, regex=True)
        # a class name is repeated as many times as the number of models
        results['model'] = [val['model'].__class__.__name__] * len(results)

        dfs.append(results)

    # concatenate the scores of each model in a dataframe and fill NAs with 0
    df = pd.concat(dfs).fillna(0)

    # last column becomes first
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    return df


def score_estimators(X, y, estimators, score_metric, custom_params, type_reference):
    """
    calculates scores and params per model. the output of this function fully resembles sklearn_grid_search() output
    :param type_reference: 'train' or 'test' to distinguish in the dataframe
    :param score_metric: https://scikit-learn.org/stable/modules/model_evaluation.html
    :param custom_params: a dictionary of estimator class names as keys and list of parameter names as values
    :param X: X_train/test
    :param y: y_train/test
    :param estimators: list of sklearn trained models
    :return: a dataframe with scores and params per model
    """

    results = []
    # turn estimators into a list if it's one instance
    estimators = [estimators] if not isinstance(estimators, list) else estimators
    for index, estimator in enumerate(estimators):
        try:
            score = abs(score_metric(y, estimator.predict(X)))
        except ValueError as exp:
            print(exp)
            print('A weighted setting has been used')
            try:
                # multiclass datasets (e.g. Yeast) require to specify the average of f1_score
                score = abs(score_metric(y, estimator.predict(X), average='weighted'))
            except (TypeError, ValueError) as exp2:
                # certain MLP models fail for choosing contradicting parameters and estimator.predict(X) includes NANs
                # their corresponding score will be 0
                print(exp2)
                print('score equals 0')
                score = 0

        # make a dictionary of scores and params that will be converted to a pd.Dataframe
        # the output of this function fully resembles sklearn_grid_search() output
        params = {
            custom_params[estimator.__class__.__name__][idx]: estimator.get_params()[model_name] for
            idx, model_name in enumerate(custom_params[estimator.__class__.__name__])
        }
        params_formatted = ', '.join('{}: {}'.format(key, value) for key, value in params.items())
        estimator_info = {'model': estimator.__class__.__name__,
                          '_'.join(['mean', type_reference, score_metric.__name__]): score,
                          'params': estimator.get_params() if custom_params is None else params_formatted}
        results.append(pd.DataFrame(estimator_info, index=[0]))
    df = pd.concat(results).fillna(0)
    return df


def convert(listA, len_2d):
    """
    convert 1d list to 2d given a list of indices
    :param listA: targeted 1d list
    :param len_2d: list of indices
    :return: 2d list
    """
    res = iter(listA)
    return [list(islice(res, i)) for i in len_2d]


def groupby_type(estimators):
    """
    finds the number of estimators for each estimator class
    :param estimators: list of estimators (not necessarily trained)
    :return: a dictionary of estimator class as key and frequency as value
    """
    unique_classes = {}
    for estimator in estimators:
        clf_name = estimator.__class__.__name__
        if clf_name not in list(unique_classes.keys()):
            unique_classes[clf_name] = 0
        unique_classes[clf_name] += 1
    return unique_classes


def cleanup_scores(*args, comments=False):
    """
    remove duplicate columns, concatenate args (being pd.Dataframes), round floats, reset_index column, and if comments
    are enables, creates new columns that rank the estimator per estimator class per scoring metric
    :param args: score pd.Dataframes
    :param comments: (bool) creates new columns that rank the estimator per estimator class per scoring metric
    :return: cleaned dataframe
    """

    # make copy of unique column named 'model'
    model_col = pd.DataFrame(args[0]['model']).iloc[:, 0]
    # make copy of unique column called 'params'
    params_col = pd.DataFrame(args[0]['params']).iloc[:, 0]
    # drop all instances of the columns named 'model' and 'params'
    df = pd.concat(([i.drop(['model', 'params'], axis=1) for i in args]), axis=1)
    # round floats
    df = df.round(2)
    # add the copies of the unique columns 'model' and 'params'
    df = pd.concat([model_col, params_col, df], axis=1).fillna(0)
    df.reset_index(drop=True, inplace=True)

    if comments is True:
        mean_test_columns = [i for i in list(df.columns) if 'mean_test' in i]
        for i in mean_test_columns:
            # group df by 'model' and find max score per scoring metric
            large_score_est = df.loc[df.groupby(['model'])[i].idxmax()]
            large_score_indices = large_score_est.index
            new_column = i.replace('mean', 'ranking')
            df[new_column] = ['-']*len(df)
            df[new_column].loc[large_score_indices] = 'highest of its class'

            # group df by 'model' and find min score per scoring metric
            small_score_est = df.loc[df.groupby(['model'])[i].idxmin()]
            small_score_indices = small_score_est.index
            df[new_column].loc[small_score_indices] = 'lowest of its class'
    return df


def custom_grid_search(X, y, search_grid):
    """
    Exhaustive search over specified parameter values for an estimator. Tries every combination of parameters.
    Similar to sklearn_grid_search()
    :param X: X_train
    :param y: y_train
    :param search_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values
    :return: a dataframe with best_score, and best_params per model and a list of parameters
    """

    clf = []
    lst_lst_param_keys = {}
    for estimator_name, model_params in search_grid.items():
        lst_param_values, lst_param_keys = [], []
        for param_keys, param_val in model_params['params'].items():
            try:
                if all(isinstance(n, (int, float)) for n in param_val):
                    if not all(param_val[i] <= param_val[i + 1] for i in range(len(param_val) - 1)):
                        raise ValueError('Numerical values must be increasing')
            except ValueError as exp:
                print("\nError", exp)
                sys.exit()
            lst_param_values.append(param_val)
            lst_param_keys.append(param_keys)
        lst_lst_param_keys[model_params['model'].__class__.__name__] = lst_param_keys
        # make unique combinations of params
        unique_combinations = list(itertools.product(*lst_param_values))
        for val in unique_combinations:
            params = {lst_param_keys[i]: val[i] for i in range(len(lst_param_keys))}
            class_ = model_params['model'].__class__
            # get all params of estimator that are initialized in 'model'
            # which are to be used in all new estimators regardless of their combinations
            initial_attributes = model_params['model'].get_params()
            # combine initialized params and each new combination
            model = class_(**{**initial_attributes, **params})
            print('Training {}'.format(model))
            try:
                model.fit(X, y)
                clf.append(model)
            except ValueError as exp2:
                # wrong parameters in the model param_grid
                print("An error was encountered while running this model: ", exp2)
                print("Must exclude this model from the param_grid: ", model)
                sys.exit()
    return clf, lst_lst_param_keys
