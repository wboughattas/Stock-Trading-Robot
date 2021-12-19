import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from Models import read_files_param_grid, metadata_to_xy_trn_tst, custom_grid_search, score_estimators, \
    plot_estimator_scores, cleanup_scores, export_df, sklearn_grid_search, plot_learning_curve
from Models.Classifier_interpretability import classifier_interpretability

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # extract all filenames from read_files_param_grid in Models/import_data.py
    filenames = [read_files_param_grid[dataset_name]['filenames'] for dataset_name in read_files_param_grid.keys()]
    for filename in filenames:
        # converts metadata to a dictionary that contains data that is ready for training and visualizing
        # the dictionary includes X_train, X_test, y_train, y_test, headers, scaler objects, decoders, etc...
        # handles datasets that are initially divided into separate train and test files
        metadata = metadata_to_xy_trn_tst(read_files_param_grid, *filename)
        # Exhaustive search over specified parameter values for each estimator by trying every combination of parameters
        estimators, parameters = custom_grid_search(metadata['X_train'], metadata['y_train'], metadata['model_params'])

        # iterates over the estimator score metrics that are specified in read_files_param_grid in Models/import_data.py
        # and runs the models that are specified in each {dataset}.py in Models/Training_parameters/
        scores = []
        for scoring_metric in metadata['estimator_scoring']:
            # runs the models and extract the train and test scores
            all_tst_scores = score_estimators(metadata['X_test'], metadata['y_test'], estimators, scoring_metric,
                                              parameters, 'test')
            all_trn_scores = score_estimators(metadata['X_train'], metadata['y_train'], estimators, scoring_metric,
                                              parameters, 'train')
            # # creates directory (if not exists), plots estimator scores, and exports to png file (with the option to
            # # overwrite) into Results/{dataset_name}/Train_and_test_{scoring_metric}_VS_model_parameters
            # plot_estimator_scores(metadata['dataset_dirname'], metadata['y_header'], estimators, all_trn_scores,
            #                       all_tst_scores, parameters, metadata['model_params'], scoring_metric)
            # transforms *args to pd.Dataframes, drops duplicated columns, and reorders them
            scores_temp = cleanup_scores(all_tst_scores, all_trn_scores)
            # append the unique scores to a scores list
            scores.append(scores_temp)

        # concat all scores from every model class in a dataframe and applies a cleanup
        CustomGridSearchCV_scores = cleanup_scores(*scores, comments=True)
        # creates directory (if not exists) and exports dictionary values, being dataframes, to csv files (with the
        # option to overwrite) into Results/{dataset_name}
        export_df(metadata['dataset_dirname'], {'CustomGridSearchCV_scores': CustomGridSearchCV_scores})

        # # similar to customGridSearchCV_scores, executes exhaustive/random search over specified parameter values for
        # # each estimator using scikit_learn's GridSearchCV/RandomSearchCV APIs
        # GridSearchCV_scores = sklearn_grid_search(metadata['X_test'], metadata['y_test'], GridSearchCV,
        #                                           metadata['model_params'], scoring=metadata['estimator_scoring'])
        # GridSearchCV_scores = cleanup_scores(GridSearchCV_scores, comments=True)
        #
        # RandomSearchCV_scores = sklearn_grid_search(metadata['X_test'], metadata['y_test'], RandomizedSearchCV,
        #                                             metadata['model_params'], scoring=metadata['estimator_scoring'])
        # RandomSearchCV_scores = cleanup_scores(RandomSearchCV_scores, comments=True)
        #
        # # creates directory (if not exists) and exports dictionary values, being dataframes, to csv files (with the
        # # option to overwrite) into Results/{dataset_name}
        # export_df(metadata['dataset_dirname'],
        #           {'GridSearchCV_scores': GridSearchCV_scores, 'RandomizedSearchCV_scores': RandomSearchCV_scores})
        #
        # # iterates over learning curve metrics that are specified in read_files_param_grid in Models/import_data.py
        # for score_type in metadata['learning_curve_scoring']:
        #     # creates directory (if not exists), plots estimator scores, and exports to png file (with the option to
        #     # overwrite) into Results/{dataset_name}/{list(scoring_metric)}_VS_Training_size
        #     plot_learning_curve(metadata['dataset_dirname'], metadata['y_header'], estimators, metadata['X_train'],
        #                         metadata['y_train'], parameters, score_type)

        print()  # set a breakpoint here for debugging
    print()  # set a breakpoint here for debugging

    # Run classifier interpretability model
    # classifier_interpretability.initialize_ci()

    # todo: handle statlog (cost matrix)
    # todo: handle yeast (multiclass)
    # todo: get the right params
    # todo: comment each line of code :(
