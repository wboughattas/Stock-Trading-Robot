import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from Models.import_data import read_files_param_grid, metadata_to_xy_trn_tst
from Models.plotting import plot_estimator_scores, plot_learning_curve, plot_conf_matrix, plot_ROC_curve, plot_PR_curve
from Models.modelling import cleanup_scores, sklearn_grid_search, custom_grid_search, score_estimators
from Models.export_data import export
from Models.Classifier_interpretability import classifier_interpretability

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # extract all filenames from read_files_param_grid in Models/import_data.py
    filenames = [read_files_param_grid[dataset_name]['filenames'] for dataset_name in read_files_param_grid.keys()]
    for filename in filenames:
        # converts metadata to a dictionary that includes X_train, X_test, y_train, y_test, headers, scaler objects, etc
        # handles datasets that are initially divided into separate train and test files
        metadata = metadata_to_xy_trn_tst(read_files_param_grid, *filename)
        # Exhaustive search over specified parameter values for each estimator by trying every combination of parameters
        estimators, parameters = custom_grid_search(metadata['X_train'], metadata['y_train'], metadata['model_params'])

        # iterates over the estimator score metrics that are specified in read_files_param_grid in Models/import_data.py
        # and runs the models that are specified in each {dataset}.py in Models/Training_parameters/
        scores = []
        for scoring_metric in metadata['estimator_scoring']:
            # runs the models and extract the train and test scores
            all_test_scores = score_estimators(metadata['X_test'], metadata['y_test'], estimators, scoring_metric,
                                               parameters, 'test')
            all_train_scores = score_estimators(metadata['X_train'], metadata['y_train'], estimators, scoring_metric,
                                                parameters, 'train')
            # transforms *args to pd.Dataframes, drops duplicated columns, and reorders them
            scores_temp = cleanup_scores(all_test_scores, all_train_scores)
            # append the unique scores to a scores list
            scores.append(scores_temp)

        # concat all scores from every model class in a dataframe and applies a cleanup
        CustomGridSearchCV_scores = cleanup_scores(*scores, comments=True)

        # GridSearchCV
        GridSearchCV_scores = sklearn_grid_search(metadata['X_test'], metadata['y_test'], GridSearchCV,
                                                  metadata['model_params'], scoring=metadata['estimator_scoring'])
        GridSearchCV_scores = cleanup_scores(GridSearchCV_scores, comments=True)

        # RandomizedSearchCV
        RandomSearchCV_scores = sklearn_grid_search(metadata['X_test'], metadata['y_test'], RandomizedSearchCV,
                                                    metadata['model_params'], scoring=metadata['estimator_scoring'])
        RandomSearchCV_scores = cleanup_scores(RandomSearchCV_scores, comments=True)

        # find the trained model that corresponds to best_estimator in read_files_param_grid
        best_estimator = [i for i in estimators
                          if i.get_params().items() >= metadata['best_estimator'].get_params().items()][0]

        # a dict of plt.Figures
        estimator_plots = {}
        if 'best_estimator_vs_parameters' in metadata['plots']:
            for scoring_metric in metadata['estimator_scoring']:
                test_scores = score_estimators(metadata['X_test'], metadata['y_test'], estimators,
                                               scoring_metric, parameters, 'test')
                train_scores = score_estimators(metadata['X_train'], metadata['y_train'], estimators,
                                                scoring_metric, parameters, 'train')
                # creates directory (if not exists), plots estimator scores, and exports to png file (with the option
                # to overwrite) into Results/{dataset_name}/Train_and_test_{scoring_metric}_VS_model_parameters
                estimator_plot = plot_estimator_scores(metadata['dataset_dirname'], metadata['y_header'],
                                                       best_estimator, estimators, train_scores, test_scores,
                                                       parameters, metadata['model_params'], scoring_metric)
                # append to dict
                estimator_plots[scoring_metric.__name__] = estimator_plot

        learning_curve = None
        if 'learning_curve' in metadata['plots']:
            for score_type in metadata['learning_curve_scoring']:
                learning_curve = plot_learning_curve(metadata['dataset_dirname'], metadata['y_header'], estimators,
                                                     metadata['X_train'],
                                                     metadata['y_train'], parameters, score_type)

        confusion_matrix = None
        if 'confusion_matrix' in metadata['plots']:
            y_labels = list(metadata['decoder'][metadata['y_header'][0]].keys())
            confusion_matrix = plot_conf_matrix(best_estimator, metadata['X_test'], metadata['y_test'], y_labels)

        ROC_curve = None
        if 'ROC_curve' in metadata['plots']:
            ROC_curve = plot_ROC_curve(best_estimator, metadata['X_test'], metadata['y_test'])

        PR_curve = None
        if 'PR_curve' in metadata['plots']:
            PR_curve = plot_PR_curve(best_estimator, metadata['X_test'], metadata['y_test'])

        # creates directory (if not exists) and exports dictionary values, being dataframes or plt.Figures, to csv or
        # png files (with the option to overwrite) into Results/{dataset_name}
        export(metadata['dataset_dirname'], {
            'CustomGridSearchCV_scores': CustomGridSearchCV_scores,
            'GridSearchCV_scores': GridSearchCV_scores,
            'RandomizedSearchCV_scores': RandomSearchCV_scores,
            **estimator_plots,
            'learning_curve': learning_curve,
            'confusion_matrix': confusion_matrix,
            'ROC_curve': ROC_curve,
            'PR_curve': PR_curve
        })

        print()  # set a breakpoint here for debugging
    print()  # set a breakpoint here for debugging

    # Run classifier interpretability model
    classifier_interpretability.initialize_ci()
