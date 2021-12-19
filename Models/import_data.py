import csv
import os
import sklearn
from sklearn.tree import DecisionTreeClassifier
from Models.Training_parameters import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xlrd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from __init__ import ROOT_DIR


def files_to_metadata(param_grid):
    """Reads the csv files in current directory and extracts dataset, X, X_train, X_test, y, y_train, y_test,
    dataset_header, X_header, y_header, decoder, scaler object, scoring metrics, and types of plots for each Data file
    and transform them according to reading parameters that are specific to each dataset

    :param param_grid: Dictionary with dataset names (str) as keys corresponding to the name of directory that contains
    the filename(s). and dictionaries of parameter settings to read and edit the corresponding dataset as values. The
    param setting are: filename(s) (list) including the file extension, header (str): None if header is within dataset,
    delimiter (str), y (list): name of y column(s), skip_columns (int), train_size (float), split_random_state (int),
    and 'included_test_train' (bool): True if the provided files are test and train datasets. XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    :return: Dictionary with filenames (str) as keys and dictionaries of metadata. The metadata include dataset, X,
    X_train, X_test, y, y_train, y_test, header, X_header, y_header, and decoder, scaler object, scoring metrics,
    and types of plots

    :Reasoning for not opting for traditional numpy.loadtxt() or pandas.read_csv():
    1. flexible parameters using a param_grid (similar idea to GridSearchCV's param_grid)
    2. automate reading files
    3. very easy access to: dataset, X, X_train, X_test, y, y_train, y_test, dataset_header, X_header, y_header,
    and decoder by indexing the output of files_to_metadata (dict)
    4. easy to debug
    """

    # get data folder
    dir_abspath = os.path.join(ROOT_DIR, 'Data')
    datasets, headers, decoders = [], [], []
    metadata_grid = {}
    dataset_count = 0

    for key, value in param_grid.items():
        # a list of a list of filename. The inner list is unique to each dataset
        # The inner list can contain train and test files
        path_names = [os.path.join(dir_abspath, key, i) for i in value['filenames']]
        for file_idx, file in enumerate(path_names):
            filename = os.path.basename(str(file))
            print('Reading {}'.format(filename))
            # utf-8-sig encoding used to treat BOM as file info, instead of string
            with open(file, newline='', encoding='utf-8-sig') as datafile:
                # exception for excel file
                if os.path.splitext(file)[1] == '.xls':
                    # access first sheet
                    sheet = xlrd.open_workbook(file).sheet_by_index(0)
                    # create an iterator of rows
                    reader = iter([sheet.row_values(row) for row in range(sheet.nrows)])
                else:
                    # All text files can be read using csv module except for excel files
                    reader = csv.reader(datafile, skipinitialspace=True, delimiter=value['delimiter'])
                # skip n rows in the iterator
                [next(reader, None) for _ in range(value['skip_rows'][file_idx])]
                if value['header'] is None:
                    # save headers and move to next iteration
                    header = next(reader)
                else:
                    # open file that contains column headers
                    header_file_name = os.path.join(dir_abspath, key, value['header'])
                    with open(header_file_name, newline='', encoding='utf-8-sig') as header_file:
                        # header is saved as list
                        header = [j for sub in list(csv.reader(header_file)) for j in sub]
                data = []
                decoder = {}
                # iterate by rows
                for i, row in enumerate(reader):
                    lst = []  # saves the modified row (and its values)
                    col_idx = 0
                    # iterate by columns (aka values across the rows)
                    for s in row:
                        try:
                            # if number, convert to float and append to the list. if string, throw exception
                            lst.append(float(s))
                        except (Exception,):  # value is not a number
                            if s in ['', '?']:
                                # replace empty/unknown values with float Nan
                                lst.append(float('Nan'))
                                col_idx += 1
                                # move on to the next column
                                continue
                            # ENCODING starts here
                            # keep track of the the header that corresponds to the current column
                            col_idx_str = header[col_idx]
                            # create empty decoder if none of the strings across current column are not decoded
                            if col_idx_str not in decoder.keys():
                                decoder[col_idx_str] = {}
                            # get column decoder keys and values
                            existing_original_str = decoder[col_idx_str].keys()
                            existing_decoded_str = decoder[col_idx_str].values()
                            # current string (or similar equivalent string) is not encoded
                            if str(s) not in existing_original_str:
                                if len(existing_original_str) > 0:
                                    # next string is encoded to a the preceding encoder value + 1
                                    decoded_val = max(existing_decoded_str) + 1
                                else:
                                    # no encoded value exists, then current string is encoded to 0
                                    decoded_val = 0
                                # save encoded value
                                decoder[col_idx_str][str(s)] = decoded_val
                                # add encoded value to new row
                                lst.append(decoded_val)
                            # current string (or similar equivalent string) is not encoded
                            else:
                                lst.append(decoder[col_idx_str][str(s)])
                        col_idx += 1
                    # append the new temp row to final dataset (2d list)
                    data.append(lst) if len(lst) != 0 else None

                # multiple datasets, headers, decoders exist because function can handle reading and transforming
                # a multitude of datasets
                datasets.append(data)
                headers.append(header)
                decoders.append(decoder)

            head = headers[dataset_count]
            decode = decoders[dataset_count]
            # indices of columns to skip in dataset (specified by user)
            skip_columns_indices = [head.index(i) for i in value['skip_columns']]
            # convert 2d data to np.array
            dataset = np.array(datasets[dataset_count])
            # delete the columns that are specified to be skipped
            dataset = np.delete(dataset, skip_columns_indices, axis=1)

            # impute Nans with mean values
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            dataset = imp_mean.fit_transform(dataset)

            # incorporate the deleted columns to their header list
            head = [head[i] for i in range(len(head)) if i not in skip_columns_indices]

            # if dataset has multiple target columns (specified by user)
            if len(value['y']) != 1:
                y_indices = [head.index(i) for i in value['y']]
                # take a mean of the columns' values along each row (N columns are transformed to 1 mean column)
                target_mean = dataset[:, y_indices].mean(axis=1).reshape((-1, 1))
                # delete the target columns
                dataset = np.delete(dataset, y_indices, axis=1)
                # add the mean column to the data table
                dataset = np.append(dataset, target_mean, axis=1)

                # repeat the changes to the headers list
                head = [head[i] for i in range(len(head)) if i not in y_indices]
                head.append('mean_target')
                value['y'] = ['mean_target']
                head[y_indices[0]] = 'mean_target'

            # get X, y indices
            y_indices = [head.index(i) for i in value['y']]
            X_indices = [idx for idx in range(len(head)) if idx not in y_indices]

            # get X, y column headers
            y_header = [head[i] for i in y_indices]
            X_header = [head[i] for i in X_indices]

            # get X, y
            y = dataset[:, y_indices]
            X = dataset[:, X_indices]

            y = y.reshape(-1)  # reshape y to 1d array
            if all([i.is_integer() for i in y]):
                # convert floats that have no decimals to int (and keep floats with decimals as floats)
                y = y.astype(int)

            # get unique y values
            unique_y = np.unique(y)
            # Since there are 3 types of estimating: binary-classification (CL), multi-classification (CL), regression
            # this block of code encodes the binary-CL target values to 0s and 1s (e.g.: german statlog: 1->0, 2->1)
            if len(unique_y) == 2 and not all(unique_y == np.array([0, 1])):
                # positive label is the positive value. None means dataset is not binary-CL (specified by user)
                if value['positive_label'] is not None:
                    # since 'filenames' variable accept a list of separate train and test files, 'positive-label' must
                    # the positive value from each file
                    if all(isinstance(value['positive_label'][i], (int, float)) for i in range(len(path_names))):
                        # get the positive label for the unique file
                        pos_label = value['positive_label'][file_idx]
                        # negative value is the other y value beside the positive value (only works for binary-CL)
                        neg_label = np.delete(unique_y, np.where(unique_y == pos_label))[0]
                        # mapping/encoding
                        map_func = lambda x: 1 if x == pos_label else 0
                        y = [map_func(i) for i in y]

                        # save encoding to the decoder
                        decode[value['y'][0]] = {}
                        decode[value['y'][0]][neg_label] = 0
                        decode[value['y'][0]][pos_label] = 1

            # update the dataset with the newly modified target values
            dataset[:, y_indices[0]] = y
            scaler = StandardScaler()

            # user must specify (T,F) if test and train files are separately included
            if value['included_test_train'] is False:
                # split dataset to train and test given train_size (usually specified by user as 0.75),
                # stratify (usually equals y values if binary-classification to evenly split y; specified by user)
                # and random state for reproducibility (specified by user)
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=value['train_size'],
                                                                    stratify=y if value['stratify'] is True else None,
                                                                    random_state=value['split_random_state'])
                # scale all X values; fitted on the train values
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            else:
                # no splitting for existing train and test files
                X_train, X_test, y_train, y_test = None, None, None, None

            # name of the directory holding the dataset files
            filename = os.path.basename(str(filename))
            # see the param_grid at the bottom of this file for examples
            metadata_grid[filename] = {
                'dataset': dataset,
                'X': X,
                'y': y,
                'y_train': y_train,
                'y_test': y_test,
                'X_train': X_train,
                'X_test': X_test,
                'header': head,
                'X_header': X_header,
                'y_header': y_header,
                'decoder': decode,
                'scaler': scaler,
                'estimator_scoring': value['estimator_scoring'],
                'learning_curve_scoring': value['learning_curve_scoring'],
                'plots': value['plots'] if 'plots' in value else [],
                'best_estimator': value['best_estimator']
            }
            dataset_count += 1

    return metadata_grid


def metadata_to_xy_trn_tst(param_grid, *filenames):
    """
    handle datasets with separate train and test files (adult.data, adult.test)
    Future iterations of the code will aim at the dividing the roles of files_to_metadata() and metadata_to_xy_trn_tst()
    since the former function already transforms the data to xy_trn_tst

    :param param_grid: Dictionary with dataset names (str) as keys corresponding to the name of directory that contains
    the filename(s). and dictionaries of parameter settings to read and edit the corresponding dataset as values. The
    param setting are: filename(s) (list) including the file extension, header (str): None if header is within dataset,
    delimiter (str), y (list): name of y column(s), skip_columns (int), train_size (float), split_random_state (int),
    and 'included_test_train' (bool): True if the provided files are test and train datasets.

    :param filenames: name of file(s), at most two, that contains the dataset. If >1, Files must be a train and test
    datasets respectively

    :return X_train, X_test, y_train, y_test, X_header, y_header, decoder, scaler object, scoring metrics, types of
    plots, and extra model_params of the dataset
    """

    Data_abspath = os.path.join(ROOT_DIR, 'Data')
    dataset_dirname = None

    # walk the directories to find the data file
    for root, dirs, files in os.walk(Data_abspath):
        for file in files:
            if file == filenames[0]:
                # save the directory name that contains the data file
                dataset_dirname = os.path.basename(root)

    # find the value that stores the model parameters corresponding the dataset in-hand
    model_params = globals()['{}_params'.format(dataset_dirname)]
    # create a dictionary with one value to ingest it to the function files_to_metadata()
    import_param = {dataset_dirname: param_grid[dataset_dirname]}
    METADATA_S = files_to_metadata(import_param)
    # handle train and test separate files
    metadata = METADATA_S[filenames[0]]

    header, X_header, y_header, scaler, estimator_scoring, learning_curve_scoring, plots, best_estimator = \
        metadata['header'], metadata['X_header'], metadata['y_header'], metadata['scaler'], \
        metadata['estimator_scoring'], metadata['learning_curve_scoring'], metadata['plots'], metadata['best_estimator']

    # no separate train and test
    if len(filenames) == 1:
        metadata = METADATA_S[filenames[0]]
        dataset = metadata['dataset']
        X = metadata['X']
        y = metadata['y']
        y_train = metadata['y_train']
        y_test = metadata['y_test']
        X_train = metadata['X_train']
        X_test = metadata['X_test']
        decoder = metadata['decoder']
    else:
        # train dataset as np array
        train_metadata = METADATA_S[filenames[0]]
        # test dataset as np array
        test_metadata = METADATA_S[filenames[1]]

        X_train, X_test = train_metadata['X'], test_metadata['X']
        y_train, y_test = train_metadata['y'], test_metadata['y']

        # not used for modelling, used for debugging
        # can be scaled to handle splits with different train and test sizes
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=None)

        # scaled and fitted on train data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        dataset = np.append(X, y.reshape((-1, 1)), axis=1)
        # decoder is 2d. can't use it as label for confusionMatrix
        decoder = [train_metadata['decoder'], test_metadata['decoder']]

    output = {
        'dataset': dataset,
        'X': X,
        'y': y,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test,
        'header': header,
        'X_header': X_header,
        'y_header': y_header,
        'decoder': decoder,
        'scaler': scaler,
        'estimator_scoring': estimator_scoring,
        'learning_curve_scoring': learning_curve_scoring,
        'model_params': model_params,
        'dataset_dirname': dataset_dirname,
        'plots': plots,
        'best_estimator': best_estimator
    }

    return output


read_files_param_grid = {
    'diabetic_retinopathy': {
        'filenames': ['messidor_features.arff'],
        'header': 'columns',
        'delimiter': ',',
        'y': ['class label'],
        'skip_columns': [],
        'skip_rows': [24],
        'train_size': 0.75,
        'split_random_state': 0,
        'stratify': True,
        'included_test_train': False,
        'estimator_scoring': [accuracy_score, precision_score],
        'learning_curve_scoring': [[log_loss]],
        'positive_label': [1],
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.svm.SVC(C=1, kernel='poly', gamma='scale', coef0=30, random_state=0)
    },
    'credit_default': {
        'filenames': ['default of credit card clients.xls'],
        'header': None,
        'delimiter': None,
        'y': ['default payment next month'],
        'skip_columns': ['ID'],
        'skip_rows': [1],
        'train_size': 0.75,
        'split_random_state': 0,
        'stratify': True,
        'included_test_train': False,
        'estimator_scoring': [accuracy_score, precision_score],
        'learning_curve_scoring': [[log_loss]],
        'positive_label': [1],
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.ensemble.AdaBoostClassifier(algorithm='SAMME', n_estimators=50, learning_rate=0.1,
                                                              random_state=0)
    },
    'breast_cancer': {
        'filenames': ['breast-cancer-wisconsin.data'],
        'header': 'columns',
        'delimiter': ',',
        'y': ['Class'],
        'skip_columns': ['Sample code number'],
        'skip_rows': [0],
        'train_size': 0.75,
        'split_random_state': 0,
        'stratify': True,
        'included_test_train': False,
        'estimator_scoring': [accuracy_score, precision_score],
        'learning_curve_scoring': [[log_loss]],
        'positive_label': [4],
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.linear_model.LogisticRegression(fit_intercept=True, penalty='none', C=1.0,
                                                                  random_state=0, max_iter=10000)
    },
    'german_credit': {
        'filenames': ['german.data-numeric'],
        'header': 'columns.data-numeric',
        'delimiter': ' ',
        'y': ['target'],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'split_random_state': 0,
        'stratify': True,
        'included_test_train': False,
        'estimator_scoring': [accuracy_score, recall_score, average_precision_score, roc_auc_score],
        'learning_curve_scoring': [[log_loss]],
        'positive_label': [2],
        'plots': ['best_estimator_vs_parameters', 'learning_curve', 'ROC_curve', 'PR_curve'],
        'best_estimator': sklearn.linear_model.LogisticRegression(fit_intercept=True, penalty='none', C=10,
                                                                  random_state=0, solver='sag',
                                                                  class_weight={0: 1., 1: 5.})
    },
    'adult': {
        'filenames': ['adult.data', 'adult.test'],  # start with the train file because train_standard_scaler is used
        'header': 'columns',
        'delimiter': ',',
        'y': ['yearly salary'],
        'skip_columns': [],
        'skip_rows': [0, 1],
        'train_size': None,
        'split_random_state': 0,
        'stratify': True,
        'included_test_train': True,
        'estimator_scoring': [accuracy_score, precision_score],
        'learning_curve_scoring': [[log_loss]],
        'positive_label': ['>50K', '>50K.'],
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.ensemble.RandomForestClassifier(max_depth=9, n_estimators=100, criterion='gini',
                                                                  random_state=0)
    },
    # multiclass
    'yeast': {
        'filenames': ['yeast.data'],
        'header': 'columns',
        'delimiter': ' ',
        'y': ['location'],
        'skip_columns': ['Sequence Name'],
        'skip_rows': [0],
        'train_size': 0.75,
        'split_random_state': 0,
        'stratify': True,
        'included_test_train': False,
        'estimator_scoring': [accuracy_score, f1_score],
        'learning_curve_scoring': [[accuracy_score]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'confusion_matrix'],
        'best_estimator': sklearn.ensemble.RandomForestClassifier(max_depth=12, n_estimators=100, criterion='entropy',
                                                                  max_features='auto', random_state=0)
    },
    'thoracic_surgery': {
        'filenames': ['ThoraricSurgery.arff'],
        'header': 'columns',
        'delimiter': ',',
        'y': ['Risk1Yr'],
        'skip_columns': [],
        'skip_rows': [21],
        'train_size': 0.75,
        'split_random_state': 0,
        'stratify': True,
        'included_test_train': False,
        'estimator_scoring': [accuracy_score, precision_score],
        'learning_curve_scoring': [[log_loss]],
        'positive_label': ['T'],
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.svm.SVC(C=0.5, kernel='poly', max_iter=100, random_state=0)
    },
    'seismic_bumps': {
        'filenames': ['seismic-bumps.arff'],
        'header': 'columns',
        'delimiter': ',',
        'y': ['class'],
        'skip_columns': [],
        'skip_rows': [154],
        'train_size': 0.75,
        'split_random_state': 0,
        'stratify': True,
        'included_test_train': False,
        'estimator_scoring': [accuracy_score, precision_score],
        'learning_curve_scoring': [[log_loss]],
        'positive_label': [1],
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.ensemble.AdaBoostClassifier(algorithm='SAMME.R', n_estimators=200, learning_rate=0.1,
                                                              random_state=0)
    },
    'wine_quality_white': {
        'filenames': ['winequality-white.csv'],
        'header': None,
        'delimiter': ';',
        'y': ['quality'],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'split_random_state': 0,
        'stratify': False,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.ensemble.RandomForestRegressor(criterion='squared_error', max_depth=50,
                                                                 n_estimators=200, max_features='auto', random_state=0)
    },
    'wine_quality_red': {
        'filenames': ['winequality-red.csv'],
        'header': None,
        'delimiter': ';',
        'y': ['quality'],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'stratify': False,
        'split_random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=50, max_iter=100, random_state=0)
    },
    'crime_predict': {
        'filenames': ['communities.data'],
        'header': 'communities.names',
        'delimiter': ',',
        'y': ['ViolentCrimesPerPop numeric'],
        'skip_columns': ['state numeric', 'county numeric', 'community numeric', 'communityname string',
                         'fold numeric'],
        'skip_rows': [0],
        'train_size': 0.75,
        'stratify': False,
        'split_random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.ensemble.RandomForestRegressor(criterion='squared_error', max_depth=50,
                                                                 n_estimators=200, max_features='auto', random_state=0)
    },
    'aquatic_toxicity': {
        'filenames': ['qsar_aquatic_toxicity.csv'],
        'header': 'columns.csv',
        'delimiter': ';',
        'y': ['quantitative response'],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'stratify': False,
        'split_random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.svm.SVR(C=10, gamma='auto', kernel='rbf')
    },
    'facebook_metrics': {
        'filenames': ['dataset_Facebook.csv'],
        'header': None,
        'delimiter': ';',
        'y': ['Total Interactions'],
        'skip_columns': ['Lifetime Post Total Reach', 'Lifetime Post Total Impressions', 'Lifetime Engaged Users',
                         'Lifetime Post Consumers', 'Lifetime Post Consumptions',
                         'Lifetime Post Impressions by people who have liked your Page',
                         'Lifetime Post reach by people who like your Page',
                         'Lifetime People who have liked your Page and engaged with your post', 'comment', 'like',
                         'share'],
        'skip_rows': [0],
        'train_size': 0.75,
        'stratify': False,
        'split_random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.ensemble.RandomForestRegressor(criterion='squared_error', max_depth=50,
                                                                 n_estimators=100, max_features='auto', random_state=0)
    },
    'bike_sharing': {
        'filenames': ['hour.csv'],
        'header': None,
        'delimiter': ',',
        'y': ['cnt'],
        'skip_columns': ['instant'],
        'skip_rows': [0],
        'train_size': 0.75,
        'stratify': False,
        'split_random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.tree.DecisionTreeRegressor(criterion='squared_error', max_depth=100, random_state=0,
                                                             splitter='random', min_impurity_decrease=0.9)
    },
    'student_performance_mat': {
        'filenames': ['student-mat.csv'],
        'header': None,
        'delimiter': ';',
        'y': ['G1', 'G2', 'G3'],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'stratify': False,
        'split_random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.tree.DecisionTreeRegressor(criterion='poisson', max_depth=100, random_state=0,
                                                             splitter='best', min_impurity_decrease=0.0)
    },
    'student_performance_por': {
        'filenames': ['student-por.csv'],
        'header': None,
        'delimiter': ';',
        'y': ['G1', 'G2', 'G3'],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'stratify': False,
        'split_random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.tree.DecisionTreeRegressor(criterion='poisson', max_depth=100, random_state=0,
                                                             splitter='best', min_impurity_decrease=0.0)
    },
    'compressive_strength': {
        'filenames': ['Concrete_Data.xls'],
        'header': None,
        'delimiter': ',',
        'y': ['Concrete compressive strength(MPa, megapascals) '],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'stratify': False,
        'split_random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.ensemble.RandomForestRegressor(max_depth=10, n_estimators=100, random_state=0)
    },
    'sgemm_gpu': {
        'filenames': ['sgemm_product.csv'],
        'header': None,
        'delimiter': ',',
        'y': ['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'stratify': False,
        'split_random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None,
        'plots': ['best_estimator_vs_parameters', 'learning_curve'],
        'best_estimator': sklearn.ensemble.RandomForestRegressor(max_depth=5, n_estimators=10, random_state=0)
    }
}
