import csv
import os
from Models.Training_parameters import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xlrd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *


def files_to_metadata(param_grid):
    """Reads the csv files in current directory and extracts dataset, X, X_train, X_test, y, y_train, y_test,
    dataset_header, X_header, y_header, and decoder for each Data file and transform them according to read parameters

    :param param_grid: Dictionary with dataset names (str) as keys corresponding to the name of directory that contains
    the filename(s). and dictionaries of parameter settings to read and edit the corresponding dataset as values. The
    param setting are: filename(s) (list) including the file extension, header (str): None if header is within dataset,
    delimiter (str), y (list): name of y column(s), skip_columns (int), train_size (float), random_state (int),
    and 'included_test_train' (bool): True if the provided files are test and train datasets.

    :return: Dictionary with filenames (str) as keys and dictionaries of metadata. The metadata include dataset, X,
    X_train, X_test, y, y_train, y_test, header, X_header, y_header, and decoder.

    :Reasoning for not opting for traditional numpy.loadtxt() or pandas.read_csv():
    1. flexible parameters using a param_grid (similar idea to GridSearchCV's param_grid)
    2. automate reading files
    3. very easy access to: dataset, X, X_train, X_test, y, y_train, y_test, dataset_header, X_header, y_header,
    and decoder by indexing the output of files_to_metadata (dict)
    4. avoid learning new pandas syntax
    """

    dir_abspath = os.path.join(os.path.abspath(os.curdir), 'Data')
    datasets, headers, decoders = [], [], []
    metadata_grid = {}
    dataset_count = 0

    for key, value in param_grid.items():
        path_names = [os.path.join(dir_abspath, key, i) for i in value['filenames']]
        for file_idx, file in enumerate(path_names):
            filename = os.path.basename(str(file))
            print('Reading {}'.format(filename))
            with open(file, newline='', encoding='utf-8-sig') as datafile:
                if os.path.splitext(file)[1] == '.xls':
                    sheet = xlrd.open_workbook(file).sheet_by_index(0)
                    reader = iter([sheet.row_values(row) for row in range(sheet.nrows)])
                else:
                    reader = csv.reader(datafile, skipinitialspace=True, delimiter=value['delimiter'])
                [next(reader, None) for _ in range(value['skip_rows'][file_idx])]
                if value['header'] is None:
                    header = next(reader)
                else:
                    header_file_name = os.path.join(dir_abspath, key, value['header'])
                    with open(header_file_name, newline='', encoding='utf-8-sig') as header_file:
                        header = [j for sub in list(csv.reader(header_file)) for j in sub]
                data = []
                decoder = {}
                for i, row in enumerate(reader):
                    lst = []
                    col_idx = 0
                    for s in row:
                        try:
                            lst.append(float(s))
                        except (Exception,):  # value is not a number
                            if s in ['', '?']:
                                lst.append(float('Nan'))
                                col_idx += 1
                                continue
                            col_idx_str = header[col_idx]
                            if col_idx_str not in decoder.keys():
                                decoder[col_idx_str] = {}
                            existing_original_str = decoder[col_idx_str].keys()
                            existing_decoded_str = decoder[col_idx_str].values()
                            if str(s) not in existing_original_str:
                                if len(existing_original_str) > 0:
                                    decoded_val = max(existing_decoded_str) + 1
                                else:
                                    decoded_val = 0
                                decoder[col_idx_str][str(s)] = decoded_val
                                lst.append(decoded_val)
                            else:
                                lst.append(decoder[col_idx_str][str(s)])
                        col_idx += 1
                    data.append(lst) if len(lst) != 0 else None

                datasets.append(data)
                headers.append(header)
                decoders.append(decoder)

            head = headers[dataset_count]
            decode = decoders[dataset_count]
            skip_columns_indices = [head.index(i) for i in value['skip_columns']]
            dataset = np.array(datasets[dataset_count])
            dataset = np.delete(dataset, skip_columns_indices, axis=1)

            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            dataset = imp_mean.fit_transform(dataset)

            head = [head[i] for i in range(len(head)) if i not in skip_columns_indices]

            if len(value['y']) != 1:
                y_indices = [idx for i in value['y'] if (idx := head.index(i))]
                target_mean = dataset[:, y_indices].mean(axis=1).reshape((-1, 1))
                dataset = np.delete(dataset, y_indices, axis=1)
                dataset = np.append(dataset, target_mean, axis=1)

                head = [head[i] for i in range(len(head)) if i not in y_indices]
                head.append('mean_target')
                value['y'] = ['mean_target']
                head[y_indices[0]] = 'mean_target'

            y_indices = [head.index(i) for i in value['y']]
            X_indices = [idx for idx in range(len(head)) if idx not in y_indices]

            y_header = [head[i] for i in y_indices]
            X_header = [head[i] for i in X_indices]

            y = dataset[:, y_indices]
            X = dataset[:, X_indices]

            y = y.reshape(-1)
            if all([i.is_integer() for i in y]):
                y = y.astype(int)

            unique_y = np.unique(y)
            if len(unique_y) == 2 and not all(unique_y == np.array([0, 1])):
                if value['positive_label'] is not None:
                    if all(isinstance(value['positive_label'][i], (int, float)) for i in range(len(path_names))):
                        pos_label = value['positive_label'][file_idx]
                        neg_label = np.delete(unique_y, np.where(unique_y == pos_label))[0]
                        map_func = lambda x: 1 if x == pos_label else 0
                        y = [map_func(i) for i in y]

                        decode['y_col'] = {}
                        decode['y_col'][neg_label] = 0
                        decode['y_col'][pos_label] = 1

            dataset[:, y_indices[0]] = y
            scaler = StandardScaler()

            if value['included_test_train'] is False:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=value['train_size'],
                                                                    random_state=value['random_state'])
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            else:
                X_train, X_test, y_train, y_test = None, None, None, None

            filename = os.path.basename(str(filename))
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
                'learning_curve_scoring': value['learning_curve_scoring']
            }
            dataset_count += 1

    return metadata_grid


def metadata_to_xy_trn_tst(param_grid, *filenames):
    """
    converts metadata to a dictionary that contains data that is ready for training and visualizing
    The dictionary includes X_train, X_test, y_train, y_test, etc...
    :param param_grid: CL_param_grid or REGR_param_grid
    :param filenames: name of file(s), at most two, that contains the dataset. If >1, Files must be a train and test
    datasets respectively
    :return X_train, X_test, y_train, y_test, and model_params of the dataset
    """

    Data_abspath = os.path.join(os.path.abspath(os.curdir), 'Data/')
    dataset_dirname = None

    for root, dirs, files in os.walk(Data_abspath):
        for file in files:
            if file == filenames[0]:
                dataset_dirname = os.path.basename(root)

    model_params = globals()['{}_params'.format(dataset_dirname)]
    import_param = {dataset_dirname: param_grid[dataset_dirname]}
    METADATA_S = files_to_metadata(import_param)
    metadata = METADATA_S[filenames[0]]

    header, X_header, y_header, scaler, estimator_scoring, learning_curve_scoring = \
        metadata['header'], metadata['X_header'], metadata['y_header'], metadata['scaler'], \
        metadata['estimator_scoring'], metadata['learning_curve_scoring']

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
        train_metadata = METADATA_S[filenames[0]]
        test_metadata = METADATA_S[filenames[1]]

        X_train, X_test = train_metadata['X'], test_metadata['X']
        y_train, y_test = train_metadata['y'], test_metadata['y']

        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=None)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        dataset = np.append(X, y.reshape((-1, 1)), axis=1)
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
        'dataset_dirname': dataset_dirname
    }

    return output


read_files_param_grid = {
    # 'diabetic_retinopathy': {
    #     'filenames': ['messidor_features.arff'],
    #     'header': 'columns',
    #     'delimiter': ',',
    #     'y': ['class label'],
    #     'skip_columns': [],
    #     'skip_rows': [24],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [accuracy_score, precision_score, log_loss],
    #     'learning_curve_scoring': [[log_loss]],
    #     'positive_label': [1]
    # },
    # 'credit_default': {
    #     'filenames': ['default of credit card clients.xls'],
    #     'header': None,
    #     'delimiter': None,
    #     'y': ['default payment next month'],
    #     'skip_columns': ['ID'],
    #     'skip_rows': [1],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [accuracy_score, precision_score, log_loss],
    #     'learning_curve_scoring': [[log_loss]],
    #     'positive_label': [1]
    # },
    # 'breast_cancer': {
    #     'filenames': ['breast-cancer-wisconsin.data'],
    #     'header': 'columns',
    #     'delimiter': ',',
    #     'y': ['Class'],
    #     'skip_columns': ['Sample code number'],
    #     'skip_rows': [0],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [accuracy_score, precision_score, log_loss],
    #     'learning_curve_scoring': [[log_loss]],
    #     'positive_label': [4]
    # },
    # # requires attention: last 5 columns in "german.data-numeric" have no headers. Target column is not specified
    # # "german.data" is used instead. Last unnamed column is used as target
    # 'german_credit': {
    #     'filenames': ['german.data'],
    #     'header': 'columns',
    #     'delimiter': ' ',
    #     'y': ['noname1'],
    #     'skip_columns': [],
    #     'skip_rows': [0],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss],
    #     'learning_curve_scoring': [[log_loss]],
    #     'positive_label': [2]
    # },
    # 'adult': {
    #     'filenames': ['adult.data', 'adult.test'],  # start with the train file because train_standard_scaler is used
    #     'header': 'columns',
    #     'delimiter': ',',
    #     'y': ['yearly salary'],
    #     'skip_columns': [],
    #     'skip_rows': [0, 1],
    #     'train_size': None,
    #     'random_state': 0,
    #     'included_test_train': True,
    #     'estimator_scoring': [accuracy_score, precision_score, log_loss],
    #     'learning_curve_scoring': [[log_loss]],
    #     'positive_label': ['>50K', '>50K.']
    # },
    # # multiclass
    # 'yeast': {
    #     'filenames': ['yeast.data'],
    #     'header': 'columns',
    #     'delimiter': ' ',
    #     'y': ['location'],
    #     'skip_columns': ['Sequence Name'],
    #     'skip_rows': [0],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [accuracy_score, f1_score],
    #     'learning_curve_scoring': [[log_loss]],
    #     'positive_label': None
    # },
    # 'thoracic_surgery': {
    #     'filenames': ['ThoraricSurgery.arff'],
    #     'header': 'columns',
    #     'delimiter': ',',
    #     'y': ['Risk1Yr'],
    #     'skip_columns': [],
    #     'skip_rows': [21],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [accuracy_score, precision_score, log_loss],
    #     'learning_curve_scoring': [[log_loss]],
    #     'positive_label': ['T']
    # },
    # 'seismic_bumps': {
    #     'filenames': ['seismic-bumps.arff'],
    #     'header': 'columns',
    #     'delimiter': ',',
    #     'y': ['class'],
    #     'skip_columns': [],
    #     'skip_rows': [154],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [accuracy_score, precision_score, log_loss],
    #     'learning_curve_scoring': [[log_loss]],
    #     'positive_label': [1]
    # },
    # 'wine_quality_white': {
    #     'filenames': ['winequality-white.csv'],
    #     'header': None,
    #     'delimiter': ';',
    #     'y': ['quality'],
    #     'skip_columns': [],
    #     'skip_rows': [0],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
    #     'learning_curve_scoring': [[mean_squared_error]],
    #     'positive_label': None
    # },
    # 'wine_quality_red': {
    #     'filenames': ['winequality-red.csv'],
    #     'header': None,
    #     'delimiter': ';',
    #     'y': ['quality'],
    #     'skip_columns': [],
    #     'skip_rows': [0],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
    #     'learning_curve_scoring': [[mean_squared_error]],
    #     'positive_label': None
    # },
    # 'crime_predict': {
    #     'filenames': ['communities.data'],
    #     'header': 'communities.names',
    #     'delimiter': ',',
    #     'y': ['ViolentCrimesPerPop numeric'],
    #     'skip_columns': ['state numeric', 'county numeric', 'community numeric', 'communityname string',
    #                      'fold numeric'],
    #     'skip_rows': [0],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
    #     'learning_curve_scoring': [[mean_squared_error]],
    #     'positive_label': None
    # },
    # 'aquatic_toxicity': {
    #     'filenames': ['qsar_aquatic_toxicity.csv'],
    #     'header': 'columns.csv',
    #     'delimiter': ';',
    #     'y': ['quantitative response'],
    #     'skip_columns': [],
    #     'skip_rows': [0],
    #     'train_size': 0.75,
    #     'random_state': 0,
    #     'included_test_train': False,
    #     'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
    #     'learning_curve_scoring': [[mean_squared_error]],
    #     'positive_label': None
    # },
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
        'random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None
    },
    'bike_sharing': {
        'filenames': ['hour.csv'],
        'header': None,
        'delimiter': ',',
        'y': ['cnt'],
        'skip_columns': ['instant'],
        'skip_rows': [0],
        'train_size': 0.75,
        'random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None
    },
    'student_performance_mat': {
        'filenames': ['student-mat.csv'],
        'header': None,
        'delimiter': ';',
        'y': ['G1', 'G2', 'G3'],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None
    },
    'student_performance_por': {
        'filenames': ['student-por.csv'],
        'header': None,
        'delimiter': ';',
        'y': ['G1', 'G2', 'G3'],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None
    },
    'compressive_strength': {
        'filenames': ['Concrete_Data.xls'],
        'header': None,
        'delimiter': ',',
        'y': ['Concrete compressive strength(MPa, megapascals) '],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None
    },
    'sgemm_gpu': {
        'filenames': ['sgemm_product.csv'],
        'header': None,
        'delimiter': ',',
        'y': ['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'],
        'skip_columns': [],
        'skip_rows': [0],
        'train_size': 0.75,
        'random_state': 0,
        'included_test_train': False,
        'estimator_scoring': [mean_absolute_error, explained_variance_score, r2_score, mean_squared_error],
        'learning_curve_scoring': [[mean_squared_error]],
        'positive_label': None
    }
}
