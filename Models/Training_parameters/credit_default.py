from Models import *

credit_default_params = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=0),
        'params': {
            'fit_intercept': [False, True],
            'penalty': ['none', 'l2'],
            'C': [10, 50],
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
        }
    },
    'SVC': {
        'model': SVC(random_state=0),
        'params': {
            'C': [0.5, 1, 10, 20],
            'kernel': ['rbf', 'linear', 'poly'],
            'max_iter': [1, 100, 10000]
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=0),
        'params': {
            'max_depth': [10, 100, 1000],
            'splitter': ['random', 'best'],
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=0),
        'params': {
            'max_depth': [50, 100, 200],
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto']
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [50, 100, 200],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        }
    },
    'AdaBoostClassifier': {
        'model': AdaBoostClassifier(random_state=0),
        'params': {
            'algorithm': ['SAMME.R', 'SAMME'],
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 0.9]
        }
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 0.00001, 0.001]
        }
    },
    'MLPClassifier': {
        'model': MLPClassifier(random_state=0, solver='sgd', momentum=0.9, verbose=True, batch_size=100),
        'params': {
            'max_iter': [100, 1000],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.01]
        }
    }
}
