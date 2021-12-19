from Models import *

diabetic_retinopathy_params = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=0),
        'params': {
            'fit_intercept': [False, True],
            'penalty': ['none', 'l2'],
            'C': [1, 5, 10, 20, 50],
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
        }
    },
    'SVC': {
        'model': SVC(random_state=0),
        'params': {
            'C': [0.5, 1],
            'kernel': ['rbf', 'linear', 'poly'],
            'coef0': [30],
            'gamma': ['scale']
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=0),
        'params': {
            'max_depth': [1, 5, 20],
            'splitter': ['random', 'best'],
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=0),
        'params': {
            'max_depth': [10, 15, 20],
            'n_estimators': [10, 100, 500],
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [5, 10, 200, 500],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2, 5, 10]
        }
    },
    'AdaBoostClassifier': {
        'model': AdaBoostClassifier(random_state=0),
        'params': {
            'algorithm': ['SAMME.R', 'SAMME'],
            'n_estimators': [1, 5, 10, 50, 100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1]
        }
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 0.1, 1, 10]
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
