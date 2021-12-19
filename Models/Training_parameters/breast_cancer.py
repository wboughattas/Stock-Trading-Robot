from Models import *

breast_cancer_params = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=0, max_iter=10000),
        'params': {
            'fit_intercept': [False, True],
            'penalty': ['none', 'l2'],
            'C': [1.0, 2.5]
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
            'max_depth': [1, 5, 20],
            'splitter': ['random', 'best'],
            'criterion': ['gini', 'entropy']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=0),
        'params': {
            'max_depth': [4, 9, 12],
            'n_estimators': [3, 100, 500],
            'criterion': ['gini', 'entropy']
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [10, 200, 500]
        }
    },
    'AdaBoostClassifier': {
        'model': AdaBoostClassifier(random_state=0),
        'params': {
            'algorithm': ['SAMME.R', 'SAMME'],
            'n_estimators': [3, 8],
            'learning_rate': [0.1, 0.5, 1]
        }
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-09, 1e-05, 0.1]
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
