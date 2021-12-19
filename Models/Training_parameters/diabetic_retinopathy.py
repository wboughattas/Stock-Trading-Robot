from Models import *

diabetic_retinopathy_params = {
    'logistic_regression': {
        'model': LogisticRegression(random_state=0, max_iter=10000),
        'params': {
            'fit_intercept': [False, True],
            'penalty': ['none', 'l2'],
            'C': [1, 5, 10]
        }
    },
    'svm': {
        'model': SVC(random_state=0),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear', 'poly'],
            'max_iter': [10, 100, 1000],
            'gamma': ['scale', 'auto']
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(random_state=0),
        'params': {
            'max_depth': [1, 5, 20],
            'splitter': ['random', 'best'],
            'criterion': ['gini', 'entropy']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(random_state=0),
        'params': {
            'max_depth': [10, 15, 20],
            'n_estimators': [10, 100, 500],
            'criterion': ['gini', 'entropy']
        }
    },
    'K_means': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [10, 200, 500],
            'weights': ['uniform', 'distance']
        }
    },
    'adaBoost': {
        'model': AdaBoostClassifier(random_state=0),
        'params': {
            'algorithm': ['SAMME.R', 'SAMME'],
            'n_estimators': [1, 5, 10],
            'learning_rate': [0.01, 0.1, 1]
        }
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [0.1, 1, 10]
        }
    },
    'neural_network': {
        'model': MLPClassifier(random_state=0, solver='sgd', momentum=0.9, verbose=True, batch_size=100),
        'params': {
            'max_iter': [100, 1000],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
    }
}
