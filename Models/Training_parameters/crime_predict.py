from Models import *

crime_predict_params = {
    'linear_regression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': ['True', 'False']
        }
    },
    'svr': {
        'model': SVR(),
        'params': {
            'C': [1, 10, 20],
            'gamma': ['auto', 'scale'],
            'kernel': ['rbf', 'linear']
        }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor(random_state=0),
        'params': {
            'max_depth': [1, 5, 10],
            'splitter': ['random', 'best'],

        }
    },
    'random_forest': {
        'model': RandomForestRegressor(random_state=0),
        'params': {
            'max_depth': [1, 5, 10],
            'n_estimators': [1, 10, 100]
        }
    },
    'K_means': {
        'model': KNeighborsRegressor(),
        'params': {
            'weights': ['uniform', 'distance'],
            'n_neighbors': [1, 5, 10]
        }
    },
    'adaBoost': {
        'model': AdaBoostRegressor(random_state=0),
        'params': {
            'n_estimators': [1, 5, 10],
            'learning_rate': [0.1, 0.5, 1]
        }
    },
    'GaussianNB': {
        'model': GaussianProcessRegressor(),
        'params': {
            'alpha': [0.5, 1, 1.5]
        }
    },
    'neural_network': {
        'model': MLPRegressor(random_state=0),
        'params': {
            'hidden_layer_sizes': [50, 100, 150],
            'max_iter': [100, 200, 300],

        }
    }
}
