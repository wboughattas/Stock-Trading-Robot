from Models import *

compressive_strength_params = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': ['True', 'False']
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [1, 10, 20],
            'gamma': ['auto', 'scale'],
            'kernel': ['rbf', 'linear']
        }
    },
    'DecisionTreeRegressor': {
        'model': DecisionTreeRegressor(random_state=0),
        'params': {
            'max_depth': [1, 5, 10],
            'splitter': ['random', 'best'],

        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=0),
        'params': {
            'max_depth': [1, 5, 10],
            'n_estimators': [1, 10, 100]
        }
    },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'weights': ['uniform', 'distance'],
            'n_neighbors': [1, 5, 10]
        }
    },
    'AdaBoostRegressor': {
        'model': AdaBoostRegressor(random_state=0),
        'params': {
            'n_estimators': [1, 5, 10],
            'learning_rate': [0.1, 0.5, 1]
        }
    },
    'GaussianProcessRegressor': {
        'model': GaussianProcessRegressor(),
        'params': {
            'alpha': [0.5, 1, 1.5]
        }
    },
    'MLPRegressor': {
        'model': MLPRegressor(random_state=0),
        'params': {
            'hidden_layer_sizes': [50, 100],
            'max_iter': [100, 200],
        }
    }
}
