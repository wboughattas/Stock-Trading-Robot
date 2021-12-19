from Models import *

sgemm_gpu_params = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': ['True']
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [1],
            'gamma': ['auto'],
            'kernel': ['rbf']
        }
    },
    'DecisionTreeRegressor': {
        'model': DecisionTreeRegressor(random_state=0),
        'params': {
            'max_depth': [1],
            'splitter': ['best'],

        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=0),
        'params': {
            'max_depth': [5],
            'n_estimators': [10]
        }
    },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'weights': ['uniform'],
            'n_neighbors': [2]
        }
    },
    'AdaBoostRegressor': {
        'model': AdaBoostRegressor(random_state=0),
        'params': {
            'n_estimators': [5],
            'learning_rate': [0.1]
        }
    },
    'MLPRegressor': {
        'model': MLPRegressor(random_state=0),
        'params': {
            'hidden_layer_sizes': [10],
            'max_iter': [100],
        }
    }
}
