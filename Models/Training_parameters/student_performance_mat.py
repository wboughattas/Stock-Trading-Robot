from Models import *

student_performance_mat_params = {
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
            'max_depth': [100, 500, 1000],
            'splitter': ['random', 'best'],
            'min_impurity_decrease': [0., 0.1, 0.5, 0.9]
        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=0),
        'params': {
            'max_depth': [50, 100, 200],
            'n_estimators': [100, 200],
            'max_features': ['auto']
        }
    },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree'],
            'n_neighbors': [50, 100, 200]
        }
    },
    'AdaBoostRegressor': {
        'model': AdaBoostRegressor(random_state=0),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 0.9],
            'loss': ['linear', 'square', 'exponential']
        }
    },
    'GaussianProcessRegressor': {
        'model': GaussianProcessRegressor(),
        'params': {
            'alpha': [1e-10, 0.00001, 0.001],
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
