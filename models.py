from array import array
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV


def train(param_grid: {}, model: GradientBoostingRegressor, x_train: array, y_train: array) -> GradientBoostingRegressor:
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(x_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Get the best model

    return grid_search.best_estimator_


def predict(model: GradientBoostingRegressor, x_test: array, y_test: array) -> array:
    # Make predictions with the best model
    y_pred = model.predict(x_test)

    # Evaluate the best model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)
    return y_pred
