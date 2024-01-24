import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import data_preprocessing
from sklearn.preprocessing import StandardScaler

X, y = data_preprocessing.read(filename='diabetes_prediction_dataset.csv', X_end=-2)

X = data_preprocessing.simple_encode(X, single_column=0)
X = data_preprocessing.hot_encode(X, single_column=4)
# Split the dataset; svr doesnt autoscale and not doing this will cause calculation problems

x_scaler, y_scaler = StandardScaler(), StandardScaler()
X = x_scaler.fit_transform(X)
# y doesnt need scaling

# X = data_preprocessing.backward_eliminatation(X, y)  # reduces precision, so i didnt use it

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the model (Random Forest Regressor in this example)
model = RandomForestRegressor(random_state=0)
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid for hyperparameter tuning
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }


# # Perform grid search with cross-validation
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# # Get the best model from grid search
# best_model = grid_search.best_estimator_

# Make predictions with the best model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error after tuning: {mse:.2f}")
print(f"R-squared after tuning: {r2:.2f}")

# Visualize the predicted vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Blood Glucose Level")
plt.ylabel("Predicted Blood Glucose Level")
plt.title("Actual vs. Predicted Blood Glucose Level")
plt.show()

diffs = [0] * len(y_test)
i = 0
for yt in y_test:
    diffs[i] = yt - y_pred[i]
    i += 1
