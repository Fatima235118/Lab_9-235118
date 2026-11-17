
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset and remove duplicate rows
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()

# Select input and target
target = "Concrete compressive strength"

X = concrete.drop(columns=target)
y = concrete[target]

print("Total rows after removing duplicates:", len(concrete))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Create five shuffled folds
folds = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = "neg_root_mean_squared_error"

# Linear Regression baseline
linear_model = LinearRegression()

linear_scores = -cross_val_score(
    linear_model,
    X_train,
    y_train,
    cv=folds,
    scoring=scoring
)

print("\nLinear Regression Fold RMSE:")
print(linear_scores.round(2))

linear_mean = linear_scores.mean()

print("Linear Mean CV RMSE:", round(linear_mean, 2))

# Create Polynomial and Ridge Pipeline
ridge_model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge())
])

# Define parameter grid
param_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

# Perform Grid Search
search = GridSearchCV(
    ridge_model,
    param_grid,
    cv=folds,
    scoring=scoring,
    refit=True
)

search.fit(X_train, y_train)

# Display best settings
print("\nBest Parameters:")
print(search.best_params_)

print("Best Mean CV RMSE:", round(-search.best_score_, 2))

# Display three highest-ranked settings
results = pd.DataFrame(search.cv_results_)

top_three = results.sort_values(
    "rank_test_score"
).head(3)

print("\nTop Three Grid Search Settings:")

for _, row in top_three.iterrows():
    print(
        "Degree:", row["param_poly__degree"],
        "Alpha:", row["param_ridge__alpha"],
        "Mean CV RMSE:", round(-row["mean_test_score"], 2),
        "Rank:", row["rank_test_score"]
    )

# Evaluate selected model on test set
test_predictions = search.predict(X_test)

test_rmse = mean_squared_error(
    y_test, test_predictions
) ** 0.5

test_mae = mean_absolute_error(
    y_test, test_predictions
)

print("\nTest RMSE:", round(test_rmse, 2))
print("Test MAE:", round(test_mae, 2))

# Compare baseline and tuned model
print("\nBaseline Mean CV RMSE:", round(linear_mean, 2))
print("Tuned Mean CV RMSE:", round(-search.best_score_, 2))

# Plot actual and predicted strength
plt.scatter(
    y_test,
    test_predictions,
    label="Predicted Values"
)

minimum = min(y_test.min(), test_predictions.min())
maximum = max(y_test.max(), test_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    label="Perfect Prediction (y=x)"
)

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")
plt.legend()
plt.grid()

plt.savefig("task3_concrete.png")
plt.show()
