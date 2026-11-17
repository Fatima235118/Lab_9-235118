
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score

# Load dataset
yachts = pd.read_csv(
    "yacht_hydrodynamics.data",
    sep=r"\s+",
    header=None
)

# Select input and target
X = yachts[[5]]
y = yachts[6]

# Linear Regression model
linear_model = LinearRegression()

# Quadratic Polynomial Regression model
quadratic_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

# Create 5 shuffled folds
folds = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Set scoring method
scoring = "neg_root_mean_squared_error"

# Calculate Linear Regression RMSE
linear_rmse = -cross_val_score(
    linear_model,
    X,
    y,
    cv=folds,
    scoring=scoring
)

# Calculate Quadratic Regression RMSE
quadratic_rmse = -cross_val_score(
    quadratic_model,
    X,
    y,
    cv=folds,
    scoring=scoring
)

# Display Linear Regression results
print("Linear Regression Fold RMSE:")
print(linear_rmse.round(2))

print("Linear Mean RMSE:", round(linear_rmse.mean(), 2))

# Display Quadratic Regression results
print("\nQuadratic Regression Fold RMSE:")
print(quadratic_rmse.round(2))

print("Quadratic Mean RMSE:", round(quadratic_rmse.mean(), 2))

# Compare both models
if quadratic_rmse.mean() < linear_rmse.mean():
    print("\nBetter Model: Quadratic Regression")
else:
    print("\nBetter Model: Linear Regression")
