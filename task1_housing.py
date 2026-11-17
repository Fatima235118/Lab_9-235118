
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
homes = pd.read_csv("real_estate.csv")

# Convert distance into 100-meter units
homes["distance_100m"] = homes[
    "X3 distance to the nearest MRT station"
] / 100

# Select input and output
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Display slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict price at 500 meters
prediction = model.predict(
    pd.DataFrame({"distance_100m": [5]})
)

print("Predicted Price at 500 meters:", prediction[0])

# Plot actual data
plt.scatter(X["distance_100m"], y, label="Actual Data")

# Plot regression line
line_x = pd.DataFrame({
    "distance_100m": sorted(X["distance_100m"])
})

line_y = model.predict(line_x)

plt.plot(
    line_x["distance_100m"],
    line_y,
    color="red",
    label="Regression Line"
)

plt.xlabel("Distance to MRT Station (100 meters)")
plt.ylabel("House Price per Unit Area")
plt.title("Housing Price Prediction")
plt.legend()
plt.grid()

# Save graph
plt.savefig("task1_housing.png")

plt.show()
