# ==========================================
# TASK 8: SIMPLE & MULTIPLE LINEAR REGRESSION
# Dataset: Housing.csv
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# STEP 1: Load Dataset
# ==========================================

df = pd.read_csv("Housing.csv")

print("First 5 Rows:")
print(df.head())

# ==========================================
# STEP 2: Convert Yes/No Columns
# ==========================================

df.replace({
    'yes': 1,
    'no': 0
}, inplace=True)

# Convert Furnishing Status

df['furnishingstatus'] = df['furnishingstatus'].map({
    'unfurnished': 0,
    'semi-furnished': 1,
    'furnished': 2
})

# ==========================================
# SIMPLE LINEAR REGRESSION
# Price Prediction using Area
# ==========================================

print("\n===== SIMPLE LINEAR REGRESSION =====")

X = df[['area']]
y = df['price']

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Model

simple_model = LinearRegression()

# Train Model

simple_model.fit(X_train, y_train)

# Prediction

y_pred = simple_model.predict(X_test)

# Evaluation

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Intercept:", simple_model.intercept_)
print("Coefficient:", simple_model.coef_[0])
print("MSE:", mse)
print("R2 Score:", r2)

# ==========================================
# Regression Line Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.scatter(X_test, y_test)

plt.plot(X_test, y_pred)

plt.title("Simple Linear Regression")
plt.xlabel("Area")
plt.ylabel("Price")

plt.show()

# ==========================================
# MULTIPLE LINEAR REGRESSION
# ==========================================

print("\n===== MULTIPLE LINEAR REGRESSION =====")

# Select Features

X_multi = df[['area', 'bedrooms', 'bathrooms', 'parking']]

y_multi = df['price']

# Train-Test Split

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi,
    y_multi,
    test_size=0.2,
    random_state=42
)

# Create Model

multi_model = LinearRegression()

# Train Model

multi_model.fit(X_train_m, y_train_m)

# Prediction

y_pred_m = multi_model.predict(X_test_m)

# Evaluation

mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)

print("Intercept:", multi_model.intercept_)

print("\nCoefficients:")

for feature, coef in zip(X_multi.columns, multi_model.coef_):
    print(feature, ":", coef)

print("\nMSE:", mse_m)
print("R2 Score:", r2_m)

# ==========================================
# Actual vs Predicted Values
# ==========================================

results = pd.DataFrame({
    'Actual Price': y_test_m,
    'Predicted Price': y_pred_m
})

print("\nActual vs Predicted:")
print(results.head(10))

# ==========================================
# Residual Error Graph
# ==========================================

errors = y_test_m - y_pred_m

plt.figure(figsize=(8,5))

plt.scatter(y_pred_m, errors)

plt.axhline(y=0)

plt.title("Residual Error Analysis")
plt.xlabel("Predicted Price")
plt.ylabel("Error")

plt.show()

print("\nTask 8 Completed Successfully!")
