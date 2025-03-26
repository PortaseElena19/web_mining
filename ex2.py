import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('fisier.csv')
print(data.head())

data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creăm caracteristici polinomiale pentru variabilele numerice (de exemplu, 'sqft' și 'bedrooms')
numeric_features = ['sqft', 'bedrooms']  

# Aplicăm PolynomialFeatures doar pe caracteristicile numerice
poly = PolynomialFeatures(degree=2)  
X_train_poly = poly.fit_transform(X_train[numeric_features])
X_test_poly = poly.transform(X_test[numeric_features])

X_train_poly_full = np.concatenate([X_train_poly, X_train.drop(numeric_features, axis=1).values], axis=1)
X_test_poly_full = np.concatenate([X_test_poly, X_test.drop(numeric_features, axis=1).values], axis=1)

poly_model = LinearRegression()
poly_model.fit(X_train_poly_full, y_train)

y_pred_poly = poly_model.predict(X_test_poly_full)

r2_poly = r2_score(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)

print(f"Model polinomial (cu caracteristici de ordinul 2):")
print(f"R²: {r2_poly:.4f}")
print(f"MSE: {mse_poly:.4f}")
print(f"MAE: {mae_poly:.4f}")

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

r2_linear = r2_score(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

print(f"\nModel liniar de bază:")
print(f"R²: {r2_linear:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"MAE: {mae_linear:.4f}")




