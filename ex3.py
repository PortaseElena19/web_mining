import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

data = pd.read_csv('fisier.csv')
print(data.head())

data = pd.get_dummies(data, drop_first=True)

X = data.drop('price', axis=1) 
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setați parametrii pentru a preveni supraadaptarea (overfitting)
tree_model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)  # Ajustează max_depth și min_samples_leaf
tree_model.fit(X_train, y_train)

# Predicții pe setul de test
y_pred_tree = tree_model.predict(X_test)

r2_tree = r2_score(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)

print(f"Model Decision Tree:")
print(f"R²: {r2_tree:.4f}")
print(f"MSE: {mse_tree:.4f}")
print(f"MAE: {mae_tree:.4f}")

plt.figure(figsize=(20, 10))
plot_tree(tree_model, filled=True, feature_names=X.columns, fontsize=12)
plt.title("Arborele de decizie pentru regresie")
plt.show()
