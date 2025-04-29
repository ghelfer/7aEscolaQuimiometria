import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Dados fornecidos
x_c = [0.00, 0.00, 9.66, 9.66, 19.78, 19.78, 40.06, 40.06, 59.97, 59.97, 81.16, 81.16, 100.0, 100.0]
y_c = [0.0700935525, 0.161636353, 10.5638821, 9.12936026, 19.5548922, 19.3732015, 40.5262296, 39.5583642, 60.9920251, 58.5657849, 81.7435364, 81.4624962, 99.5435625, 100.014935]

x_w = [0.0, 0.0, 10.1, 10.1, 19.3, 19.3, 39.7, 39.7, 60.0, 60.0, 81.5, 81.5, 100.0, 100.0]
y_w = [-0.0841366188, 0.140126654, 10.4761767, 9.86506895, 19.3750561, 18.9086708, 39.6645730, 39.8627969, 59.8604673, 60.2154311, 81.8307484, 81.0614454, 99.4126881, 100.610887]

# Unir os dados
X_all = np.array(x_c + x_w)
y_all = np.array(y_c + y_w)

r2 = r2_score(X_all, y_all)
rmsep = mean_squared_error(X_all, y_all, squared=False)

# Plot
plt.figure(figsize=(9, 6)) 
plt.scatter(X_all, y_all, color='blue', edgecolors='k')  # pontos
plt.plot([X_all.min(), X_all.max()], [X_all.min(), X_all.max()], 'r--')  # linha ideal
plt.xlabel("Valor Real")
plt.ylabel("Valor Predito")
plt.title("Regressão PLS - Real vs Predito")
plt.grid(False)
# Inserção do R² e RMSEP no gráfico
plt.text(min(X_all), max(y_all), f"R² = {r2:.4f}\nRMSEP = {rmsep:.4f}", 
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.tight_layout()
plt.show()
