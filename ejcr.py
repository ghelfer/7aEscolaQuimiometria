import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f
from sklearn.linear_model import LinearRegression
import os

# Lista de caminhos dos arquivos CSV com y_true e y_pred diferentes
files = [
    "y_crea_pls.csv",
    "y_whey_pls.csv",
    "y_crea_svr.csv",
    "y_whey_svr.csv",
    "y_crea_tf.csv",
    "y_whey_tf.csv",
]

# Cores e marcadores para diferenciar modelos
colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
markers = ['o', 's', '^', 'D', 'x', '*']

# Plot do gráfico
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(0, 1, 'go', label='Modelo ideal (0,1)', markersize=10)

for i, file in enumerate(files):
    df = pd.read_csv(file)
    y = df['y_true'].values
    y_pred = df['y_pred'].values
    X = y_pred.reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    intercept = model.intercept_
    slope = model.coef_[0]

    y_fit = model.predict(X)
    residuals = y - y_fit
    n = len(y)
    X_design = np.column_stack((np.ones_like(X), X))
    cov_matrix = np.linalg.inv(X_design.T @ X_design) * np.sum(residuals**2) / (n - 2)

    # Elipse EJCR
    b0 = np.array([0, 1])
    alpha = 0.05
    F_crit = f.ppf(1 - alpha, dfn=2, dfd=n - 2)
    radius = np.sqrt(F_crit)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    L = np.linalg.cholesky(cov_matrix * radius**2)
    ellipse = b0[:, None] + L @ circle

    # Nome do modelo (opcional: extrair nome do arquivo)
    model_name = os.path.basename(file).replace(".csv", "").replace("y_", "")

    ax.plot(ellipse[0], ellipse[1], label=f'EJCR {model_name}', color=colors[i % len(colors)], markersize=12)
    ax.plot(intercept, slope, marker=markers[i % len(markers)], color=colors[i % len(colors)], label=f'Estimado {model_name}')

# Personalização do gráfico
ax.set_xlabel('Intercepto')
ax.set_ylabel('Inclinação')
ax.set_title('EJCR 95% - Comparação de Múltiplos Modelos')
ax.axhline(1, color='gray', linestyle='--')
ax.axvline(0, color='gray', linestyle='--')
ax.grid(False)
ax.legend()
ax.set_aspect('auto')
plt.tight_layout()
plt.show()
