import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Mapeamento entre nomes amigáveis e caminhos dos arquivos
csv_mapping = {
    "crea_pls": "y_crea_pls.csv",
    "crea_svr": "y_crea_svr.csv",
    "crea_tf": "y_crea_tf.csv",
    "whey_pls": "y_whey_pls.csv",
    "whey_svr": "y_whey_svr.csv",
    "whey_tf": "y_whey_tf.csv",
}

# Cores distintas
colors = {
    "crea_pls": "blue",
    "crea_svr": "green",
    "crea_tf": "red",
    "whey_pls": "cyan",
    "whey_svr": "magenta",
    "whey_tf": "orange",
}

# Criar figura
plt.figure(figsize=(8, 8))

# Loop pelos arquivos
for name, file_path in csv_mapping.items():
    df = pd.read_csv(file_path)
    if {"y_true", "y_pred"}.issubset(df.columns):
        x = df["y_true"].values.reshape(-1, 1)
        y = df["y_pred"].values

        # Pontos
        plt.scatter(x, y, color=colors[name], alpha=1.0, marker='o', label=f"{name}")

        # Regressão linear
        model = LinearRegression()
        model.fit(x, y)
        r2 = r2_score(y, model.predict(x))

        # Linha de tendência
        x_range = np.linspace(-20, 120, 200).reshape(-1, 1)
        y_trend = model.predict(x_range)
        plt.plot(x_range, y_trend, color=colors[name], linewidth=1.5, label=f"{name} R²={r2:.2f}")

# Delinear eixo X=0 e Y=0
plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
plt.axvline(x=0, color='gray', linestyle='-', linewidth=1)

# Configurações finais do gráfico
plt.xlim(-5, 105)
plt.ylim(-5, 105)
plt.xlabel("y_true")
plt.ylabel("y_pred")
plt.title("Y_pred vs Y_true - Modelagem da Validação Cruzada")
plt.legend()
#plt.grid(True)
plt.tight_layout()
plt.show()
