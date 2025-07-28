import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, ELU
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Lendo o CSV com a primeira coluna como índice
df = pd.read_csv("whey.csv", index_col=0)

print(df.head())

def snv(data):          
    # Define a new array and populate it with the corrected data        
    output_data = np.zeros_like(data)      
    for i in range(data.shape[0]):            
        # Apply correction          
        output_data[i,:] = (data[i,:] - np.mean(data[i,:])) / np.std(data[i,:])        
    return output_data
    
# Padronização Snv, Golay e Mean-center
X = df.T.values
X_snv = snv(X)
X_savgol = savgol_filter(X_snv, window_length=9, polyorder = 2, deriv=1)
X = X_savgol - np.mean(X_savgol, axis=0)

targets = {
    'Whey_a': 0.0,'Whey_b': 0.0,
    'W10a': 10.1, 'W10b': 10.1,
    'W20a': 19.3, 'W20b': 19.3,
    'W40a': 39.7, 'W40b': 39.7,
    'W60a': 60.0, 'W60b': 60.0,
    'W80a': 81.5, 'W80b': 81.5,
    'Amb': 100.0, 'Ama': 100.0,
}

y = np.array([targets[col] for col in df.transpose().index])

# Define a função que cria o modelo
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='RMSprop', loss='mse')
    return model

# Compilação    
model = build_model(X.shape[1])

# Treinamento
model.fit(X, y, epochs=200, batch_size=4, verbose=0)

# Avaliação
y_pred = model.predict(X).flatten()
rmsep = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
mae = np.mean(np.abs(y - y_pred))
acc_aprox = 100 - (mae / np.mean(y)) * 100

print(f"R²: {r2:.4f} | RMSEP: {rmsep:.4f} | MAE: {mae:.4f} | Acurácia aprox.: {acc_aprox:.2f}%")

#Visualizar real vs predito
plt.figure(figsize=(9, 6))
plt.scatter(y, y_pred, color='blue', edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Predito")
plt.title("Tensor Flow - Creatina - Real vs Predito")
# Inserção do R² e RMSEP no gráfico
plt.text(min(y), max(y), f"R² = {r2:.4f}\nRMSEP = {rmsep:.4f}", 
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
plt.grid(True)
plt.tight_layout()
plt.show()

# Leave-One-Out manual
loo = LeaveOneOut()
y_true = []
y_pred = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = build_model(X.shape[1])
    model.fit(X_train, y_train, epochs=200, batch_size=4, verbose=0)

    y_pred.append(model.predict(X_test, verbose=0)[0][0])
    y_true.append(y_test[0])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Salvar y_pred em CSV
y_pred_df = pd.DataFrame({'y_true': y, 'y_pred': y_pred.ravel()})
y_pred_df.to_csv("y_whey_tf.csv", index=False)

# Avaliação
rmsecv = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSECV: {rmsecv:.6f}")


# Métrica REP (Relative Error of Prediction)
rep = (rmsep / np.mean(y)) * 100  # em %
# Métrica RPD (Ratio of Performance to Deviation)
std_y = np.std(y, ddof=1)
rpd = std_y / rmsep

# Exibir métricas
print(f"R²:     {r2:.6f}")
print(f"RMSEP:  {rmsep:.6f}")
print(f"RMSECV: {rmsecv:.6f}")
print(f"REP (%):  {rep:.2f}")
print(f"RPD:      {rpd:.2f}")

#Gráfico EJCR
from scipy.stats import f
from sklearn.linear_model import LinearRegression

# Carregar os dados
df = pd.read_csv("y_whey_tf.csv")
y = df['y_true'].values
y_pred = df['y_pred'].values

# Regressão linear: y_true em função de y_pred
X = y_pred.reshape(-1, 1)

model = LinearRegression().fit(X, y)
intercept = model.intercept_
slope = model.coef_[0]

# Resíduos e matriz de covariância dos coeficientes
y_fit = model.predict(X)
residuals = y - y_fit
n = len(y)
X_design = np.column_stack((np.ones_like(X), X))  # matriz de projeto [1, x]
cov_matrix = np.linalg.inv(X_design.T @ X_design) * np.sum(residuals**2) / (n - 2)

# Parâmetros do EJCR
b = np.array([intercept, slope])
b0 = np.array([0, 1])  # modelo ideal
alpha = 0.05
F_crit = f.ppf(1 - alpha, dfn=2, dfd=n - 2)
radius = np.sqrt(F_crit)

# Elipse EJCR
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])
L = np.linalg.cholesky(cov_matrix * radius**2)
ellipse = b0[:, None] + L @ circle

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(ellipse[0], ellipse[1], label='EJCR (95%)', markersize=8)
ax.plot(intercept, slope, 'ro', label='Modelo estimado', markersize=8)
ax.plot(0, 1, 'go', label='Modelo ideal (0,1)')
ax.set_xlabel('Intercepto')
ax.set_ylabel('Inclinação')
ax.set_title('Região de Confiança Conjunta Estendida (EJCR)')
ax.axhline(1, color='gray', linestyle='--')
ax.axvline(0, color='gray', linestyle='--')
ax.grid(True)
ax.legend()
ax.set_aspect('auto')
plt.tight_layout()
plt.show()


