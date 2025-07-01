import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import numpy as np

def snv(data):          
    # Define a new array and populate it with the corrected data        
    output_data = np.zeros_like(data)      
    for i in range(data.shape[0]):            
        # Apply correction          
        output_data[i,:] = (data[i,:] - np.mean(data[i,:])) / np.std(data[i,:])        
    return output_data

# Lendo o CSV com a primeira coluna como índice
df = pd.read_csv("whey.csv", index_col=0)

print(df.head())

#Partial Least Squares Regression

# Padronização Snv, Golay
X = df.T.values
X_snv = snv(X)
X_savgol = savgol_filter(X_snv, window_length=9, polyorder = 2, deriv=1)
#X_centered = X_savgol - np.mean(X_savgol, axis=0)
X = X_savgol

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

# Aplicar PLS com 6 componentes centrado na média
pls = PLSRegression(n_components=6, scale=False)
pls.fit(X, y)

# Obter predições
y_pred = pls.predict(X)
#print(y_pred)

# Avaliações
r2 = r2_score(y, y_pred)
rmsep = np.sqrt(mean_squared_error(y, y_pred))

# Previsão com validação cruzada Leave-One-Out
loo = LeaveOneOut()
y_loo_pred = cross_val_predict(pls, X, y, cv=loo).ravel()
rmsecv = np.sqrt(mean_squared_error(y, y_loo_pred))

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

#Visualizar real vs predito
plt.figure(figsize=(9, 6))
plt.scatter(y, y_pred, color='blue', edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Predito")
plt.title("Regressão PLS - Real vs Predito")
plt.text(min(y), max(y), f"R² = {r2:.4f}", 
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
plt.grid(True)
plt.tight_layout()
plt.show()

#Gráfico EJCR
from scipy.stats import f
from sklearn.linear_model import LinearRegression
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
ax.plot(ellipse[0], ellipse[1], label='EJCR (95%)')
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





