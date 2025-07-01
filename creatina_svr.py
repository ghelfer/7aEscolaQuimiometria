import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import savgol_filter
import numpy as np

# Lendo o CSV com a primeira coluna como índice
df = pd.read_csv("creatina.csv", index_col=0)

print(df.head())

def snv(data):          
    # Define a new array and populate it with the corrected data        
    output_data = np.zeros_like(data)      
    for i in range(data.shape[0]):            
        # Apply correction          
        output_data[i,:] = (data[i,:] - np.mean(data[i,:])) / np.std(data[i,:])        
    return output_data
    
#Partial Least Squares Regression

# Padronização Snv, Golay e Mean-center
X = df.T.values
X_snv = snv(X)
X_savgol = savgol_filter(X_snv, window_length=9, polyorder = 2, deriv=1)
X = X_savgol - np.mean(X_savgol, axis=0)

targets = {  
    'Cr100': 0.00, 'Cr100b': 0.00,
    'Cr10a': 9.66, 'Cr10b': 9.66,
    'Cr20a': 19.78,'Cr20b': 19.78,
    'Cr40a': 40.06,'Cr40b': 40.06,
    'Cr60a': 59.97,'Cr60b': 59.97,
    'Cr80a': 81.16,'Cr80b': 81.16,
    'Amb': 100.0, 'Ama': 100.0,
}

y = np.array([targets[col] for col in df.transpose().index])
y0 = y

# Modelo SVR
svr = SVR(kernel='poly', C=100, gamma='scale', epsilon=0.1)
svr.fit(X, y)

# Predições
y_pred = svr.predict(X)
y1 = y_pred

# Avaliações
rmsep = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Previsão com validação cruzada Leave-One-Out
loo = LeaveOneOut()
y_loo_pred = cross_val_predict(svr, X, y, cv=loo).ravel()
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
plt.title("SVR - Creatina - Real vs Predito")
# Inserção do R² e RMSEP no gráfico
plt.text(min(y), max(y), f"R² = {r2:.4f}", 
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
plt.grid(True)
plt.tight_layout()
plt.show()


# Reduz para 1D com PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# Mostra a variância explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_

# Padroniza a saída
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
# Treina o SVR sobre a dimensão 1D
svr = SVR(kernel='poly', C=100, epsilon=0.1)
svr.fit(X_pca, y)

# Geração da curva
X_grid = np.linspace(X_pca.min(), X_pca.max(), 500).reshape(-1, 1)
y_pred = svr.predict(X_grid)
epsilon = svr.epsilon
y_upper = y_pred + epsilon
y_lower = y_pred - epsilon

# Gráfico
plt.figure(figsize=(10, 6))
plt.scatter(X_pca, y, color='blue', label='Dados projetados (PCA)')
plt.plot(X_grid, y_pred, color='black', label='Regressão SVR')
plt.plot(X_grid, y_upper, 'r--', label='+ε')
plt.plot(X_grid, y_lower, 'r--', label='-ε')
plt.fill_between(X_grid.ravel(), y_lower, y_upper, color='red', alpha=0.1, label='ε-tube')
plt.xlabel("Componente principal 1 (PCA)")
plt.ylabel("y (padronizado)")
plt.title("SVR com ε-tube após redução com PCA")
plt.legend()
plt.show()


#Gráfico EJCR
from scipy.stats import f
from sklearn.linear_model import LinearRegression
# Regressão linear: y_true em função de y_pred
X = y1.reshape(-1, 1)
y = y0.reshape(-1, 1)

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


