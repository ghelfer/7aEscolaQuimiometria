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
df = pd.read_csv("whey.csv", index_col=0)

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
    'Whey_a': 0.0,'Whey_b': 0.0,
    'W10a': 10.1, 'W10b': 10.1,
    'W20a': 19.3, 'W20b': 19.3,
    'W40a': 39.7, 'W40b': 39.7,
    'W60a': 60.0, 'W60b': 60.0,
    'W80a': 81.5, 'W80b': 81.5,
    'Amb': 100.0, 'Ama': 100.0,
}

y = np.array([targets[col] for col in df.transpose().index])


# Modelo SVR
svr = SVR(kernel='poly', C=100, gamma='scale', epsilon=0.1)
svr.fit(X, y)

# Predições
y_pred = svr.predict(X)

# Avaliações
rmsep = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Previsão com validação cruzada Leave-One-Out
loo = LeaveOneOut()
y_loo_pred = cross_val_predict(svr, X, y, cv=loo).ravel()
rmsecv = np.sqrt(mean_squared_error(y, y_loo_pred))

print(f"R²: {r2:.4f}")
print(f"RMSEP: {rmsep:.4f}")
print(f"RMSECV: {rmsecv:.6f}")

#Visualizar real vs predito
plt.figure(figsize=(9, 6))
plt.scatter(y, y_pred, color='blue', edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Predito")
plt.title("SVR - Whey Protein - Real vs Predito")
# Inserção do R² e RMSEP no gráfico
plt.text(min(y), max(y), f"R² = {r2:.4f}\nRMSEP = {rmsep:.4f}", 
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
plt.xlabel(f'PC1: {explained_variance[0]*100:.2f}%')
plt.ylabel("y (padronizado)")
plt.title("SVR com ε-tube após redução com PCA")
plt.legend()
plt.show()