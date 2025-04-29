import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
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
X_snv = snv(df.T.values)
X_savgol = savgol_filter(X_snv, window_length=9, polyorder = 2, deriv=1)
X_centered = X_savgol - np.mean(X_savgol, axis=0)

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


#Aplicar PLS com 6 componentes
pls = PLSRegression(n_components=6, scale=False)
pls.fit(X_centered, y)

#Obter predições
y_pred = pls.predict(X_centered)
print(y)
print(y_pred)

r2 = r2_score(y, y_pred)
rmsep = np.sqrt(mean_squared_error(y, y_pred))

# Previsão com validação cruzada Leave-One-Out
loo = LeaveOneOut()
y_loo_pred = cross_val_predict(pls, X_centered, y, cv=loo).ravel()
rmsecv = np.sqrt(mean_squared_error(y, y_loo_pred))

# Exibir métricas
print(f"R²:     {r2:.6f}")
print(f"RMSEP:  {rmsep:.6f}")
print(f"RMSECV: {rmsecv:.6f}")

#Visualizar real vs predito
plt.figure(figsize=(9, 6))
plt.scatter(y, y_pred, color='blue', edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Predito")
plt.title("Regressão PLS - Real vs Predito")
plt.grid(True)
plt.tight_layout()
plt.show()