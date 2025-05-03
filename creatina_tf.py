import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
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

# Avaliação
rmsecv = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSECV: {rmsecv:.6f}")
