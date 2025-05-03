import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
df = pd.read_csv("creatina.csv", index_col=0)

print(df.head())

# Padronização Snv, Golay e Mean-center
X = df.T.values
X_snv = df.T.values
X_savgol = savgol_filter(X_snv, window_length=9, polyorder = 2, deriv=1)
X_mean_centered = X_savgol - np.mean(X_savgol, axis=0)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_mean_centered)


explained_var = pca.explained_variance_ratio_ * 100  # em %

# Gráfico PCA

# Agrupamento: usar o segundo caractere do nome da amostra
sample_names = df.columns

# Agrupamento pelo 2º caractere do nome da amostra
groups = [name[1] for name in sample_names]

# Montar DataFrame com resultado do PCA e grupo
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['grupo'] = groups
df_pca['amostra'] = sample_names

# Plot com rótulos de amostra
plt.figure(figsize=(9, 6))
for grupo in sorted(df_pca['grupo'].unique()):
    subset = df_pca[df_pca['grupo'] == grupo]
    plt.scatter(subset['PC1'], subset['PC2'])
    for _, row in subset.iterrows():
        plt.text(row['PC1'], row['PC2'], row['amostra'], fontsize=8, ha='right', va='center')

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.title("PCA - Projeção Creatina adulterada")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()
