# Recarregar bibliotecas e código após o reset do ambiente
import pandas as pd
import matplotlib.pyplot as plt

# Caminho do arquivo após novo upload
file_path = "rmsecv_fator.csv"

# Carregar o arquivo CSV
df = pd.read_csv(file_path, index_col=0)

# Garantir que o índice está corretamente interpretado como numérico (número de fatores)
df.index = pd.to_numeric(df.index)

# Encontrar os mínimos de cada coluna
min_creatina_x = df['Creatina'].idxmin()
min_creatina_y = df['Creatina'].min()

min_whey_x = df['Whey'].idxmin()
min_whey_y = df['Whey'].min()

# Plot
plt.figure(figsize=(9, 7))
plt.plot(df.index, df['Creatina'], label='Creatina', marker='o')
plt.plot(df.index, df['Whey'], label='Whey', marker='s')

# Linhas verticais nos mínimos
plt.axvline(min_creatina_x, color='blue', linestyle='--', alpha=0.6)
plt.axvline(min_whey_x, color='orange', linestyle='--', alpha=0.6)

# Pontos mínimos
plt.plot(min_creatina_x, min_creatina_y, 'bo')
plt.plot(min_whey_x, min_whey_y, 'or')

# Anotações
plt.text(min_creatina_x + 0.2, min_creatina_y, f'{min_creatina_y:.2f}', color='blue')
plt.text(min_whey_x + 0.2, min_whey_y, f'{min_whey_y:.2f}', color='orange')

plt.xlabel('Número de Fatores')
plt.ylabel('RMSECV')
plt.title('RMSECV por Número de Fatores - Whey e Creatina')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



