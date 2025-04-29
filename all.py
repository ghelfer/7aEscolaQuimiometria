import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Lendo o CSV com a primeira coluna como índice
df = pd.read_csv("all.csv", index_col=0)

print(df.head())

df.plot()
plt.xlabel("cm-1")
plt.ylabel("abs")
plt.title("Creatina e Whey Protein - MIR")
plt.legend().set_visible(False)
plt.grid(False)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()

