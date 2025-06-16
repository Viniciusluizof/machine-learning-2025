# %%

import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")

df['General Jedi encarregado'].unique()

# %%

features = ['Massa(em kilos)', 'Estatura(cm)']

df.groupby('Status ')[features].mean()