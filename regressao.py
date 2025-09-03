# %%

import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()
# %%
from sklearn import linear_model

X = df[['cerveja']] # Isso é uma matriz (df)
y = df['nota'] # isso é um vetor (series)

# Isso aqui é o modelo, aprendizado de maquina

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X,y)
# %%
reg.coef_
# %%
a,b = reg.intercept_, reg.coef_[0]

print(a,b)
# %%
predict = reg.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.plot(X.drop_duplicates()['cerveja'],
         predict)
plt.legend(['Observado', f'y = {a:.3f} + {b:.3f}x'])

plt.show()
# %%
