# %%

import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()
# %%
from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']] # Isso é uma matriz (df)
y = df['nota'] # isso é um vetor (series)

# Isso aqui é o modelo, aprendizado de maquina
"""
linear_model.LinearRegression:
Ele cria um modelo de regressão linear, que é um dos algoritmos mais básicos e importantes de aprendizado de máquina supervisionado.

👉 Em resumo, ele:

- Ajusta uma reta (ou hiperplano, no caso de múltiplas variáveis) aos dados para modelar a relação entre variáveis independentes (X) e a variável dependente (y).
- Calcula os coeficientes (pesos) e o intercepto que melhor explicam a relação entre as variáveis, minimizando o erro quadrático médio (MSE).
- Depois, pode ser usado para fazer previsões em novos dados.
"""
reg = linear_model.LinearRegression(fit_intercept=True)

"""
.fit() no scikit-learn é sempre usado para treinar/ajustar o modelo aos dados.

No caso de LinearRegression, ele faz o seguinte:
- Recebe os dados de entrada (X) → as variáveis independentes (features).
- Recebe o alvo (y) → a variável dependente (o que você quer prever).
- Calcula os coeficientes (pesos) e o intercepto que melhor se ajustam aos dados, minimizando o erro quadrático médio (método dos mínimos quadrados).
- Armazena esses valores dentro do objeto do modelo (reg.coef_ e reg.intercept_).
"""
reg.fit(X,y)

# Aqui a gente descobre qual o valor de A e B 
# Da formula f(x) = a + bx
a,b = reg.intercept_, reg.coef_[0]

# Aqui estamos prevendo y com base em X
predict_reg = reg.predict(X.drop_duplicates())


# Aqui fazemos para a arvore de decisão
# Aqui ela está ajustada 100% em cima dos dados
arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X,y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

# Aqui fazemos para a arvore de decisão
# Aqui você está limitando até onde a arvore vai quebrar
arvore_d2 = tree.DecisionTreeRegressor(random_state=42, 
                                       max_depth=2)
arvore_d2.fit(X,y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

# Plotamos a predição
plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.plot(X.drop_duplicates()['cerveja'],
         predict_reg)

plt.plot(X.drop_duplicates()['cerveja'],predict_arvore_full)
plt.plot(X.drop_duplicates()['cerveja'],predict_arvore_d2)

plt.legend(['Observado', 
            f'y = {a:.3f} + {b:.3f}x',
            'Arvore Full',
            'Arvore Depth = 2'])
plt.show()
# %%

plt.figure(dpi=400)
tree.plot_tree(arvore_d2,
               feature_names=['cerveja'],
               filled=True)
# %%
