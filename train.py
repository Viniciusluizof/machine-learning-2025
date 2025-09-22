# %%

import pandas as pd

df = pd.read_csv(r"data\abt_churn.csv")
df.head()
# %%
# SAMPLE 
"""
Esta etapa envolve a escolha de um subconjunto do conjunto 
de dados de volume apropriado a partir de um vasto conjunto de dados 
fornecido para a construção do modelo. O objetivo desta etapa inicial do 
processo é identificar variáveis ​​ou fatores (dependentes e independentes) 
que influenciam o processo. As informações coletadas são então 
classificadas em categorias de preparação e validação.
"""
# Aqui definimos o Out of Time 
# (conjunto de dados mais recentes para validar o modelo)

oot = df[df['dtRef']==df['dtRef'].max()].copy()
df_train = df[df['dtRef']<df['dtRef'].max()].copy()
# %%

# Antes de definimos treino e teste é interessante definir o que é variável de interesse

# Essas são as variáveis
features = df_train.columns[2:-1]

# Essa é nossa target
target = 'flagChurn'

X, y = df_train[features], df_train[target]
# %%
## SAMPLE

from sklearn import model_selection

# Definindo treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    random_state=43,
                                                                    test_size=0.2,
                                                                    stratify=y)


print("Taxa variével resposta Geral:",y.mean())
print("Taxa variével resposta Treino:",y_train.mean())
print("Taxa variével resposta Teste:",y_test.mean())

"""
Se as variáveis são raras é preciso fazer balanceamento das amostras
(under samplyng)
"""
# %%
# EXPLORE
"""
A partir daqui só usamos a base de treino
"""

#Identificando os valores nulos
X_train.isna().sum().sort_values(ascending=False)
# %%
# Analise bivariada
"""
Consiste em descobri quais variáveis interferem na variável resposta
"""
df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by =target).agg(["mean","median"]).T
sumario

sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_relativa'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diff_relativa'],ascending=False)


# %%

from sklearn import tree
import matplotlib.pyplot as plt


arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train,y_train)

# Identifica a importância de cada variável
feature_importance = pd.Series(arvore.feature_importances_, index=X_train.columns).sort_values(ascending=False).reset_index()
feature_importance['acumum.'] = feature_importance[0].cumsum()
feature_importance
# %%
