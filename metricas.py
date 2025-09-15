# %%

import pandas as pd

df = pd.read_csv(r"C:\Users\vlof\Desktop\machine-learning-2025\data\Dados Comunidade (respostas) - dados.csv")
df.head()
# %%
df = df.replace({'Sim':1,'Não':0})
df
# %%

num_vars = [
    "Curte games?",
    "Curte futebol?",
    "Curte livros?",
    "Curte jogos de tabuleiro?",
    "Curte jogos de fórmula 1?",
    "Curte jogos de MMA?",
    "Idade",
]

dummy_vars = [
    "Como conheceu o Téo Me Why?",
    "Quantos cursos acompanhou do Téo Me Why?",
    "Estado que mora atualmente",
    "Área de Formação",
    "Tempo que atua na área de dados",
    "Posição da cadeira (senioridade)",
]
# transforma variáveis categoricas em numericas
df_analise =pd.get_dummies(df[dummy_vars]).astype(int)

#Adicionando as variaveis numericas no df analise
df_analise[num_vars] = df[num_vars].copy()

#Adicionando a variável resposta (a que queremos descobrir)
df_analise["pessoa feliz?"] = df["Você se considera uma pessoa feliz?"].copy()
df_analise

# %%

#importando o modelo que vamos usar
from sklearn import tree

# Definindo nossas variáveis
features = df_analise.columns[:-1].to_list()
X = df_analise[features]
y = df_analise['pessoa feliz?']

arvore = tree.DecisionTreeClassifier(random_state=42,
                                     min_samples_leaf=5) # min_samples_leaf= Na minha folha final tenho que ter 5 amostras

# Adicionando nossas variáveis
arvore.fit(X,y)
# %%

# Aqui vamos fazer a predição para vermos quanto nosso modelo esta aderente ou não

arvore_predict = arvore.predict(X)

df_predict = df_analise[["pessoa feliz?"]].rename(columns={"pessoa feliz?": "valor_verdadeiro"})
df_predict['predict_arvore'] = arvore_predict
df_predict
# %%

# Vemos o quanto o modelo acertou porém ele não da uma ideia de onde estamos acertando ou errando

# Acuracia - proporção de acertos
(df_predict['valor_verdadeiro'] == df_predict["predict_arvore"]).mean()

# %%
# para uma analise mais precisa usamos matriz de confusão

pd.crosstab(df_predict['valor_verdadeiro'],df_predict['predict_arvore'])
# %%

from sklearn import metrics

acc_arvore =  metrics.accuracy_score(df_predict['valor_verdadeiro'], df_predict['predict_arvore'])
precisao_arvore = metrics.precision_score(df_predict['valor_verdadeiro'], df_predict['predict_arvore'])
recall_arvore = metrics.recall_score(df_predict['valor_verdadeiro'], df_predict['predict_arvore'])
roc = metrics.roc_curve(df_predict['valor_verdadeiro'], df_predict['predict_arvore'])

# %%
import matplotlib.pyplot as plt

plt.plot(roc[0],roc[1])
# %%
