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
df_analise.columns

# %%

#importando o modelo que vamos usar
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model

# Definindo nossas variáveis
features = df_analise.columns[:-1].to_list()
X = df_analise[features]
y = df_analise['pessoa feliz?']

arvore = tree.DecisionTreeClassifier(random_state=42,
                                     min_samples_leaf=5) # min_samples_leaf= Na minha folha final tenho que ter 5 amostras

# Adicionando nossas variáveis
arvore.fit(X,y)


naive = naive_bayes.GaussianNB()
naive.fit(X,y)

reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)
reg.fit(X,y)
# %%

# Aqui vamos fazer a predição para vermos quanto nosso modelo esta aderente ou não

arvore_predict = arvore.predict(X)

df_predict = df_analise[["pessoa feliz?"]].rename(columns={"pessoa feliz?": "valor_verdadeiro"}).copy()
df_predict['predict_arvore'] = arvore_predict
df_predict['proba_arvore'] = arvore.predict_proba(X)[:,1]

df_predict['predict_naive'] = naive.predict(X)
df_predict['proba_naive'] = naive.predict_proba(X)[:,1]

df_predict['predict_reg'] = reg.predict(X)
df_predict['proba_reg'] = reg.predict_proba(X)[:,1]

# %%

from sklearn import metrics

acc_arvore =  metrics.accuracy_score(df_predict['valor_verdadeiro'], df_predict['predict_arvore'])
precisao_arvore = metrics.precision_score(df_predict['valor_verdadeiro'], df_predict['predict_arvore'])
recall_arvore = metrics.recall_score(df_predict['valor_verdadeiro'], df_predict['predict_arvore'])
roc_arvore = metrics.roc_curve(df_predict['valor_verdadeiro'], df_predict['proba_arvore'])
auc_arvore = metrics.roc_auc_score(df_predict['valor_verdadeiro'], df_predict['proba_arvore'])
auc_arvore


acc_naive =  metrics.accuracy_score(df_predict['valor_verdadeiro'], df_predict['predict_naive'])
precisao_naive = metrics.precision_score(df_predict['valor_verdadeiro'], df_predict['predict_naive'])
recall_naive = metrics.recall_score(df_predict['valor_verdadeiro'], df_predict['predict_naive'])
roc_naive = metrics.roc_curve(df_predict['valor_verdadeiro'], df_predict['proba_naive'])
auc_naive = metrics.roc_auc_score(df_predict['valor_verdadeiro'], df_predict['proba_naive'])
auc_naive

acc_reg =  metrics.accuracy_score(df_predict['valor_verdadeiro'], df_predict['predict_reg'])
precisao_reg = metrics.precision_score(df_predict['valor_verdadeiro'], df_predict['predict_reg'])
recall_reg = metrics.recall_score(df_predict['valor_verdadeiro'], df_predict['predict_reg'])
roc_reg = metrics.roc_curve(df_predict['valor_verdadeiro'], df_predict['proba_reg'])
auc_reg = metrics.roc_auc_score(df_predict['valor_verdadeiro'], df_predict['proba_reg'])
auc_reg
# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)
plt.plot(roc_arvore[0],roc_arvore[1],"-o")
plt.plot(roc_naive[0],roc_naive[1],"-o")
plt.plot(roc_reg[0],roc_reg[1],"-o")
plt.grid(True)
plt.title("ROC Curve")
plt.xlabel("1 - Especificidade")
plt.ylabel("Recall")
plt.legend([
    f"Arvore: {auc_arvore:.2f}",
    f"Naive: {auc_naive:.2f}",
    f"Reg log: {auc_reg:.2f}"])
plt.show()

# %%

pd.Series({"model":reg,"features":features}).to_pickle("model_feliz.pkl")
# %%
