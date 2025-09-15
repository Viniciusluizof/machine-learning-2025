# %%

import pandas as pd
import matplotlib.pyplot as plt

# importanto modelos de machine learning
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df

df["aprovado"] = (df['nota'] > 5).astype(int)
df
# %%

# Plotando os dados em um gráfico
plt.plot(df['cerveja'],df['aprovado'],'o',color='royalblue')
plt.grid(True)
plt.title("Cervejas vs Aprovação")
plt.xlabel("Cervejas")
plt.ylabel("Aprovado")
plt.show()
# %%

# regressão logistica
reg = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)
reg.fit(df[['cerveja']],df['aprovado'])
# Aqui eles esta prevendo os resultados com base nos dados
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())

# Aqui está calculando a probalidade
reg_proba = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.figure(dpi=400)
plt.plot(df['cerveja'],df['aprovado'],'o',color='royalblue')
plt.grid(True)
plt.title("Cervejas vs Aprovação")
plt.xlabel("Cervejas")
plt.ylabel("Aprovado")
plt.hlines(0.5,xmin=1,xmax=9,linestyles='--',colors='Black')
plt.plot(df['cerveja'].drop_duplicates(),reg_predict,color='tomato')
plt.plot(df['cerveja'].drop_duplicates(),reg_proba,color='red')
plt.legend(["Obsevação",
            "Reg Predict", 
            "Reg Proba"])
plt.show()
# %%

# arvore de decisão sem profundidade definida
arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']],df['aprovado'])
arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_proba = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.figure(dpi=400)
plt.plot(df['cerveja'],df['aprovado'],'o',color='royalblue')
plt.grid(True)
plt.title("Cervejas vs Aprovação")
plt.xlabel("Cervejas")
plt.ylabel("Aprovado")
plt.hlines(0.5,xmin=1,xmax=9,linestyles='--',colors='Black')
plt.plot(df['cerveja'].drop_duplicates(),arvore_full_predict,color='green')
plt.plot(df['cerveja'].drop_duplicates(),arvore_full_proba,color='magenta')
plt.legend(["Obsevação",
            "Base line",
            "Arvore Full Predict", 
            "Arvore Full Proba"])
plt.show()
# %%
# Arvore de decisão com profundidade definida (criterio de parada)
arvore_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
arvore_d2.fit(df[['cerveja']],df['aprovado'])
arvore_d2_predict = arvore_d2.predict(df[['cerveja']].drop_duplicates())
arvore_d2_proba = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.figure(dpi=400)
plt.plot(df['cerveja'],df['aprovado'],'o',color='royalblue')
plt.grid(True)
plt.title("Cervejas vs Aprovação")
plt.xlabel("Cervejas")
plt.ylabel("Aprovado")
plt.hlines(0.5,xmin=1,xmax=9,linestyles='--',colors='Black')
plt.plot(df['cerveja'].drop_duplicates(),arvore_d2_predict,color='blue')
plt.plot(df['cerveja'].drop_duplicates(),arvore_d2_proba,color='black')
plt.legend(["Obsevação",
            "Base line",
            "Arvore d2 Predict", 
            "Arvore d2 Proba"])
plt.show()

# %%

# usando modelo naive bayes
nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']],df['aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_proba = nb.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.figure(dpi=400)
plt.plot(df['cerveja'],df['aprovado'],'o',color='royalblue')
plt.grid(True)
plt.title("Cervejas vs Aprovação")
plt.xlabel("Cervejas")
plt.ylabel("Aprovado")
plt.hlines(0.5,xmin=1,xmax=9,linestyles='--',colors='Black')
plt.plot(df['cerveja'].drop_duplicates(),nb_predict, color='blue')
plt.plot(df['cerveja'].drop_duplicates(),nb_proba,color='green')
plt.legend(["Obsevação",
            "Base line",
            "Naive Bayes - predict",
            "Naive Bayes - probabilidade"
           ])
plt.show()
# %%
