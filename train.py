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
best_features = feature_importance[feature_importance['acumum.']< 0.96]['index'].tolist()
best_features
# %%

# MODIFY

# Padronização - os dados precisam estar na mesma escala
# Pre-processing
'''
Nomalização
- Pegar a media, subtrair os valores da media e dividir por sigma

padronização min max
- Pega o x1 menos min(x1) dividido por (max(x1)-min(x1))
'''

from feature_engine import discretisation, encoding
from sklearn import pipeline

tree_discr = discretisation.DecisionTreeDiscretiser(variables=best_features, 
                                                    regression=False, 
                                                    bin_output='bin_number',
                                                    cv=3)

# Onehot
onehot = encoding.OneHotEncoder(variables=best_features,ignore_format=True)

# %%
# MODEL 

from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble

# model = linear_model.LogisticRegression(penalty=None, random_state=42,max_iter=100000)
# model = naive_bayes.BernoulliNB()
# model = ensemble.RandomForestClassifier(random_state=42,
#                                         min_samples_leaf=20
#                                         n_jobs=-1,
#                                         n_estimators=500
#                                         )
# model = tree.DecisionTreeClassifier(random_state=42,min_samples_leaf=20)
model = ensemble.AdaBoostClassifier(random_state=42,
                                    n_estimators=500,
                                    learning_rate=0.99)




model_pipeline = pipeline.Pipeline(
    steps=[
        ('Discretizar',tree_discr),
        ('Onehot',onehot),
        ('Model',model)
    ]
)

import mlflow
from sklearn import metrics

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id=196164843398636059)

with mlflow.start_run(run_name=model.__str__()):
    mlflow.sklearn.autolog()
    model_pipeline.fit(X_train[best_features],y_train)

# ASSESS


#Aqui é na lista de exercicio
    y_train_predic = model_pipeline.predict(X_train[best_features])
    y_train_proba = model_pipeline.predict_proba(X_train[best_features])[:,1]

    acc_train = metrics.accuracy_score(y_train, y_train_predic)
    auc_train = metrics.roc_auc_score(y_train, y_train_proba)
    roc_train = metrics.roc_curve(y_train,y_train_proba)

    print("Acuracia Treino", acc_train)
    print("AUC Treino",auc_train)


    # Aqui é a prova

    y_test_predic = model_pipeline.predict(X_test[best_features])
    y_test_proba = model_pipeline.predict_proba(X_test[best_features])[:,1]

    acc_test = metrics.accuracy_score(y_test, y_test_predic)
    auc_test = metrics.roc_auc_score(y_test, y_test_proba)
    roc_teste = metrics.roc_curve(y_test,y_test_proba)


    print("Acuracia Teste", acc_test)
    print("AUC Teste",auc_test)

    # OOT

    y_oot_predic = model_pipeline.predict(oot[best_features])
    y_oot_proba = model_pipeline.predict_proba(oot[best_features])[:,1]

    acc_oot = metrics.accuracy_score(oot[target], y_oot_predic)
    auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
    roc_oot = metrics.roc_curve(oot[target],y_oot_proba)

    print("Acuracia oote", acc_oot)
    print("AUC oote",auc_oot)

    mlflow.log_metrics({
    "acc_train":acc_train,
    "auc_train":auc_train,
    "acc_test":acc_test,
    "auc_test":auc_test,
    "acc_oot":acc_oot,
    "auc_oot":auc_oot
    }
    )
# %%
plt.figure(dpi=400)
plt.plot(roc_train[0],roc_train[1])
plt.plot(roc_teste[0],roc_teste[1])
plt.plot(roc_oot[0],roc_oot[1])
plt.plot([0,1],[0,1],'--',color='black')
plt.grid(True)
plt.ylabel('Sensibilidade')
plt.xlabel('1 - Especificidade')
plt.title("Curva ROC")
plt.legend([
    f"Treino: {100*auc_train:.2f}",
    f"Teste: {100*auc_test:.2f}",
    f"Out-of-time: {100*auc_oot:.2f}"]
    
)
plt.show()
# %%
