# %%

import pandas as pd

from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_parquet("data/dados_clones.parquet")
df
# %%
features = ["Massa(em kilos)","Estatura(cm)","Tempo de existência(em meses)"]
target = 'Status '

x= df[features]
y= df[target]
# %%
model = tree.DecisionTreeClassifier()
model.fit(X=x,y=y)
# %%
plt.figure(dpi = 400)

tree.plot_tree(
    model,
    feature_names=features,
    class_names=model.classes_,
    filled=True, max_depth=3
)

plt.show()
# %%
df.groupby(['Status '])[features].mean()
# %%
