# %%

import pandas as pd

df = pd.read_excel("dados/dados_cerveja.xlsx")
df
# %%

features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

x = df[features]
y = df[target]

x = x.replace({
    'mud':1, 'pint':0,
    'sim':1, 'não':0,
    'clara':1, 'escura':0,
})#transformando dados string em numeros

x

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(x, y)
# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, 
               feature_names=features,
               class_names=model.classes_,
               filled=True
               )
# %%
