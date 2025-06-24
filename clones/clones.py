# %%

import pandas as pd

df = pd.read_parquet('dados/dados_clones.parquet')
# %%

features = ['Massa(em kilos)','Estatura(cm)' , 
          'Distância Ombro a ombro', 'Tamanho do crânio', 'Tamanho dos pés', 'Tempo de existência(em meses)']

target = 'Status '

x = df[features]
y = df[target]

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(x,y)
# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True,
               max_depth=3
               )
#max_depth limita o tamanho da arvore