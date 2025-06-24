# %%

import pandas as pd

df = pd.read_excel('dados/dados_frutas.xlsx')
df

# %%

from sklearn import tree #importando modelos de arvore

arvore = tree.DecisionTreeClassifier(random_state=42)
#o random state garante que o modelo seja o mesmo caso rodar em outra maquina
# %%

y = df['Fruta'] #todas as frutas estão em y

caracteristicas = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
x = df[caracteristicas] #caracteristicas das frutas em x

# %%

#ensinando para a maquina as frutas
arvore.fit(x, y)
# %%

#passando os parametros para encontrar alguma fruta
arvore.predict([[0,0,0,0]])

# %%

import matplotlib.pyplot as plt #importando lib que usa plot de grafico

plt.figure(dpi=400)#só para melhorar a qualidade da imagem

tree.plot_tree(arvore, 
               feature_names=caracteristicas, 
               class_names=arvore.classes_, 
               filled=True)

#desenhando o grafico
# %%

proba = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=arvore.classes_)

#ver a probabilidade de cada fruta em cada amostra
