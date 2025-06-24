#%%
import pandas as pd

df = pd.read_excel('dados/dados_cerveja_nota.xlsx')
df.head()
# %%

from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']]
y = df[['nota']]

regressao = linear_model.LinearRegression()
regressao.fit(X,y)
# %%

a, b = regressao.intercept_, regressao.coef_[0]
a = a.item()
b = b.item()

predicao = regressao.predict(X.drop_duplicates())

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X,y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

arvore_dp2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore_dp2.fit(X,y)
predict_arvore_dp2 = arvore_dp2.predict(X.drop_duplicates())

#%%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title('Relacao cerveja vs Nota')
plt.xlabel('Cerveja')
plt.ylabel('Nota')

plt.plot(X.drop_duplicates()['cerveja'], predicao)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_dp2)

plt.legend(['Observado', f'y = {a:.3f} + {b:.3f} x'
            'Arvore Full',
            'Arvore Depth = 2'
            ])

#%%

plt.figure(dpi=400)

tree.plot_tree(arvore_dp2,
               feature_names=['cerveja'],
               filled=True
               )