import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
Df_X = pd.read_csv('./Data/Data_X.csv')
Df_Y = pd.read_csv('./Data/Data_Y.csv')

# Fusion des ensembles de données
DfNew_X = pd.merge(Df_X, Df_Y)

# Selection d'un sous-ensemble ici Country
print(DfNew_X['COUNTRY'])

# Calcul de nouveaux attributs
DfNew_X['FR_cons - net_import'] = DfNew_X['FR_CONSUMPTION'] - \
    DfNew_X['FR_NET_IMPORT']
print('FR_cons - net_import', DfNew_X['FR_cons - net_import'])

# Trie des datas
DfNew_X.sort_index(inplace=True)

# Vérification des valeurs manquantes dans les données
print("Nombre de valeurs manquantes par colonne :")
print(DfNew_X.isnull().sum())


# Suppression des valeurs manquantes
DfNew_X.dropna(inplace=True)

print(DfNew_X)

# Séparation en variables explicatives et variable cible
X = DfNew_X.drop(['ID', 'TARGET'], axis=1)
y = DfNew_X['TARGET']

# Fractionnement des données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Vérification des valeurs manquantes dans les données
# print("Nombre de valeurs manquantes par colonne :")
# print(DfNew_X.isnull().sum())

# Vérification de la possible comparaison des données
print(X_train.describe())

# Visualisation du type de chaque variable
print(X_train.dtypes)

# # Tracer un histogramme pour chaque variable explicative
# for col in X_train.columns:
#     plt.figure(figsize=(5, 4))
#     sns.histplot(X_train[col])
#     plt.title(col, fontsize=12)
#     plt.xlabel("Valeur")
#     plt.ylabel("Fréquence")
#     plt.xlim(X_train[col].min(), X_train[col].max())
#     plt.show()

# # Visualisation de la distribution de chaque variable en 1 onglet
# fig, axs = plt.subplots(ncols=3, nrows=12, figsize=(20, 60))
# index = 0
# for i, col in enumerate(X_train.columns):
#     sns.histplot(X_train[col], ax=axs[i//3, i % 3])
#     axs[i//3, i % 3].set_title(col, fontsize=12)
#     axs[i//3, i % 3].set_xlabel("Valeur")
#     axs[i//3, i % 3].set_ylabel("Fréquence")
# plt.tight_layout()
# plt.show()
"""
# Tracer un diagramme en boîte pour chaque variable explicative
for col in X_train.columns:
    plt.figure(figsize=(5, 4))
    sns.boxplot(x=y_train, y=X_train[col])
    plt.title(col, fontsize=12)
    plt.xlabel("Cible")
    plt.ylabel("Valeur")
    plt.show()
"""
