import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement des données
Df_X = pd.read_csv('Data_X.csv')
Df_Y = pd.read_csv('Data_Y.csv')

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

# Suppression des valeurs manquantes
DfNew_X.dropna(inplace=True)

print(DfNew_X)
