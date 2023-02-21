##########################################################################################
# IMPORT LIBRAIRIES
##########################################################################################

import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

##########################################################################################
# IMPORT DONNEES
##########################################################################################

df_wine = pd.read_csv('data/wine.csv')
df_domaine = pd.read_csv('https://raw.githubusercontent.com/murpi/wilddata/master/domaine_des_croix.csv')


##########################################################################################
# EXPLO DF DOMAINE
##########################################################################################

df_domaine.info()

"""
A nettoyer :
- Supprimer colonnes inutiles
- Récupérer le millésime
- Récupérer la désignation si vide ? (équivaut au nom de la région ?)
"""

##########################################################################################
# EXPLO DF WINE
##########################################################################################

df_wine.info()

df_wine.describe()

######
# VALEURS MANQUANTES
######

# Colonnes avec nans
nans_columns = df_wine.columns[df_wine.isna().any()].tolist()

# Lignes avec nans
nans_rows = df_wine.loc[:, nans_columns]

# Nombre de nans par colonnes
for colonne in df_wine.columns:
    print(f'{colonne} : ', end='')
    print(df_wine[str(colonne)].isna().sum())

######
# PREMIERS VISUELS
######

# Heatmap
sns.heatmap(df_wine.corr(),
            cmap='coolwarm',
            center=0,
            annot=True
            )
plt.title('Correlation Heatmap')
plt.show()

# Violinplot prix
fig1 = px.violin(df_wine,
                 y="price",
                 box=True,
                 title="Répartition des prix")
fig1.show()

# Violinplot notes
fig2 = px.violin(df_wine,
                 y="points",
                 box=True,
                 title="Répartition des notes")
fig2.show()

# Violinplot prix France et USA
df_US_fr = df_wine.loc[(df_wine['country'] == 'US') | (df_wine['country'] == 'France')]
fig3 = px.violin(df_US_fr,
                 y="price",
                 color='country',
                 box=True,
                 title="Répartition des prix entre la France et les USA")
fig3.show()


"""
A nettoyer :
- Suppression colonnes inutiles : region_2, taster_name, taster_twitter_handle (permettra de gérer bcp de nans)
- Récupérer millésime
- Suppression des nans dans 'price' et 'region1' :
    --> Sans prix et region_1 on ne pourra pas travailler les données
    --> Voir autres colonnes : à voir après premier traitement de la table
- Récupérer la désignation si vide ? (équivaut au nom de la région ?)

Première analyse :
- Notes entre 80 et 100, moyenne et mediane de 88
- Prix entre 2 et 1902 $, moyenne de 36 et mediane de 29, visiblement on a des vins très très chers qui représentent des outliers
- Province = region pour certaines données
"""