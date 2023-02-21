##########################################################################################
# IMPORT LIBRAIRIES
##########################################################################################

import numpy as np
import pandas as pd
import re
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


##########################################################################################
# IMPORT DONNEES
##########################################################################################

df_wine = pd.read_csv('data/wine.csv')


##########################################################################################
# NETTOYAGE
##########################################################################################

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

######
# Suppression colonnes inutiles
######

df_wine.drop(columns=['region_2', 'taster_name', 'taster_twitter_handle'],
             inplace=True)



##########################################################################################
# RECUPERATION MILLESIMES
##########################################################################################

# Motif Regex
motif_millesime = '\d\d\d\d'

# Création colonne vide pour entrer les millesimes
df_wine['millesime'] = np.nan

# On récupère les millesimes avec une boucle, pour pouvoir gérer les erreurs
for index, row in df_wine.iterrows():
    try:
        df_wine.loc[index, 'millesime'] = re.search(motif_millesime, df_wine.loc[index, 'title']).group(0)
    except:
        pass

# Suppression des nans dans colonne millesime
df_wine.dropna(subset='millesime',
               inplace=True)

# Conversion de la colonne numérique au format interger
df_wine[['millesime']] = df_wine[['millesime']].astype(int)

# Recherche de dates potentiellement problématiques
df_millesime_pb = df_wine.loc[(df_wine['millesime'] > 2023) (df_wine['millesime'] < 2000)]
# Index de ces données
millesime_pb_index = df_millesime_pb.index
millesime_pb_index = millesime_pb_index.to_list()
# Suppression des données où millesime inférieur à 2000 ou supérieur à 2023 (241 lignes)
df_wine.drop(index=millesime_pb_index,
             inplace=True)
# Verification : OK
#df_wine.loc[(df_wine['millesime'] > 2023) | (df_wine['millesime'] < 2000)]

##########################################################################################
# GESTION DES NANS
##########################################################################################

######
# OBSERVATION DE CE QU'ON VA SUPPRIMER
######

# Nombre de nans par colonnes
for colonne in df_wine.columns:
    print(f'{colonne} : ', end='')
    print(df_wine[str(colonne)].isna().sum())

# Vérification que les nan dans les millesimes ne sont pas dus à des erreurs
df_nan_millesime = df_wine.loc[df_wine['millesime'].isna() == True]

# Regard sur quelques lignes avec nans avant suppression
df_nan_province = df_wine.loc[df_wine['province'].isna() == True]

######
# SUPPRESSION DE CERTAINS NANS
######

# Liste colonnes dans lesquelles on ne veut plus de nans
list_columns_nan = ['country', 'price', 'variety', 'province']

# Suppression
df_wine.dropna(subset=list_columns_nan,
               inplace=True)

# Conversion de la colonne numérique au format interger
df_wine[['millesime']] = df_wine[['millesime']].astype(int)

"""
Perte du dataset : environ 10% 
"""

##########################################################################################
# VISUELS
##########################################################################################

df_wine.info()
df_wine.describe()

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
                 y="millesime",
                 box=True,
                 title="Répartition des millesimes")
fig1.show()

##########################################################################################
# EXPORT DONNEE
##########################################################################################

df_wine.to_csv('data/clean_wine.csv')
