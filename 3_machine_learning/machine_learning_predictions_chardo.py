##########################################################################################
# IMPORT LIBRAIRIES
##########################################################################################

import numpy as np
import pandas as pd

import re

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


##########################################################################################
# IMPORT DONNEES
##########################################################################################

df_wine = pd.read_csv('data/clean_wine.csv')
df_domaine = pd.read_csv('data/clean_domaine.csv')



##########################################################################################
# SELECTION DES DONNÉES
##########################################################################################

######
# SELECTION SEPAGE CHARDONNAY
######

df_chardonnay = df_wine.loc[df_wine['variety'] == 'Chardonnay']

######
# SELECTION PAYS FRANCE
######

df_chardonnay_fr = df_chardonnay.loc[df_chardonnay['country'] == 'France']

######
# SELECTION DES PRIX
######

"""
# D'après le dashboard : on a de gros outliers pour les prix au dessus de 382, on les supprime.
"""

# Recherche de données à supprimer
df_prix_to_drop = df_chardonnay_fr.loc[df_chardonnay_fr['price'] > 382]
# Index de ces données
index_prix_to_drop = df_prix_to_drop.index
list_prix_to_drop = index_prix_to_drop.to_list()
# Suppression des données où millesime inférieur à 2000 ou supérieur à 2023 (241 lignes)
df_chardonnay_fr.drop(index=list_prix_to_drop,
                      inplace=True)
# Verification : OK, suppression de 9 vins
df_chardonnay_fr.describe()

##########################################################################################
# PREPARATION DES DONNÉES
##########################################################################################

df_chardonnay_fr.drop(columns='Unnamed: 0',
                      inplace=True)

# Encodage des provinces
df_chardonnay_fr['province'].value_counts()
df_chardonnay_fr = pd.concat([df_chardonnay_fr, df_chardonnay_fr['province'].str.get_dummies()],
                             axis = 1)

# Déclaration des données d'entrées et de sorties
X = df_chardonnay_fr.select_dtypes('number').drop(columns='price')
y = df_chardonnay_fr['price']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    train_size = 0.75,
                                                    random_state = 42)
print("The length of the initial dataset is :", len(X))
print("The length of the train dataset is   :", len(X_train))
print("The length of the test dataset is    :", len(X_test))

# Standardisation
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


##########################################################################################
# ENTRAINEMENT ML
##########################################################################################

######
# GRADIENT BOOSTING REGRESSOR
######

modelGBR = GradientBoostingRegressor().fit(X_train_scaled, y_train)
print(modelGBR.score(X_train_scaled, y_train))
print(modelGBR.score(X_test_scaled, y_test))

# Scores : 0.71 et 0.68
# Scores df_pinot_noir_fr_burg : 0.70 et 0.58
# Scores df_pinot_noir_country : 0.74 et 0.63
# PINOT : 0,71 et 0,68
# CHARDO : 0.77 et 0.62

######
# GRIDSEARCH
######

#dico = {'max_depth' : range(1, 10),
#        'min_samples_leaf' : range(1, 10),
#        'min_samples_split' : range(1, 10)}
#grid = GridSearchCV(GradientBoostingRegressor(), dico)
#grid.fit(X_train_scaled,y_train)

#print("best score:",grid.best_score_)
#print("best parameters:",grid.best_params_)

# Best score : 0.63 donc je laisse tomber... C'est peut-être le cross-validation ?

######
# APPLICATION
######

#modelDTR = DecisionTreeRegressor(min_samples_leaf=4, min_samples_split=4).fit(X_train_scaled, y_train)
#modelDTR.score(X_train_scaled, y_train)
#modelDTR.score(X_test_scaled, y_test)


##########################################################################################
# PREDICTION
##########################################################################################

######
# PREPARATION DF DOMAINE ML POUR PREDICTION
######

# Suppression colonnes
df_domaine.drop(columns='Unnamed: 0',
                inplace=True)

# Suppression du chardonnay
df_domaine = df_domaine.loc[df_domaine['variety'] == 'Chardonnay']

# Copie du df
df_domaine_ml = df_domaine.copy()

# Ajout colonnes nécessaires
list_columns_to_add = df_chardonnay_fr.columns[11:]
for column in list_columns_to_add:
    df_domaine_ml[str(column)] = 0
df_domaine_ml['Burgundy'] = 1

# Données d'entrée
X_domaine = df_domaine_ml.select_dtypes('number')

# Standardisation
X_domaine_scaled = scaler.transform(X_domaine)


######
# PREDICTION
######

df_domaine['suggested_price'] = modelGBR.predict(X_domaine_scaled)

# Préparation df à présenter
df_domaine['suggested_price'] = df_domaine['suggested_price'].apply(lambda x : round(x))
df_domaine_final = df_domaine[['variety', 'title', 'points', 'millesime', 'suggested_price']]

##########################################################################################
# PREPARATION DF FINAL AVEC PRIX SUGGERES
##########################################################################################

######
# REGEX POUR AVOIR BELLES DESIGNATIONS
######

# Recuperation de la designation du vin dans le title
motif = '(?<=\d\d\d\d\s)+.*'
df_domaine_final["title"] = df_domaine_final["title"].apply(lambda x: re.search(motif, x).group(0))

# Nettoyage espaces restants
motif = '^\s'
for index, row in df_domaine_final.iterrows():
    try:
        df_domaine_final.loc[index, 'title'] = re.sub(motif, '', df_domaine_final.loc[index, 'title'])
    except:
        pass

# Vérification
#df_chardonnay_fr.sort_values('price',
#                             inplace=True,
#                            ascending=False)
#df_domaine_final.sort_values('suggested_price',
#                             inplace=True,
#                             ascending=False)


##########################################################################################
# EXPORT DONNEE
##########################################################################################

df_domaine_final.to_csv('data/suggested_price_chardo.csv')
df_chardonnay_fr.to_csv('data/clean_wine_chardo_fr.csv')