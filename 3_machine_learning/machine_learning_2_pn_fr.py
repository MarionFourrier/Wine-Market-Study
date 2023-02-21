##########################################################################################
# IMPORT LIBRAIRIES
##########################################################################################

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesRegressor

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

"""
CONCLUSION MACHINE LEARNING 1 :
- Les scores sont moins bons après le gridsearch...
- Les scores sont de toute manière assez mauvais. 
- Meilleur modèle : DecisionTreeRegressor
- Perspectives : 
    1) Enlever du dataset les données dont on ne va pas se servir et qui peuvent être outliers : 
        * cépages différents
        * pays différent
        * régions différentes ?
        * vins trop chers ?
    1) Retravailler le dataset et faire une matrice de corrélation. D'après l'explo, la région et le cépage doivent jouer.
"""

######
# SELECTION SEPAGE PINOT NOIR
######

df_pinot_noir = df_wine.loc[df_wine['variety'] == 'Pinot Noir']

######
# SELECTION PAYS FRANCE
######

df_pinot_noir_fr = df_pinot_noir.loc[df_pinot_noir['country'] == 'France']

######
# SELECTION DES PRIX
######

"""
# D'après le dashboard : on a de gros outliers pour les prix au dessus de 348, on les supprime.
"""

# Recherche de données à supprimer
df_prix_to_drop = df_pinot_noir_fr.loc[df_pinot_noir_fr['price'] > 348]
# Index de ces données
index_prix_to_drop = df_prix_to_drop.index
list_prix_to_drop = index_prix_to_drop.to_list()
# Suppression des données où millesime inférieur à 2000 ou supérieur à 2023 (241 lignes)
df_pinot_noir_fr.drop(index=list_prix_to_drop,
                      inplace=True)
# Verification : OK, suppression de 10 vins
df_pinot_noir_fr.describe()


######
# MATRICE CORRELATION
######

# Suppression colonne 1
df_pinot_noir_fr.drop(columns='Unnamed: 0',
                      inplace=True)

# Encodage des provinces
df_pinot_noir_fr['province'].value_counts()
df_pinot_noir_fr = pd.concat([df_pinot_noir_fr, df_pinot_noir_fr['province'].str.get_dummies()],
                             axis = 1)

# Heatmap
sns.heatmap(df_pinot_noir_fr.corr(),
            cmap='coolwarm',
            center=0,
            annot=True
            )
plt.title('Correlation Heatmap')
plt.show()

######
# EXPORT DONNEE
######

df_pinot_noir_fr.to_csv('clean_wine_pinot_fr.csv')

######
# SELECTION REGION
######

df_pinot_noir_fr_burg = df_pinot_noir_fr.loc[df_pinot_noir_fr['province'] == 'Burgundy']


######
# MATRICE CORRELATION
######

# Suppression colonnes
df_pinot_noir_fr_burg.drop(columns=['Alsace', 'Burgundy', 'Champagne', 'France Other', 'Languedoc-Roussillon', 'Loire Valley', 'Provence'],
                           inplace=True)

# Heatmap
sns.heatmap(df_pinot_noir_fr_burg.corr(),
            cmap='coolwarm',
            center=0,
            annot=True
            )
plt.title('Correlation Heatmap')
plt.show()


##########################################################################################
# PREPARATION DES DONNÉES
##########################################################################################

# Déclaration des données d'entrées et de sorties
X = df_pinot_noir_fr_burg.select_dtypes('number').drop(columns='price')
y = df_pinot_noir_fr_burg['price']

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
# TEST DE PLUSIEURS MODELES DE ML
##########################################################################################

######
# LINEAR REGRESSION
######

modelLR = LinearRegression().fit(X_train_scaled, y_train)
modelLR.score(X_train_scaled, y_train)
modelLR.score(X_test_scaled, y_test)

# Scores df_pinot_noir_fr: 0.53 et 0.57
# Scores df_pinot_noir_fr_burg : 0.52 et 0.50


######
# SVR
######

#modelSVR = SVR().fit(X_train_scaled, y_train)
#modelSVR.score(X_train_scaled, y_train)
#modelSVR.score(X_test_scaled, y_test)

# Trop long...


######
# LINEAR SVR
######

modelLSVR = LinearSVR().fit(X_train_scaled, y_train)
modelLSVR.score(X_train_scaled, y_train)
modelLSVR.score(X_test_scaled, y_test)

# Scores : 0.41 et 0.45
# Scores df_pinot_noir_fr_burg : 0.39 et 0.42


######
# RANDOM FOREST REGRESSOR
######

modelRFR = RandomForestRegressor().fit(X_train_scaled, y_train)
modelRFR.score(X_train_scaled, y_train)
modelRFR.score(X_test_scaled, y_test)

# Scores : 0.73 et 0.65
# Scores df_pinot_noir_fr_burg : 0.72 et 0.48


######
# DECISION TREE REGRESSOR
######

modelDTR = DecisionTreeRegressor().fit(X_train_scaled, y_train)
modelDTR.score(X_train_scaled, y_train)
modelDTR.score(X_test_scaled, y_test)

# Scores : 0.74 et 0.64
# Scores df_pinot_noir_fr_burg : 0.73 et 0.43


######
# GRADIENT BOOSTING REGRESSOR
######

modelGBR = GradientBoostingRegressor().fit(X_train_scaled, y_train)
modelGBR.score(X_train_scaled, y_train)
modelGBR.score(X_test_scaled, y_test)

# Scores : 0.71 et 0.68
# Scores df_pinot_noir_fr_burg : 0.70 et 0.58


######
# LASSO
######

modelL = linear_model.Lasso().fit(X_train_scaled, y_train)
modelL.score(X_train_scaled, y_train)
modelL.score(X_test_scaled, y_test)

# Scores : 0.53 et 0.56
# Scores df_pinot_noir_fr_burg : 0.51 et 0.50


######
# EXTRA TREES REGRESSOR
######

modelETR = ExtraTreesRegressor().fit(X_train_scaled, y_train)
modelETR.score(X_train_scaled, y_train)
modelETR.score(X_test_scaled, y_test)

# Scores : 0.74 et 0.65
# Scores df_pinot_noir_fr_burg : 0.73 et 0.46

"""
Modèle avec les meilleurs résultats : DecisionTreeRegressor, 0.74 et 0.64.
Modèle avec le mois d'overfitting : GradientBoostingRegressor, 0.71 et 0.68.
"""


##########################################################################################
# ESSAIS D'OPTIMISATION DU MODELE
##########################################################################################

######
# GRIDSEARCH
######

dico = {#'max_depth' : range(1, 5),
        'min_samples_leaf' : range(1, 5),
        'min_samples_split' : range(1, 5)}
grid = GridSearchCV(DecisionTreeRegressor(), dico)
grid.fit(X_train_scaled,y_train)

print("best score:",grid.best_score_)
print("best parameters:",grid.best_params_)


######
# APPLICATION DECISION TREE REGRESSOR
######

modelDTR = DecisionTreeRegressor(min_samples_leaf=4, min_samples_split=4).fit(X_train_scaled, y_train)
modelDTR.score(X_train_scaled, y_train)
modelDTR.score(X_test_scaled, y_test)


