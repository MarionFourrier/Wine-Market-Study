##########################################################################################
# IMPORT LIBRAIRIES
##########################################################################################

import pandas as pd

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
# PREPARATION DES DONNEES
##########################################################################################

# On vérifie qu'on a pas de nan dans les colonnes qu'on veut utiliser
df_wine.isna().sum()

# Déclaration des données d'entrées et de sorties
X = df_wine[['points', 'millesime']]
y = df_wine['price']

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

# Scores pas incroyables (0.36 et 0.37), peu d'overfitting


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

# Scores pas incroyables (0.30, 0.31), peu d'overfitting


######
# RANDOM FOREST REGRESSOR
######

modelRFR = RandomForestRegressor().fit(X_train_scaled, y_train)
modelRFR.score(X_train_scaled, y_train)
modelRFR.score(X_test_scaled, y_test)

# Scores un peu mieux (0.494, 0.475), peu d'overfitting


######
# DECISION TREE REGRESSOR
######

modelDTR = DecisionTreeRegressor().fit(X_train_scaled, y_train)
modelDTR.score(X_train_scaled, y_train)
modelDTR.score(X_test_scaled, y_test)

# Presque mêmes scores que RandomForestRegressor (0.495, 0.471), et un peu plus rapide, peu d'overfitting


######
# GRADIENT BOOSTING REGRESSOR
######

modelGBR = GradientBoostingRegressor().fit(X_train_scaled, y_train)
modelGBR.score(X_train_scaled, y_train)
modelGBR.score(X_test_scaled, y_test)

# Presque mêmes scores que RandomForestRegressor (0.47, 0.49), peu d'overfitting


######
# LASSO
######

modelL = linear_model.Lasso().fit(X_train_scaled, y_train)
modelL.score(X_train_scaled, y_train)
modelL.score(X_test_scaled, y_test)

# Scores pas incroyables (0.36, 0.37), peu d'overfitting


######
# EXTRA TREES REGRESSOR
######

modelETR = ExtraTreesRegressor().fit(X_train_scaled, y_train)
modelETR.score(X_train_scaled, y_train)
modelETR.score(X_test_scaled, y_test)

# Scores toujours bien, comme pour les autres modèles de decision tree (0.49, 0.47), peu d'overfitting

"""
Modèle avec les meilleurs résultats et le plus rapide : DecisionTreeRegressor.
Essai d'amélioration du modèle.
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

modelDTR = DecisionTreeRegressor(min_samples_leaf=4, min_samples_split=1).fit(X_train_scaled, y_train)
modelDTR.score(X_train_scaled, y_train)
modelDTR.score(X_test_scaled, y_test)

"""
CONCLUSION :
- Les scores sont moisn bons après le gridsearch...
- Les scores sont de toute manière assez mauvais. 
- Perspectives : 
    1) Retravailler le dataset et faire une matrice de corrélation. D'après l'explo, la région et le cépage doivent jouer.
    2) Enlever du dataset les données dont on ne va pas se servir : cépages différents, pays différent, régions différentes, vins trop chers...
"""