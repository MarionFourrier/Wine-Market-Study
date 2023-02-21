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
CONCLUSION MACHINE LEARNING 2 :
- Meilleurs scores avec df : 
    country = France
    variety = Pinot Noir
    price = < 348
- Essai1 : garder 10 pays les plus présents mais que sur Pinot Noir
- Essai2 : avec plus de données : garder 10 pays les plus présents, 10 cépages des plus présents ?
"""

######
# SELECTION SEPAGE PINOT NOIR
######

df_pinot_noir = df_wine.loc[df_wine['variety'] == 'Pinot Noir']

######
# SELECTION 10 PAYS LES PLUS PRÉSENTS
######

df_pinot_noir['country'].value_counts()
df_pinot_noir_country = df_pinot_noir.loc[(df_pinot_noir['country'] == 'France') |
                                          (df_pinot_noir['country'] == 'US') |
                                          (df_pinot_noir['country'] == 'New Zealand') |
                                          (df_pinot_noir['country'] == 'Chile') |
                                          (df_pinot_noir['country'] == 'Australia') |
                                          (df_pinot_noir['country'] == 'Argentina') |
                                          (df_pinot_noir['country'] == 'Austria') |
                                          (df_pinot_noir['country'] == 'Germany') |
                                          (df_pinot_noir['country'] == 'Canada') |
                                          (df_pinot_noir['country'] == 'South Africa')
                                         ]

######
# SELECTION DES PRIX
######

"""
# D'après le dashboard : on a de gros outliers pour les prix au dessus de 378, on les supprime.
"""

# Recherche de données à supprimer
df_prix_to_drop = df_pinot_noir_country.loc[df_pinot_noir_country['price'] > 378]
# Index de ces données
index_prix_to_drop = df_prix_to_drop.index
list_prix_to_drop = index_prix_to_drop.to_list()
# Suppression des données où millesime inférieur à 2000 ou supérieur à 2023 (241 lignes)
df_pinot_noir_country.drop(index=list_prix_to_drop,
                      inplace=True)
# Verification : OK, suppression de 6 vins
df_pinot_noir_country.describe()


######
# MATRICE CORRELATION
######

# Suppression colonne 1
df_pinot_noir_country.drop(columns='Unnamed: 0',
                           inplace=True)

# Encodage des provinces
df_pinot_noir_country = pd.concat([df_pinot_noir_country, df_pinot_noir_country['country'].str.get_dummies()],
                                  axis = 1)

# Heatmap
sns.heatmap(df_pinot_noir_country.corr(),
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
X = df_pinot_noir_country.select_dtypes('number').drop(columns='price')
y = df_pinot_noir_country['price']

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
# Scores df_pinot_noir_country : 0.50 et 0.51


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
# Scores df_pinot_noir_country : 0.43 et 0.45


######
# RANDOM FOREST REGRESSOR
######

modelRFR = RandomForestRegressor().fit(X_train_scaled, y_train)
modelRFR.score(X_train_scaled, y_train)
modelRFR.score(X_test_scaled, y_test)

# Scores : 0.73 et 0.65
# Scores df_pinot_noir_fr_burg : 0.72 et 0.48
# Scores df_pinot_noir_country : 0.77 et 0.62


######
# DECISION TREE REGRESSOR
######

modelDTR = DecisionTreeRegressor().fit(X_train_scaled, y_train)
modelDTR.score(X_train_scaled, y_train)
modelDTR.score(X_test_scaled, y_test)

# Scores : 0.74 et 0.64
# Scores df_pinot_noir_fr_burg : 0.73 et 0.43
# Scores df_pinot_noir_country : 0.78 et 0.59


######
# GRADIENT BOOSTING REGRESSOR
######

modelGBR = GradientBoostingRegressor().fit(X_train_scaled, y_train)
modelGBR.score(X_train_scaled, y_train)
modelGBR.score(X_test_scaled, y_test)

# Scores : 0.71 et 0.68
# Scores df_pinot_noir_fr_burg : 0.70 et 0.58
# Scores df_pinot_noir_country : 0.74 et 0.63


######
# LASSO
######

modelL = linear_model.Lasso().fit(X_train_scaled, y_train)
modelL.score(X_train_scaled, y_train)
modelL.score(X_test_scaled, y_test)

# Scores : 0.53 et 0.56
# Scores df_pinot_noir_fr_burg : 0.51 et 0.50
# Scores df_pinot_noir_country : 0.49 et 0.51


######
# EXTRA TREES REGRESSOR
######

modelETR = ExtraTreesRegressor().fit(X_train_scaled, y_train)
modelETR.score(X_train_scaled, y_train)
modelETR.score(X_test_scaled, y_test)

# Scores : 0.74 et 0.65
# Scores df_pinot_noir_fr_burg : 0.73 et 0.46
# Scores df_pinot_noir_country : 0.78 et 0.62

"""
CCL ML 2 : 
Modèle avec les meilleurs résultats : DecisionTreeRegressor, 0.74 et 0.64.
Modèle avec le mois d'overfitting : GradientBoostingRegressor, 0.71 et 0.68.

CCL ML 3 :
Modèle avec les meilleurs résultats : ExtraTreeRegressor, 0.78 et 0.62.
Modèle avec le mois d'overfitting : pareil, bcp dans tous les cas...
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

