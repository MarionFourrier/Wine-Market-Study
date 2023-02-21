##########################################################################################
# IMPORT LIBRAIRIES
##########################################################################################

import pandas as pd
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
#pip uninstall scikit-learn -y
#pip install scikit-learn==0.23.1


##########################################################################################
# IMPORT DONNEES
##########################################################################################

df_wine = pd.read_csv('data/clean_wine.csv')
df_domaine = pd.read_csv('data/clean_domaine.csv')


##########################################################################################
# DECLARATION DES DONNÉES D'ENTREES ET DE SORTIES
##########################################################################################

# On vérifie qu'on a pas de nan dans les colonnes qu'on veut utiliser
df_wine.isna().sum()

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

reg = lazypredict.Supervised.LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
models,predictions = reg.fit(X_train, X_test, y_train, y_test)