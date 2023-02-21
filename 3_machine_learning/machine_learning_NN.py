##########################################################################################
# IMPORT LIBRAIRIES
##########################################################################################

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd


##########################################################################################
# IMPORT DONNEES
##########################################################################################

df_pinot_fr = pd.read_csv('data/clean_wine_pinot_fr.csv')
df_domaine = pd.read_csv('data/clean_domaine.csv')

# Suppression colonne d'index
df_pinot_fr.drop(columns='Unnamed: 0', inplace=True)
df_domaine.drop(columns='Unnamed: 0', inplace=True)


##########################################################################################
# MACHINE LEARNING RCH PLUS PROCHES VOISINS
##########################################################################################

######
# PREPARATION DES DONNEES DU DATASET PINOT NOIR
######

# Déclaration des données d'entrées et de sorties
X_pinot = df_pinot_fr.select_dtypes('number').drop(columns='price')

# Standardisation
scaler = StandardScaler().fit(X_pinot)
X_pinot_scaled = scaler.transform(X_pinot)


######
# PREPARATION DES DONNEES DU DATASET DU DOMAINE POUR PINOT NOIR
######

# Suppression du chardonnay
df_domaine.drop(index=12,
                inplace=True)

# Copie du df
df_domaine_ml = df_domaine.copy()

# Ajout colonnes nécessaires
list_columns_to_add = df_pinot_fr.columns[11:]
for column in list_columns_to_add:
    df_domaine_ml[str(column)] = 0
df_domaine_ml['Burgundy'] = 1

# Données d'entrée
X_domaine_pinot = df_domaine_ml.select_dtypes('number')

# Standardisation
X_domaine_pinot_scaled = scaler.transform(X_domaine_pinot)


######
# INSTANCIATION ET ENTRAINEMENT MODELE
######

# Model
modelNN = NearestNeighbors(n_neighbors=10).fit(X_pinot_scaled)


######
# ENTREE UTILISATEUR
######


# Choix vin

#input_vin = st.selectbox('', df_domaine_ml['title'].unique())
input_vin = df_domaine_ml.loc[0, 'title']
# Récupération de l'index de la sélection
index_vin_selection = df_domaine_ml.loc[df_domaine_ml['title'] == input_vin].index[0]
# Récupération des X scalées à entrer dans le modèle
entree_ml = X_domaine_pinot_scaled[index_vin_selection]
entree_ml = entree_ml.reshape(1, -1)
# Récupération des plus proches voisins
neighbors_distance, neighbors_index = modelNN.kneighbors(entree_ml)
neighbors_index = neighbors_index[0][:]
nearest_neighbors_vin = df_pinot_fr.iloc[neighbors_index, :]

# Prix médians et moyens
median_prix = nearest_neighbors_vin['price'].median()
moyenne_prix = nearest_neighbors_vin['price'].mean()





