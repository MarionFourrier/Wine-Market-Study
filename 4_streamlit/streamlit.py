# streamlit run 4_streamlit/streamlit.py

##########################################################################################
# IMPORT LIBRAIRIES
##########################################################################################

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import streamlit as st



##########################################################################################
# IMPORT DONNEES
##########################################################################################

df_price_chardo = pd.read_csv('data/suggested_price_chardo.csv')
df_price_pinot = pd.read_csv('data/suggested_price_pinot.csv')
df_chardo_fr = pd.read_csv('data/clean_wine_chardo_fr.csv')
df_pinot_fr = pd.read_csv('data/clean_wine_pinot_fr.csv')
df_domaine = pd.read_csv('data/clean_domaine.csv')

# Suppression colonne d'index
df_pinot_fr.drop(columns='Unnamed: 0', inplace=True)
df_chardo_fr.drop(columns='Unnamed: 0', inplace=True)
df_price_pinot.drop(columns='Unnamed: 0', inplace=True)
df_price_chardo.drop(columns='Unnamed: 0', inplace=True)

# Concaténation df estimation des prix
df_prices = pd.concat([df_price_pinot, df_price_chardo])
df_prices.sort_values('suggested_price', inplace=True, ascending=False)

# Amélioration tableau
df_prices.reset_index(inplace=True, drop=True)
df_prices.rename(columns = {'variety':'CÉPAGE',
                            'title':'APPELLATION',
                            'points':'NOTE',
                            'millesime':'MILLÉSIME',
                            'suggested_price':'SUGGESTION DE PRIX (EN $)'},
                 inplace = True)


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

# Suppression colonnes
df_domaine.drop(columns='Unnamed: 0',
                inplace=True)

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


##########################################################################################
# MACHINE LEARNING RCH PLUS PROCHES VOISINS PRIX
##########################################################################################

######
# PREPARATION DES DONNEES DU DATASET PINOT NOIR
######

# Déclaration des données d'entrées et de sorties
X_prix = df_pinot_fr[['price']]

######
# INSTANCIATION ET ENTRAINEMENT MODELE
######

# Model
modelNN_prix = NearestNeighbors(n_neighbors=20).fit(X_prix)



##########################################################################################
# STREAMLIT
##########################################################################################

######
# ONGLET
######

st.set_page_config(
    page_title="Estimation des Prix du Vin",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
    )

######
# INTRO
######

st.markdown("<h1 style='text-align: center; color: #97678A'>ESTIMATION DES PRIX DES VINS DU DOMAINE DES CROIX</h1>",
                unsafe_allow_html=True)
st.write(' ')

# Image
from PIL import Image
image = Image.open('4_streamlit/image_bouteille_domaine.jpg')
st.image(image, caption='Domaine des Croix - Bourgogne Pinot Noir | Keeper Collection', width = 700)


######
# 1) DF SUGGESTION PRIX
######

st.markdown("<h4 style='color: #F28E2B'>Voici une estimation des prix de vente sur le marché américain :</h4>",
                unsafe_allow_html=True)
st.write(' ')
st.write(df_prices, width=700)
st.write(' ')
st.write(' ')


######
# 2) VISION SUR LE MARCHÉ
######

### ENTREE

st.markdown("<h4 style='color: #F28E2B'>Comparaison avec des vins similaires vendus aux USA :</h4>",
                unsafe_allow_html=True)

input_vin = st.selectbox('Choisissez un Pinot Noir :', df_domaine_ml['title'].unique())

### SORTIE

# Si aucun vin selectionné, pass
if input_vin == '':
    pass
else:
    # Récupération de l'index de la sélection
    index_vin_selection = df_domaine_ml.loc[df_domaine_ml['title'] == input_vin].index[0]
    # Récupération des X scalées à entrer dans le modèle
    entree_ml = X_domaine_pinot_scaled[index_vin_selection]
    entree_ml = entree_ml.reshape(1, -1)
    # Récupération des plus proches voisins
    neighbors_distance, neighbors_index = modelNN.kneighbors(entree_ml)
    neighbors_index = neighbors_index[0][:]
    nearest_neighbors_vin = df_pinot_fr.iloc[neighbors_index, :]
    # Adaptation du df
    nearest_neighbors_vin = nearest_neighbors_vin[['title', 'points', 'millesime', 'price', 'variety']]
    # Affichage du df avec les vins les plus proches
    st.write(nearest_neighbors_vin, width=700)
    # Calcul des moyennes et médianes de ces vins
    median_prix = nearest_neighbors_vin['price'].median()
    moyenne_prix = nearest_neighbors_vin['price'].mean()
    # Affichage de ces mesures
    st.write('Le prix moyen de ces vins est de :', median_prix)
    st.write('Le prix médian de ces vins est de :', moyenne_prix)

st.write('')
st.write('')


######
# 3) VISION SUR LES VINS VENDUS AUX MÊMES PRIX
######

### ENTREE

st.markdown("<h4 style='color: #F28E2B'>Comparaison avec des vins vendus aux mêmes prix aux USA :</h4>",
                unsafe_allow_html=True)

# Préparation df selection
df_prices2 = df_prices.copy()
#df_prices2 = df_prices2.loc[df_prices2['CÉPAGE'] == 'Pinot Noir']
df_prices2['choix_prix'] = df_prices2['APPELLATION'] + ' - ' + df_prices2['NOTE'].astype(str) + 'pts - ' + df_prices2['MILLÉSIME'].astype(str) + ' - ' + df_prices2['SUGGESTION DE PRIX (EN $)'].astype(str) + '$'

# Selection entree
input_vin = st.selectbox('Choisissez un Pinot Noir :', df_prices2['choix_prix'].unique())
#input_vin = 'Grèves  (Corton) - 95pts - 2016 - 231$'
input_vin_prix = df_prices2.loc[df_prices2['choix_prix'] == input_vin]
#input_vin_prix = df_prices2.loc[0, :]
#input_vin_prix = df_prices2
#input_vin_prix = df_prices2.loc[df_prices2['choix_prix'] == input_vin_prix].index[0]

# Recuperation du prix de l'entrée utilisateur
#prix_input_vin = input_vin_prix['SUGGESTION DE PRIX (EN $)']
prix_input_vin = input_vin_prix.iloc[0, 4]


### SORTIE

# Récupération des plus proches voisins
#prix_input_vin.to_numpy()
prix_input_vin = prix_input_vin.reshape(-1, 1)
neighbors_distance_prix, neighbors_index_prix = modelNN_prix.kneighbors(prix_input_vin)
neighbors_index_prix = neighbors_index_prix[0][:]
nearest_neighbors_vin_prix = df_pinot_fr.iloc[neighbors_index_prix, :]

# Adaptation du df
nearest_neighbors_vin_prix = nearest_neighbors_vin_prix[['title', 'points', 'millesime', 'price', 'variety', 'province']]

# Affichage du df avec les vins les plus proches
st.write(nearest_neighbors_vin_prix, width=700)
