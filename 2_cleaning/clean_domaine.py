##########################################################################################
# IMPORT LIBRAIRIES
##########################################################################################

import pandas as pd
import re


##########################################################################################
# IMPORT DONNEES
##########################################################################################

df_domaine = pd.read_csv('https://raw.githubusercontent.com/murpi/wilddata/master/domaine_des_croix.csv')


##########################################################################################
# NETTOYAGE
##########################################################################################

"""
A nettoyer :
- Supprimer colonnes inutiles
- Récupérer le millésime
- Récupérer la désignation si vide ? (équivaut au nom de la région ?)
"""

######
# Suppression colonnes inutiles
######

df_domaine.drop(columns=['price', 'region_2', 'taster_name', 'taster_twitter_handle'],
                inplace=True)

######
# Récupération millesime
######

# Essai sur une valeur

title_test = df_domaine.loc[0, 'title']
motif_millesime = '\d\d\d\d'
re.search(motif_millesime, title_test).group(0)

# Application sur le df

df_domaine["millesime"] = df_domaine["title"].apply(lambda x: re.search(motif_millesime, x).group(0))


##########################################################################################
# EXPORT DONNEE
##########################################################################################

df_domaine.to_csv('data/clean_domaine.csv')