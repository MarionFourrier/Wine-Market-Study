##########################################################################################
# IMPORT LIBRAIRIES
##########################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
from wordcloud import WordCloud


##########################################################################################
# IMPORT DONNEES
##########################################################################################

df_wine = pd.read_csv('data/clean_wine.csv')
df_domaine = pd.read_csv('data/clean_domaine.csv')


##########################################################################################
# PRÉPARATION NLP
##########################################################################################

nltk.download('popular')

######
# CREATION D'UN SET AVEC LES STOPS WORDS ET LA PONCTUATION
######

set_punctuation = set(string.punctuation)
set_punctuation.add('’')
set_punctuation.add('\'')
set_punctuation.add('-')
set_punctuation.add('\'s')
set_stop_words = set(nltk.corpus.stopwords.words("english"))
set_stop_words_punct = set_punctuation.union(set_stop_words)


######
# FONCTION
######

def func_clean(commentaire):
  """
      Fonction qui nettoie les stop words et la ponctuation
      Param : commentaire = le texte à nettoyer
      Return : filtered_words = le texte nettoyé
  """
  # Tokenisation et minusculisation
  tokens_words = nltk.word_tokenize(commentaire.lower())
  # On filtre les stop words et la ponctuation (dans une liste)
  filtered_words_list = [word for word in tokens_words if not word in set_stop_words_punct]
  # On transforme la liste en string
  filtered_words = str = ' '.join(filtered_words_list)
  # Return
  return(filtered_words)

# Test fonction
commentaire = 'Hello, how are you? Fine, thank you. \'s '
func_clean(commentaire)


##########################################################################################
# WORDCLOUD SUR TOUTES LES DESCRIPTIONS DES VINS
##########################################################################################

######
# PREPARATION DATA
######

# Application fonction au df
df_wine['nlp'] = df_wine['description'].apply(lambda x : func_clean(x))

# Creation texte avec toutes les descriptions
text_nlp_wine = ' '.join(df_wine['nlp'])


######
# WORDCLOUD
######

wordcloud = WordCloud(width=960,
                      height=480,
                      max_font_size=200,
                      min_font_size=10,
                      background_color="white",
                      colormap="copper")

# Génération du wordcloud
dico_nlp_wine = nltk.FreqDist(nltk.word_tokenize(text_nlp_wine))
wordcloud.generate_from_frequencies(dico_nlp_wine)

# Affichage
plt.figure()
plt.title('POUR TOUS LES VINS EN GÉNÉRAL')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

# Enregistrement image
wordcloud.to_file('wordcloud_wine.png')





##########################################################################################
# NLP SUR DESCRIPTIONS PINOT NOIR
##########################################################################################

######
# PREPARATION DATA
######

# Creation df pinot noir
df_pinot_noir = df_wine.loc[df_wine['variety'] == 'Pinot Noir']

# Application fonction au df
df_pinot_noir['nlp'] = df_pinot_noir['description'].apply(lambda x : func_clean(x))

# Creation texte avec toutes les descriptions
text_nlp_pinot_noir = ' '.join(df_pinot_noir['nlp'])


######
# WORDCLOUD
######

wordcloud2 = WordCloud(width=960,
                      height=480,
                      max_font_size=200,
                      min_font_size=10,
                      background_color="white",
                      colormap="copper")

# Génération du wordcloud
dico_nlp_pinot_noir = nltk.FreqDist(nltk.word_tokenize(text_nlp_pinot_noir))
wordcloud2.generate_from_frequencies(dico_nlp_pinot_noir)

# Affichage
plt.figure()
plt.title('POUR LES PINOTS NOIRS')
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

# Enregistrement image
wordcloud2.to_file('wordcloud_pinot_noir.png')



##########################################################################################
# NLP SUR DESCRIPTIONS PINOT NOIR DE BOURGOGNE
##########################################################################################

######
# PREPARATION DATA
######

# Creation df pinot noir
df_pinot_noir_burgundy = df_pinot_noir.loc[df_pinot_noir['province'] == 'Burgundy']

# Application fonction au df
df_pinot_noir_burgundy['nlp'] = df_pinot_noir_burgundy['description'].apply(lambda x : func_clean(x))

# Creation texte avec toutes les descriptions
text_nlp_pinot_noir_burgundy = ' '.join(df_pinot_noir_burgundy['nlp'])


######
# WORDCLOUD
######

wordcloud3 = WordCloud(width=960,
                      height=480,
                      max_font_size=200,
                      min_font_size=10,
                      background_color="white",
                      colormap="copper")

# Génération du wordcloud
dico_nlp_pinot_noir_burgundy = nltk.FreqDist(nltk.word_tokenize(text_nlp_pinot_noir_burgundy))
wordcloud3.generate_from_frequencies(dico_nlp_pinot_noir_burgundy)

# Affichage
plt.figure()
plt.title('POUR LES PINOTS NOIRS DE BOURGOGNE')
plt.imshow(wordcloud3, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

# Enregistrement image
wordcloud3.to_file('wordcloud_pinot_noir_burgundy.png')




##########################################################################################
# NLP SUR DESCRIPTIONS VINS DU DOMAINE DES CROIX
##########################################################################################

######
# PREPARATION DATA
######

# Application fonction au df
df_domaine['nlp'] = df_domaine['description'].apply(lambda x : func_clean(x))

# Creation texte avec toutes les descriptions
text_nlp_domaine = ' '.join(df_domaine['nlp'])


######
# WORDCLOUD
######

wordcloud4 = WordCloud(width=960,
                      height=480,
                      max_font_size=200,
                      min_font_size=10,
                      background_color="white",
                      colormap="copper")

# Génération du wordcloud
dico_nlp_domaine = nltk.FreqDist(nltk.word_tokenize(text_nlp_domaine))
wordcloud4.generate_from_frequencies(dico_nlp_domaine)

# Affichage
plt.figure()
plt.title('POUR LES VINS DU DOMAINE DES CROIX')
plt.imshow(wordcloud4, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

# Enregistrement image
wordcloud3.to_file('wordcloud_domaine_des_croix.png')


"""
Axes d'amélioration :
    - Enlever mots relatifs au vin (tels que 'wine'...)
"""