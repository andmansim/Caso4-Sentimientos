#procesaminto del lenguaje natural

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def procesar_texto(texto):
    # texto = texto.lower()
    # texto = re.sub(r"[\W\d_]+", " ", texto)
    # palabras = word_tokenize(texto)
    # stop_words = set(stopwords.words("english"))
    # palabras = [palabra for palabra in palabras if palabra not in stop_words]
    # stemmer = PorterStemmer()
    # palabras = [stemmer.stem(palabra) for palabra in palabras]
    # lemmatizer = WordNetLemmatizer()
    # palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras]
    # return palabras
   
   
    # Convertir el texto a minúsculas
    texto = texto.lower()
    # Eliminar caracteres no alfabéticos y dígitos
    texto = re.sub(r"[\W\d_]+", " ", texto)
    # Tokenización del texto
    palabras = word_tokenize(texto)
    # Cargar stopwords en inglés
    stop_words = set(stopwords.words("english"))
    
    # Filtrar palabras no stopwords
    palabras = [palabra for palabra in palabras if palabra not in stop_words]
    
    # Lematización (solo lematización, no stemming)
    lemmatizer = WordNetLemmatizer()
    palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras]
    
    return palabras

# Cargamos los datos del csv
import pandas as pd
df = pd.read_csv("Sentimiento.csv", encoding="latin-1", sep=",")

#Los datos que queremos que separe está en la columna 'texto' y nos lo añade en la columna 'procesado'
df['procesado'] = df['texto'].apply(procesar_texto)
print(df.head())
#Exportamos el nuevo csv con los datos procesados
df.to_csv("Sentimiento_procesado.csv", index=False)

