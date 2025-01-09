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

    # stemmer = PorterStemmer()
    # palabras = [stemmer.stem(palabra) for palabra in palabras]
   
    # Convertir el texto a minúsculas
    texto = texto.lower()
    
    # Eliminar URLs completas
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)
    
    # Eliminar caracteres no alfabéticos y dígitos
    texto = re.sub(r"[\W\d_]+", " ", texto)
    
    # Eliminar caracteres no alfabéticos y dígitos
    texto = re.sub(r"[^a-z\s]+", "", texto)  # Solo letras y espacios
    
    # Tokenización del texto
    palabras = word_tokenize(texto)
    # Cargar stopwords en inglés
    stop_words = set(stopwords.words("english"))
    
    # Filtrar palabras no stopwords
    palabras = [palabra for palabra in palabras if palabra not in stop_words]
    
    # Lematización (solo lematización, no stemming)
    lemmatizer = WordNetLemmatizer()
    palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras]
    
    # Excluir tokens no deseados (letras sueltas, etc.)
    palabras = [palabra for palabra in palabras if len(palabra) > 1]
    
    return palabras

# Cargamos los datos del csv
import pandas as pd
df = pd.read_csv("Sentimiento.csv", encoding="latin-1", sep=",")

# Asegurarnos de que no haya valores nulos en la columna 'texto'
df['texto'] = df['texto'].fillna("")

# Eliminar URLs completas y hashtags antes de procesar
df["texto"] = df["texto"].str.replace(r"https?://\S+|www\.\S+", "", regex=True)  # Eliminar URLs
df["texto"] = df["texto"].str.replace(r"(#\w+|\\\w+)", "", regex=True)          # Eliminar hashtags y backslashes

# Procesar texto y añadirlo a una nueva columna
df['procesado'] = df['texto'].apply(procesar_texto)

# Revisar los datos procesados
print(df.head())

# Exportamos el nuevo CSV con los datos procesados
df.to_csv("Sentimiento_procesado.csv", index=False, encoding="utf-8")