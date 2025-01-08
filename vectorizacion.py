#Fase de vectorización de los datos


# Comenzamos con IF-IDF

# Importamos la librería
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Cargamos los datos procesados
df = pd.read_csv("Sentimiento_procesado.csv")

# Creamos el objeto
tfidf = TfidfVectorizer()

# unir las palabras prcesadas en un solo string
X = tfidf.fit_transform(df['procesado'].apply(lambda x: ' '.join(x)))

tfidf = TfidfVectorizer(stop_words=None)  # No eliminar stopwords adicionales


#------------------------------Visualización de los datos ---------------------------------------------

# Mostrar las primeras 10 características (columnas) y sus valores correspondientes
print(X.toarray()[:, :10])  # Solo las primeras 10 columnas

# Ver el vocabulario (palabras que se están usando)
print("Vocabulario:")
print(tfidf.get_feature_names_out()[:10])  # Primeras 10 palabras del vocabulario

# Ver los índices de las palabras en el vocabulario
print("Vocabulario índice:")
print(tfidf.vocabulary_)

# Ver los valores del IDF (Inversa frecuencia de documento) para las palabras
print("IDF de las palabras:")
print(tfidf.idf_[:10])  # Primeros 10 valores IDF

# Paràmetros del modelo
print("Parámetros del modelo TF-IDF:")
print(tfidf.get_params())