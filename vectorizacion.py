#Fase de vectorización de los datos


# Comenzamos con IF-IDF

# Importamos la librería
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Cargamos los datos procesados
df = pd.read_csv("Sentimiento_procesado.csv")


# Verifica las stopwords cargadas
print("Stopwords cargadas:", stopwords.words("english"))

# Mostrar algunas filas procesadas antes de aplicar la eliminación de stopwords
df['procesado_sin_stopwords'] = df['texto'].apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word not in stopwords.words("english")]))

print(df['procesado_sin_stopwords'].head())  # Verifica que el texto procesado tiene contenido útil


# Creamos el objeto
tfidf = TfidfVectorizer()

# Filtrar filas que tienen texto procesado no vacío
df = df[df['procesado'].apply(lambda x: len(x) > 0)]


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




# Con word2Vec

#------------------------entrenamos el modelo Word2Vec--------------------------------------------
import gensim
from gensim.models import Word2Vec

# Entrenar el modelo Word2Vec
model = Word2Vec(sentences=df['procesado'], vector_size=100, window=5, min_count=1, workers=4)

# Guardar el modelo
model.save("word2vec_model.model")


#--------------Obtener el vector promedio de cada texto--------------------------------------------

import numpy as np

# Función para obtener el vector promedio de un texto
def obtener_vector_promedio(texto, model):
    # El texto es una lista de palabras, obtenemos el vector de cada una
    vectores = [model.wv[palabra] for palabra in texto if palabra in model.wv]
    
    # Si el texto tiene al menos una palabra en el vocabulario
    if len(vectores) > 0:
        # Promediamos los vectores
        return np.mean(vectores, axis=0)
    else:
        # Si no hay palabras en el vocabulario, retornamos un vector de ceros
        return np.zeros(model.vector_size)

# Obtener el vector promedio para cada fila
X_word2vec = np.array([obtener_vector_promedio(texto, model) for texto in df['procesado']])

# Ver el tamaño de la matriz de vectores resultante (filas = documentos, columnas = dimensiones del vector)
print("Matriz de vectores Word2Vec de tamaño:", X_word2vec.shape)

# Ver los primeros vectores para observar cómo se ven
print("Primeros 3 vectores promedios de Word2Vec:")
print(X_word2vec[:3])  # Primeros 3 textos transformados en vectores