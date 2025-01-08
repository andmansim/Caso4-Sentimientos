#Fase de vectorización de los datos

# Con word2Vec

#------------------------entrenamos el modelo Word2Vec--------------------------------------------
import gensim
from gensim.models import Word2Vec
import pandas as pd


# Cargamos los datos procesados
df = pd.read_csv("Sentimiento_procesado.csv")

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

if __name__ == '__main__':
    # Ver el tamaño de la matriz de vectores resultante (filas = documentos, columnas = dimensiones del vector)
    print("Matriz de vectores Word2Vec de tamaño:", X_word2vec.shape)

    # Ver los primeros vectores para observar cómo se ven
    print("Primeros 3 vectores promedios de Word2Vec:")
    print(X_word2vec[:3])  # Primeros 3 textos transformados en vectores
