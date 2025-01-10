#-------------------Clasificación de nuevos textos-------------------
from proces_lenguaje import procesar_texto
import numpy as np
from vectorizacion import obtener_vector_promedio
import joblib
from gensim.models import Word2Vec

#cargamos el modelo guardado

modelo_randomforest = joblib.load('modelo_randomforest.pkl')
model = Word2Vec.load("word2vec_model.model")



# Probando con un nuevo comentario
comentarios = [
    "I love this product, it is amazing!",  # Comentario positivo
    "This is the worst thing I have ever bought.",  # Comentario negativo
    "Potatoes and vegetables. Are vegetables. I don't know what to say. #Free_potatoes.",  # Comentario neutral
    "Research preview of our newest model: ChatGPT're trying something new with this preview: Free and immediately available for everyone (no waitlist!)"
]

# Procesar los comentarios y obtener los vectores promedio
comentarios_procesados = [procesar_texto(comentario) for comentario in comentarios]
X_nuevos_comentarios = np.array([obtener_vector_promedio(texto, model) for texto in comentarios_procesados])

# Hacer predicciones con el modelo RandomForest
predicciones = modelo_randomforest.predict(X_nuevos_comentarios)

# Mostrar los resultados
for comentario, prediccion in zip(comentarios, predicciones):
    print(f"Comentario: '{comentario}'")
    print(f"Sentimiento Predicho: {prediccion}")
    print("-------------")

# Si quieres saber el porcentaje de las clases en el conjunto de pruebas
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(predicciones)
plt.title('Distribución de Sentimientos Predichos')
plt.xlabel('Sentimiento')
plt.ylabel('Frecuencia')
plt.show()