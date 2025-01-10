#-------------------Clasificación de nuevos textos-------------------
from proces_lenguaje import procesar_texto
import numpy as np
from vectorizacion import obtener_vector_promedio
import joblib
from gensim.models import Word2Vec

#cargamos el modelo guardado

modelo_xgb = joblib.load('modelo_xgb.pkl')
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

# Hacer predicciones con el modelo xgb
predicciones = modelo_xgb.predict(X_nuevos_comentarios)
probabilidades = modelo_xgb.predict_proba(X_nuevos_comentarios)

## Decodificar las etiquetas predichas
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# label_encoder.fit(['bad', 'neutral', 'good'])  # Ajustar al conjunto de clases del modelo

# Mostrar los resultados
for comentario, prediccion, probabilidad in zip(comentarios, predicciones, probabilidades):
    # prediccion_clase = label_encoder.inverse_transform([prediccion])[0]  # Decodificar la clase
    intensidad = max(probabilidad)  # Obtener la probabilidad máxima como intensidad
    print(f"Comentario: '{comentario}'")
    print(f"Sentimiento Predicho: {prediccion}")
    print(f"Intensidad del Sentimiento: {intensidad:.2f}")
    print("-------------")

# Si quieres visualizar la distribución de las predicciones
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=predicciones )
plt.title('Distribución de Sentimientos Predichos')
plt.xlabel('Sentimiento')
plt.ylabel('Frecuencia')
plt.show()