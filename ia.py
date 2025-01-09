from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
import pandas as pd
import numpy as np


print('cargamos los modelos y el csv')

# Cargamos los datos procesados
df = pd.read_csv("Sentimiento_procesado.csv")

#cargamos el modelo
model = Word2Vec.load("word2vec_model.model")

# Cargar la matriz X_word2vec desde el archivo .npy
X_word2vec = np.load("X_word2vec.npy")

print('dividimos train y test')
# Suponiendo que tienes una columna 'sentimiento' en tu DataFrame que contiene las etiquetas
X = X_word2vec
y = df['sentimiento']  # Etiquetas de sentimiento

# Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Creamos el modelo')
# Crear el clasificador Random Forest
modelo_randomforest = RandomForestClassifier(n_estimators=200, max_depth= 15, min_samples_split=10, min_samples_leaf= 5, bootstrap=True, random_state=42, n_jobs=-1)
#n_estimators: 100 a 300, max_depth: 10 a 20, min_samples_split: 2 a 10, min_samples_leaf: 1 a 5, boos    

# Entrenar el modelo
print('Entrenando el modelo...')
modelo_randomforest.fit(X_train, y_train)

# Realizar las predicciones
print('Prediciendo...')
y_pred = modelo_randomforest.predict(X_test)



# Evaluar el rendimiento del modelo
print("Accuracy del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(modelo_randomforest, 'modelo_randomforest.pkl')

#añadimos la clasificacion al csv
df['sentimiento_predicho'] = modelo_randomforest.predict(X)


#-------------------Gráficas-------------------
import matplotlib.pyplot as plt

# Contamos las clases predichas
import seaborn as sns

sns.countplot(y_pred)
plt.title('Distribución de Sentimientos Predichos')
plt.xlabel('Sentimiento')
plt.ylabel('Frecuencia')
plt.savefig('Distribución de Sentimientos Predichos.png')
plt.show()


from collections import Counter

# Unir todas las palabras procesadas en una sola lista
todas_las_palabras = [palabra for texto in df['procesado'] for palabra in texto]

# Contar la frecuencia de cada palabra
frecuencia_palabras = Counter(todas_las_palabras)

# Mostrar las 10 palabras más comunes
print(frecuencia_palabras.most_common(10))


# Graficar las 10 palabras más comunes
palabras, frecuencia = zip(*frecuencia_palabras.most_common(10))

plt.barh(palabras, frecuencia)
plt.xlabel('Frecuencia')
plt.ylabel('Palabra')
plt.title('Palabras Más Comunes en los Comentarios')
plt.savefig('Palabras Más Comunes en los Comentarios.png')
plt.show()