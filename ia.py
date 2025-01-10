from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

print('cargamos los modelos y el csv')

# Cargamos los datos procesados
df = pd.read_csv("Sentimiento_procesado.csv")
df['procesado'] = df['procesado'].apply(ast.literal_eval)

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
# # modelo_randomforest = RandomForestClassifier(n_estimators=300, max_depth= 20, min_samples_split=10, min_samples_leaf= 5, bootstrap=True, random_state=42, n_jobs=-1)
modelo_randomforest = RandomForestClassifier( bootstrap= True, max_depth= 15, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 200, random_state=42, n_jobs=-1)
# # #n_estimators: 100 a 300, max_depth: 10 a 20, min_samples_split: 2 a 10, min_samples_leaf: 1 a 5, bootstrap = True o False    

# Codificar las etiquetas si es necesario (si son 'bad', 'good', 'neutral')


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


xgb = XGBClassifier(
    n_estimators=450,            # Número de árboles
    max_depth=10,                # Profundidad máxima de los árboles
    min_child_weight=10,          # Controla el peso mínimo de las hojas (equivalente a min_samples_leaf)
    subsample=0.8,               # Usa todo el conjunto de entrenamiento (equivalente a bootstrap=True)
    learning_rate=0.1,           # Tasa de aprendizaje, por defecto en XGBoost
    colsample_bytree = 0.6,
    random_state=42,             # Semilla aleatoria
    n_jobs=-1                    # Usar todos los núcleos disponibles
)
modelo = xgb

# Entrenar el modelo
print('Entrenando el modelo...')
modelo.fit(X_train, y_train_encoded)


# Realizar las predicciones
print('Prediciendo...')
y_pred = modelo.predict(X_test)


# Evaluar el rendimiento del modelo
print("Accuracy del modelo:", accuracy_score(y_test_encoded, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test_encoded, y_pred))



import joblib
joblib.dump(modelo, 'modelo_xgb.pkl')

#añadimos la clasificacion al csv
df['sentimiento_predicho'] = modelo.predict(X)


#-------------------Gráficas-------------------
# Decodificar las etiquetas predichas
y_pred_decoded = label_encoder.inverse_transform(y_pred)

import matplotlib.pyplot as plt

# Contamos las clases predichas
import seaborn as sns

sns.countplot(x=y_pred_decoded, order=['bad', 'good','neutral']) 
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






#----------------------------Calcular los mejores hiperparametros para un trozo de los datos--------------------------------

# Crear un subconjunto del 30% de los datos para el tuning
# X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42)




# # Define el modelo base
# modelo_randomforest = RandomForestClassifier(random_state=42)

# # Define los hiperparámetros que quieres probar
# param_grid = {
#     'n_estimators': [100, 200],      # Número de árboles en el bosque
#     'max_depth': [10, 15],           # Profundidad máxima de los árboles
#     'min_samples_split': [2, 5],     # Mínimo de muestras para dividir un nodo
#     'min_samples_leaf': [1, 2],       # Mínimo de muestras en una hoja
#     'bootstrap': [True]           # Si se utiliza muestreo con reemplazo
# }

# grid_search = GridSearchCV(
#     estimator=modelo_randomforest,
#     param_grid=param_grid,
#     cv=3,                # Solo 3 particiones en lugar de 5
#     n_jobs=-1,
#     scoring='accuracy', 
#     verbose=3  # Nivel de detalle (3 es bastante informativo)
# )

# grid_search.fit(X_sample, y_sample)

# print("Mejores parámetros encontrados:", grid_search.best_params_)
# print("Mejor puntuación:", grid_search.best_score_)

# # Usa el mejor modelo
# mejor_modelo = grid_search.best_estimator_

# mejor_modelo.fit(X_train, y_train)

# y_pred = mejor_modelo.predict(X_test)



#------------------------Probamos con un xgboost--------------------------------
# Crear un subconjunto del 30% de los datos para el tuning
# X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42)

# from xgboost import XGBClassifier
# import numpy as np
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.preprocessing import LabelEncoder

# # Codificar las etiquetas de sentimiento
# label_encoder = LabelEncoder()
# y_sample_encoded = label_encoder.fit_transform(y_sample)

# # Definir el espacio de hiperparámetros a explorar (con distribuciones)
# param_dist = {
#     'n_estimators': [400, 450, 500],
#     'max_depth': [ 7, 10, 13],
#     'learning_rate': [0.05, 0.1, 0.15],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [ 0.4, 0.6, 0.8],
#     'min_child_weight': [ 10, 13, 15],
# }

# # Crear el clasificador XGBoost
# xgb = XGBClassifier(random_state=42, n_jobs=-1)

# # Configurar RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=param_dist,
#     n_iter=50,  # Número de combinaciones aleatorias a probar
#     scoring='accuracy',
#     cv=3,
#     n_jobs=-1,
#     random_state=42,
#     verbose=2
# )

# # Ajustar el modelo con RandomizedSearchCV usando X_sample y y_sample_encoded
# print("Entrenando con RandomizedSearchCV...")
# random_search.fit(X_sample, y_sample_encoded)

# # Obtener los mejores parámetros
# print("\nMejores parámetros encontrados:", random_search.best_params_)
# print("Mejor puntuación obtenida:", random_search.best_score_)




#-----------------------------------------------------------------------------------------------------