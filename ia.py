from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
import pandas as pd
from vectorizacion import X_word2vec

# Cargamos los datos procesados
df = pd.read_csv("Sentimiento_procesado.csv")

#cargamos el modelo
model = Word2Vec.load("word2vec_model.model")

# Suponiendo que tienes una columna 'sentimiento' en tu DataFrame que contiene las etiquetas
X = X_word2vec
y = df['sentimiento']  # Etiquetas de sentimiento

# Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el clasificador Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Realizar las predicciones
y_pred = modelo.predict(X_test)

# Evaluar el rendimiento del modelo
print("Accuracy del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
