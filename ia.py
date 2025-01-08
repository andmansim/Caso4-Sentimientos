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
modelo_randomforest = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
modelo_randomforest.fit(X_train, y_train)

# Realizar las predicciones
y_pred = modelo_randomforest.predict(X_test)

if __name__ == '__main__':

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
