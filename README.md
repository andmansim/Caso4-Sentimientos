https://github.com/andmansim/Caso4-Sentimientos.git

objetivos: El objetivo del caso de uso es analizar las opiniones y comentarios 
de los usuarios sobre su experiencia con ChatGPT mediante te cnicas de 
procesamiento de lenguaje natural (NLP) y ana lisis de sentimiento.
Clasificiar en satisfaccion (good), insatisfaccion (bad), neutro (neutral)

Pasos:
Limpiar y leer los datos 
1. Procesamiento del lenguaje (EN)
    - tokenización, dividir en palabras las frases
    - quitar todo lo que no sea caracteres alfabéticos. 
    - pasar todo a minúsculas
    - stopwords, coger la raiz o base de las palabras parecidas
    - Lematización, quitar aquellas particulas que no atribuyen info, determinantes, pronombres, etc. 
2. Vectorización
    - TF-IDF, calcular la importancia de las palabras, según su frecuencia en el texto. 
    - Word Embeddings, crear los vectores en fucnión de sus relaciones semánticas
3. Aplicar IA
    Modelos clásicos
    - Naive bayes
    - SVM 
    Modelos avanzamos
    - Transformes (GPT, BERT)
    - RNN
4. Aprendizaje supervisado
    Hacer un modelo de aprendizaje suprvisado como siempre, divides en train y test. Lo lanzas, ajustas cositas, optimizas y validas. 
    Posibles modelos: RandomForestClassifier, XGBoostClassifier y puede que un árbol simple de deciión. 

