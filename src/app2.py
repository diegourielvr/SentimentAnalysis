import numpy as np
import pandas as pd
# from keras.utils import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import os

from gensim.models import Word2Vec

from src.dataset.logistica.regresionLogMulticlase import \
    regresionLogisticaMulticlase
from src.embedding.word2vec import crear_modelo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.estadisticas.estadisticas import mostrar_grafica, \
    mostrar_matriz_confusion
from src.nlp.nlp_nltk import stemming_pipe
from src.nlp.nlp_spacy import preprocesamiento, tokenizar

etiquetas_dict = {
    "NEG": 0,
    "NEU": 1,
    "POS": 2
}

def limpieza(df):

    # filtrar por columnas de interes
    df = df[["fecha_publicacion", "termino", "vistas", "text", "idioma", "polaridad"]]
    # Filtrar comentarios en español
    df = df[df["idioma"] == "es"].reset_index(drop=True)
    # Convertir las etiquetas en enteros
    df["polaridad"] = df["polaridad"].map(etiquetas_dict)
    # convertir a minusculas
    df["text"] = df["text"].str.lower()

    df["text"] = (
        df["text"]
        # Eliminar espacios extras
        .str.replace(r"\s+", " ", regex=True)
        # Eliminar espacios al inicio o final del texto
        .str.strip()
    )

    # Coreccion ortografica

    # Eliminar los que sean solo números
    df = df[~df["text"].str.match(r"^\d+$", na=False)]
    # Eliminar comentarios nulos NaN
    df = df[df["text"].notnull() & df["text"].str.strip().ne("")]
    # Contar el numero de palabras de los comentarios
    df['num_palabras'] = df['text'].apply(lambda x: len(x.split()))
    # Eliminar textos menores a una minima cantidad de palabra (opcional)
    df = df[df["num_palabras"] > 2].reset_index(drop=True)

    # NLP
    documentos = preprocesamiento(df["text"].tolist(), "es")
    #documentos = stemming_pipe(documentos, "es")
    df["text"] = [" ".join(documento) for documento in documentos]

    # Eliminar comentarios nulos NaN
    df = df[df["text"].notnull() & df["text"].str.strip().ne("")]
    # Contar el numero de palabras de los comentarios
    df['num_palabras'] = df['text'].apply(lambda x: len(x.split()))


    # --- tener el mismo numero de datos para cada etiqueta
    min_clase = df["polaridad"].value_counts().min()

    # Submuestrear cada clase
    df = (df.groupby("polaridad").apply(lambda x: x.sample(n=min_clase, random_state=42)).reset_index(drop=True))
    print(df["polaridad"].value_counts())

    return df

def tf_idf_nb(df):
    # Devidir los datos
    X = df["text"]
    Y = df["polaridad"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    tdidf_vectorizer = TfidfVectorizer(
        max_features=5000, # limite del vocabulario
        ngram_range=(1,2) # considerar unigramas y bigramas
    )

    X_train_tfidf = tdidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tdidf_vectorizer.fit_transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    print("Matriz de Confusión:")
    matriz_confusion = confusion_matrix(y_test, y_pred)
    mostrar_matriz_confusion(matriz_confusion)

def bow_svm(df):
    # Devidir los datos
    X = df["text"]
    Y = df["polaridad"]

    # label_encoder = LabelEncoder()
    # Y = label_encoder.fit_transform(Y)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Usar bow
    vectorizer = CountVectorizer()
    train_X_bow = vectorizer.fit_transform(train_X)
    test_X_bow = vectorizer.transform(test_X)

    svm_model = SVC(kernel="linear", random_state=42)
    svm_model.fit(train_X_bow, train_Y)

    pred_Y = svm_model.predict(test_X_bow)
    print("Accuracy: ", accuracy_score(test_Y, pred_Y))
    print("Reporte de clasificación")
    print(classification_report(test_Y, pred_Y))

    print("Matriz de Confusión:")
    matriz_confusion = confusion_matrix(test_Y, pred_Y)
    mostrar_matriz_confusion(matriz_confusion)

def word2vec_lstm(df):
    dim_embeddings = 300
    size_time_step = 150 # tamaño de los timsetps
    print("Entrenando modelo de embeddings...")
    documentos = tokenizar(df["text"].tolist())
    crear_modelo(documentos,
                 dim_embeddings=dim_embeddings,
                 sg=1,
                 epocas=200)
    # Cargar modelo
    wv = Word2Vec.load("word2vec_model.model")
    # Generar tokens de cada documento
    documentos = tokenizar(df["text"].tolist())
    # Obtener los embeddings de cada token
    embeddings = []
    for documento in documentos:
        embedding = []
        for token in documento:
            if token in wv.wv:
                embedding.append(wv.wv[token])
            else:
                # print("embeddings zeros")
                embedding.append(np.zeros(dim_embeddings))
        embeddings.append(embedding)

    # Debemos tener el mismo numero de secuencias o embeddings de cada documento
    embeddings_padded = pad_sequences(
        embeddings,
        # maxlen=size_time_step,# numero de secuencias
        dtype="float32",
        padding="post", # "pre" | "post"
        truncating="post"
    )

    embeddings_padded = np.array(embeddings_padded)
    # print(embeddings_padded)
    print(embeddings_padded.shape)

    X = embeddings_padded
    # Y = np.array(df["polaridad"]) # etiquetas numericas
    Y = df["polaridad"] # etiquetas numericas


    train_X, test_X, train_Y, test_Y = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    # --- Arquitectura de la NN
    model = Sequential()
    model.add(LSTM(
        64,
        input_shape=(train_X.shape[1], train_X.shape[2]), # numero de secuencias y tamaño de embeddings
    ))
    model.add(Dropout(0.5)) # Descativar el 50% de las neuronas durante el entrenamiento

    model.add(Dense(3, activation='softmax'))

    model.compile(
        loss="sparse_categorical_crossentropy", # Entropia cruzada con etiquetas numericas
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.summary()

    model.fit(
        train_X,
        train_Y,
        batch_size=64,
        epochs=10
    )

    # Guardar el modelo
    model.save("modelo_lstm_sentimiento.h5")

    # Realizar predicciones
    pred_Y_prob = model.predict(test_X)
    pred_Y = np.argmax(pred_Y_prob,axis=1) # obtener el indice de la probabilidad mas grande
    print("Reporte de clasificación:")
    print(classification_report(test_Y, pred_Y))

    # Evaluar el modelo con los datos de prueba
    loss, accuracy = model.evaluate(test_X, test_Y, verbose=1)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # 2. Calcula la matriz de confusión
    matriz_confusion = confusion_matrix(test_Y, pred_Y)
    mostrar_matriz_confusion(matriz_confusion)

def test_embeddings():
    model_wv = Word2Vec.load("word2vec_model.model")
    vocabulario = list(model_wv.wv.index_to_key)
    palabra = vocabulario[:10]
    print(palabra)
    print(len(vocabulario))
    print(model_wv.wv.most_similar("triste"))

def preprocesamiento_datos(path_dataset, path_data_clean):
    dataframes = []
    for nombre_archivo in os.listdir(path_dataset):
        if nombre_archivo.endswith(".csv"):
            print(nombre_archivo)
            df = pd.read_csv(os.path.join(path_dataset, nombre_archivo))
            dataframes.append(df)
    df_combinado = pd.concat(dataframes, ignore_index=True)

    df_combinado["fecha_publicacion"] = pd.to_datetime(df_combinado["fecha_publicacion"], errors="coerce")

    print(df_combinado.info())
    print(df_combinado)

    # Mostrar grafica con estadisticas
    df_limpio = limpieza(df_combinado)
    mostrar_grafica(df_limpio)

    print(df_limpio.info())
    print(df_limpio)

    df_limpio.to_csv(os.path.join(path_data_clean,"datos_limpios.csv"), index=False)
    return df_limpio

if "__main__" == __name__:
    path_data = os.path.join(os.getcwd(), "dataset", "sentimientos")
    path_data_cleaned = os.path.join(os.getcwd(), "dataset", "limpios")

    df = pd.read_csv(os.path.join(path_data_cleaned,"datos_limpios.csv"))

    # ----- Limpiar y NLP
    # preprocesamiento_datos(path_data, path_data_cleaned)

    # ----- entrenar modelo bow + svm
    # bow_svm(df)

    # ----- entrenar modelo tf-idf + nb
    # tf_idf_nb(df)

    # ---- Regresion lineal multiclae OneVsAll
    # regresionLogisticaMulticlase(df, 5000, 0.5)

    # ----- Entrenar embeddings con word2vec
    word2vec_lstm(df)


    # ---- Test embeddings
    # test_embeddings()

