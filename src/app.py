import os
import re

import numpy as np
# import tensorflow_datasets as tfds
import pandas as pd
from gensim.models import Word2Vec

import matplotlib.pyplot as plt

from ortografia import correcion_orografica
import unicodedata
# Bibliotecas para corregir fechas y caracteres especiales
from datetime import datetime, timedelta
import pytz # zona horaria
from langdetect import detect
import seaborn as sns

from src.embedding.word2vec import crear_modelo
from src.nlp.nlp_nltk import stemming_pipe
from src.nlp.nlp_spacy import preprocesamiento, tokenizar

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout

from sklearn.metrics import confusion_matrix, classification_report


cwd = os.getcwd()
path_datos_recolectados = os.path.join(cwd, "dataset", "tiktok_sentimiento.csv")
path_datos_preprocesados = os.path.join(cwd, "dataset", "tiktok_preprocesado.csv")
path_datos_para_embeddings = os.path.join(cwd, "dataset", "datos_para_embeddings.txt")
dim_embeddings = 100
mapa_sentimientos = {
    'NEG': 0,
    'NEU': 1,
    'POS': 2
}

def detectar_idioma(texto):
    try: return detect(texto)
    except: return None

def extra_info(dataframe, columna):
    dataframe["idioma_original"] = dataframe[columna].apply(detectar_idioma)
    return dataframe

def analisis_exploratorio(dataframe):
    print(dataframe.info())
    print(dataframe.head())

def reemplazar_nbsp(texto):
  """Reemplaza el símbolo &nbsp; o '\xa0' por un espacio en blanco
  """
  #return string.replace("\xa0", " ")
  return unicodedata.normalize("NFKD", str(texto))

def corregir_fecha_tiktok(fecha, fecha_referencia):
  """Tranformar fechas a un formato común yyyy-mm-dd
  """

  zona_horaria = pytz.timezone('America/Mexico_City')
  #fecha_actual = datetime.now(tz=pytz.utc).astimezone(zona_horaria)
  fecha_actual = pd.to_datetime(fecha_referencia)
  if fecha.startswith('Hace'):
    partes = fecha.split(" ") # <Hace> <1> <s | min | h | dia(s) | semana(s)>
    cantidad = partes[1]
    unidad = partes[2] # <s | min | h | dias(s) | semana(s)>

    if unidad == "s":
      return (fecha_actual - timedelta(seconds=int(cantidad))).date()
    elif unidad == "min":
      return (fecha_actual - timedelta(minutes=int(cantidad))).date()
    elif unidad == "h":
      return (fecha_actual - timedelta(hours=int(cantidad))).date()
    elif unidad == reemplazar_nbsp("día(s)"): # La tílde es un caracter especial. Significa que esta tilde í y esta otra í son diferentes (internamente)
      return (fecha_actual - timedelta(days=int(cantidad))).date()
    elif unidad == "semana(s)":
      return (fecha_actual - timedelta(weeks=int(cantidad))).date()

  partes = fecha.split("-")
  if len(partes) == 2: # 10-19 -> mm-dd
    mes, dia = map(int, partes)
    return datetime(fecha_actual.year, mes, dia, tzinfo=zona_horaria).date()
  elif len(partes) == 3: # 2023-10-19 -> yyyy-mm-dd
    partes[2] = partes[2].split(" ")[0]
    anio, mes, dia = map(int, partes)
    return datetime(anio, mes, dia, tzinfo=zona_horaria).date()

  return pd.NA

def normalizar_likes(likes):
    # Si el valor está en formato 'k' o 'K', lo convertimos
    if isinstance(likes, str):
        # Convertir a minúsculas para evitar problemas con 'K' y 'k'
        likes = likes.lower()
        # Verificar si el valor contiene 'k' al final
        if 'k' in likes:
            # Eliminar 'k' y convertir el número a float, luego multiplicarlo por 1000
            return float(likes.replace('k', '').replace(',', '').strip()) * 1000
    # Si no es un valor con 'k', lo devolvemos como está
    return int(likes)

def limpieza(dataframe, columna="comentario", lang="es"):
    """

    :param dataframe:
    :param columna:
    :param lang:
    :return: Dataframe con los datos limpiios y preprocesados
    """
    minimo_num_palabras = 1 # Eliminar los comentarios que tengan este numerod de apalabras

    # Filtrar por comentarios en español
    dataframe = dataframe[dataframe["idioma_original"] == lang]
    dataframe = dataframe.reset_index(drop=True)

    # Convertir la columna a texto
    dataframe[columna] = dataframe[columna].astype(str)

    # Conovertir la columna a minusculas
    dataframe[columna] = dataframe[columna].str.lower()

    # Corregir símbolos especiales como &nbsp
    dataframe[columna] = dataframe[columna].apply(reemplazar_nbsp)

    dataframe[columna] = (
        dataframe[columna]
        # Reemplazar saltos de línea invisibles (caracteres de control como \n, \r, tabulaciones, etc.)
        .str.replace(r"[\r\n\t]+", " ", regex=True)  # Reemplaza cualquier salto de línea o tabulación con un espacio
        # Remover urls
        .str.replace(r"(http\S+|www\S+|https\S+)", "", regex=True)
        # Remover hashtags
        .str.replace(r"#\w+", "", regex=True)
        # Remover menciones
        .str.replace(r"@\w+", "", regex=True)
        # Remover código HTML
        .str.replace(r"<.*?>", "", regex=True)
    # Remover emojis
        .str.replace(
            r"[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]",
            "", regex=True)
        # Eliminar comillas estándar y tipográficas
        .str.replace(r"[\"'“”‘’]", "", regex=True)
        # Eliminar otros signos de puntuacion
        .str.replace(r"[^\w\s]", "", regex=True)
        # Eliminar espacios extras
        .str.replace(r"\s+", " ", regex=True)
        # Eliminar espacios al inicio o final del texto
        .str.strip()
    )

    # Corrección ortográfica del texto (tambien lo convierte a minúsculas)
    # dataframe[columna] = correcion_orografica.corregir_pipe(dataframe[columna].tolist(), lang)

    # Filtrar comentarios: eliminar los que sean solo números
    dataframe = dataframe[~dataframe[columna].str.match(r"^\d+$", na=False)]  # Filtrar filas con solo números

    # Eliminar comentarios nulos NaN
    dataframe = dataframe[dataframe[columna].notnull() & dataframe[columna].str.strip().ne("")]

    dataframe = dataframe[dataframe[columna].notnull() & (dataframe['comentario'] != '')]

    # Eliminar textos menores a una minima cantidad de palabra (opcional)
    dataframe = dataframe[dataframe[columna].str.split().str.len() > minimo_num_palabras]


    return dataframe

def limpieza_especial(dataframe):
    dataframe["fecha_publicacion"] = dataframe["fecha_publicacion"].apply(reemplazar_nbsp)

    # Reconstruir fechas de tiktok
    dataframe["fecha_publicacion"] = dataframe.apply(
        lambda row: corregir_fecha_tiktok(row["fecha_publicacion"], row["fecha_recuperacion"]),
        axis=1
    )
    dataframe["fecha_recuperacion"] = pd.to_datetime(dataframe["fecha_recuperacion"], errors='coerce')

    # Reconstruir likes
    dataframe["num_likes"] = dataframe["num_likes"].apply(normalizar_likes)

    # Convertir etiquetas de sentimiento a formato numérico
    dataframe["sentimiento"] = dataframe["sentimiento"].map(mapa_sentimientos)

    return dataframe


def limpieza_preprocesamiento(lemmas=True, stemming=True):
    """
    :param lemmas: Aplicar lematización
    :param stemming: Aplicar stemming. Solo si lemmas=True
    :return:
    """
    # 1. Cargar los datos
    print("Cargando Datos...")
    df = pd.read_csv(path_datos_recolectados, encoding="utf-8")
    # 1.1 Agregar idioma original del texto
    print("Agregando idioma original de cada comentario...")
    df = extra_info(df, "comentario")

    # 2. EDA: Análisis exploratorio de datos
    print("Análisis de los datos cargados...")
    analisis_exploratorio(df)

    # 3. Limpieza de los datos con base en el EDA
    # 3.1 Seleccionar caracterpisticas relevantes
    df = df[["fecha_recuperacion", "comentario", "fecha_publicacion", "num_likes", "sentimiento", "idioma_original"]]
    print("Realizando limpieza...")
    df = limpieza(df, columna="comentario", lang="es")
    df = limpieza_especial(df)

    # 3.2 Seleccionar caracterpisticas relevantes
    df = df[["comentario", "fecha_publicacion", "num_likes", "sentimiento", "idioma_original"]]

    # 5. --- Procesamiento del lenguaje natural, NLP
    # Puede generar algunos simbolos o numeros, por lo que necesario volver alimpiar
    if lemmas:
        documentos = preprocesamiento(df["comentario"].tolist())
        if stemming:
            documentos = stemming_pipe(documentos, "es")
        df["comentario"] = [" ".join(documento) for documento in documentos]

    df = limpieza(df, columna="comentario", lang="es")

    # 6. EDA: Análisis exploratorio de datos limpios y preprocesados
    analisis_exploratorio(df)

    # mostrar comentarios con menos de 5 palabras
    #print(df[df["comentario"].str.split().str.len() < 2])
    df.to_csv(path_datos_preprocesados, index=False, encoding="utf-8")
    return None

def crear_embeddings():
    df = pd.read_csv(path_datos_preprocesados, encoding="utf-8")
    df["comentario"].to_csv(path_datos_para_embeddings, index=False, encoding="utf-8", header=None)
    print("Entrenando modelo...")
    crear_modelo(path_datos_para_embeddings, dim_embeddings=dim_embeddings, sg=1, epocas=100)

def obtener_embedding(documento: list[str], modelo):
    embedding = []
    for token in documento:
        if token in modelo.wv:
            embedding.append(modelo.wv[token])
        else:
            embedding.append(np.zeros(dim_embeddings))
    return embedding

def padding_embeddings(embeddings, max_len=100):
    # Aplanar la lista de embeddings para cada comentario
    embeddings_padded = pad_sequences(embeddings, maxlen=max_len,
                                      dtype='float32', padding='post',
                                      truncating='post')
    return embeddings_padded

def cargar_modelo():
    wv = Word2Vec.load("word2vec_model.model")
    df = pd.read_csv(path_datos_preprocesados, encoding="utf-8")
    df["tokens"] = tokenizar(df["comentario"].tolist())

    # Obtener embeddings de cada tokens
    df["embeddings"] = df["tokens"].apply(lambda documento: obtener_embedding(documento, wv))

    # Rellenar los embeddings (mismo numero de secuencias para cada documento)
    # df["embeddings_padded"] = df["embeddings"].apply(lambda x: padding_embeddings(x))

    embeddings_padded = pad_sequences(df["embeddings"], maxlen=dim_embeddings, dtype='float32', padding='post', truncating='post')

    df["embeddings_padded"] = list(embeddings_padded)

    # Validar la longitud de las secuencias
    longitudes = df['embeddings_padded'].apply(len)
    print("longitudes:")
    print(longitudes.value_counts())  # Deben ser iguales a `max_len`

    X = np.array(df['embeddings_padded'].tolist())  # Las secuencias de embeddings
    y = np.array(df['sentimiento'])  # Las etiquetas numéricas

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir el modelo LSTM
    model = Sequential()

    # Capa LSTM
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),
                   return_sequences=False))

    # Capa de dropout para regularización
    model.add(Dropout(0.5))

    # Capa densa de salida con 3 clases (si son 3 clases de sentimiento)
    model.add(Dense(3, activation='softmax'))

    # Compilar el modelo
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # Resumen del modelo
    model.summary()

    # Entrenar el modelo
    model.fit(X_train, y_train, batch_size=64, epochs=20,
              validation_data=(X_test, y_test))

    # Guardar el modelo
    model.save("modelo_lstm_sentimiento.h5")

    # 1. Realiza predicciones
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob,
                       axis=1)  # Convertir probabilidades a etiquetas

    # Generar el reporte de clasificación
    print(classification_report(y_test, y_pred,
                                target_names=["NEG", "NEU", "POS"]))

    # Evaluar el modelo con los datos de prueba
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # 2. Calcula la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    # Visualizar con Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["NEG", "NEU", "POS"],
                yticklabels=["NEG", "NEU", "POS"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def probar_embeddings():
    wv = Word2Vec.load("word2vec_model.model")
    # vocab = wv.wv.vocab
    # print(f"Total de palabras aprendidas por w2vec: {len(vocab)}")
    print(wv.wv.most_similar('spotify'))


if "__main__" == __name__:
    # Cargar los datos
    # Estadísticas de los datos
    # Limpiar los datos para su preprocesamiento
    # Estadísticas de los datos limpios
    # Preprocesamiento con NLP de los datos

    #limpieza_preprocesamiento(lemmas=True, stemming=False)

    # Entrenar modelo embeddings
    #crear_embeddings()

    # Crear y enrtenar Red neuronal
    #cargar_modelo()
    # Accuracy, Recall, F1 score
    # Matriz de confusion
    probar_embeddings()

