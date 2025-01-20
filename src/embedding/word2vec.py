from gensim.models import Word2Vec

NUM_HILOS = 4

def crear_modelo(fuente, dim_embeddings=100, sg=1, epocas=50):
    """ Crear y entrenar el modelo word2vec con los datos fuente.
    :param fuente: Documentos preprocesados. Puede ser una lista de lista de tokens o el nombre del archivo que contiene las documentos, en este caso cada linea representa un documento y las palabras están separadas por espacios.
    :type fuente: list[list[str]] |  str
    :param dim_embeddings: Tamaño de los embeddings
    :type dim_embeddings: int
    :param sg: Usar
    :param epocas: Número de epocas de entrenamiento
    :return: Devuelve el modelo Word2Vec entrenado
    :rtype: Word2Vec
    """
    model = None
    if isinstance(fuente, str): # Cargar los datos desde un archivo
        print(f"[word2vec]: Cargando datos de {fuente}")
        model = Word2Vec(
            corpus_file=fuente,
            vector_size=dim_embeddings,  # dimensiones de los embeddings
            epochs=epocas,
            workers=NUM_HILOS,
            sg=sg # Predecir las palabras de contextoa  partir de una palabra central
        )
    else:
        sentences = fuente
        print(f"[word2vec]: Lista recibida")
        model = Word2Vec(
            sentences=sentences,
            vector_size=dim_embeddings,  # dimensiones de los embeddings
            epochs=epocas,
            workers=NUM_HILOS,
            sg=sg # Predecir las palabras de contextoa  partir de una palabra central
        )

    model.save("word2vec_model.model")
    print("[word2vec]: Modelo entrenado y guardado")
    return None