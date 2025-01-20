import spacy
model_names = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm"
}
def tokenizar(documentos: list[str], lang="es"):
    if lang not in model_names.keys():
        print(f"Idioma no soportado: {lang}")
        return None
    nlp = spacy.load(model_names[lang])
    print(f"Modelo cargado: {model_names[lang]}")

    documentos_preprocesados = []
    for doc in nlp.pipe(documentos): # procesamiento de documentos
        doc_preprocesado = []
        for token in doc:
            doc_preprocesado.append(token.text)
        documentos_preprocesados.append(doc_preprocesado)
    print(f"Total de documentos preprocesados: {len(documentos_preprocesados)}")
    return documentos_preprocesados


def preprocesamiento(documentos: list[str], lang="es") -> list[list[str]]:
    """Procesamiento de lenguaje natural del texto.
    Tokenizar, eliminar signos de puntuacion, stopwords y lematizar.

    :param documentos: Lista de textos
    :type documentos: list[str]
    :param lang: idioma de los textos
    :type lang: str
    :return: Lista de lemas de cada documento
    :rtype: list[list[str]]
    """
    if lang not in model_names.keys():
        print(f"Idioma no soportado: {lang}")
        return None
    nlp = spacy.load(model_names[lang])
    print(f"Modelo cargado: {model_names[lang]}")

    documentos_preprocesados = []
    for doc in nlp.pipe(documentos): # procesamiento de documentos
        doc_preprocesado = []
        for token in doc:
            # filtrar lemas de palabras que no sean stopwords o signos de puntuacion
            if not token.is_stop and not token.is_punct:
                doc_preprocesado.append(token.lemma_)
        documentos_preprocesados.append(doc_preprocesado)
    print(f"Total de documentos preprocesados: {len(documentos_preprocesados)}")
    return documentos_preprocesados

def preprocesamiento_mwt(documentos: list[str], lang="es"):
    """Procesamiento de lenguaje natural del texto.
    Tokenización multipalabra con base en entidades, eliminar signos de puntuacion, stopwords y lematizar.

    :param documentos: Lista de textos
    :type documentos: list[str]
    :param lang: idioma de los textos
    :type lang: str
    :return: Lista de lemas de cada documento
    :rtype: list[list[str]]
    """
    if lang not in model_names.keys():
        print(f"Idioma no soportado: {lang}")
        return None
    nlp = spacy.load(model_names[lang])
    print(f"Modelo cargado: {model_names[lang]}")

    documentos_preprocesados = []
    for doc in nlp.pipe(documentos): # procesamiento de documentos
        # Retokenizar para agrupar entidades en un solo token
        with doc.retokenize() as retokenizer:  # Unir tokens
            # Unir entidades reconocidas en un solo token
            for ent in doc.ents:
                retokenizer.merge(ent)

        doc_preprocesado = []
        for token in doc:
            # filtrar lemas de palabras que no sean stopwords o signos de puntuacion
            if not token.is_stop and not token.is_punct:
                doc_preprocesado.append(token.lemma_)
        documentos_preprocesados.append(doc_preprocesado)
    print(f"Total de documentos preprocesados: {len(documentos_preprocesados)}")
    return documentos_preprocesados
