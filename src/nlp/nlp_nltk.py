from nltk import SnowballStemmer
from nltk.stem import SnowballStemmer

stemmers_names = {
    "en": "english",
    "es": "spanish"
}

def stemming_pipe(documentos: list[list[str]], lang: str="es"):
    """Obtener la forma raiz de los tokens de cada documento

    :param documentos: Lista de documentos tokenizados
    :type documentos: list[list[str]]
    :param lang: idioma de la lista de documentos
    :type lang: str
    :return: Lista de raices de los tokens de cada documento
    :rtype: list[list[str]]
    """
    if lang not in stemmers_names.keys():
        print(f"Idioma no soportado: {lang}")
        return None

    stemmer = SnowballStemmer(stemmers_names[lang])
    return list(map(
        lambda tokens_documento: [stemmer.stem(token) for token in tokens_documento],
        documentos
    ))