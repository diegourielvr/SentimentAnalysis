import os
from symspellpy import SymSpell

dictionary_paths = {
    "en": os.path.join(os.getcwd(), "ortografia", "diccionarios", "en-80k.txt"),
    "es": os.path.join(os.getcwd(), "ortografia", "diccionarios", "es-100l.txt")
}

def corregir(texto, lang="es"):
    if lang not in dictionary_paths.keys():
        print(f"Idioma no soportado: {lang}")
        return None
    spell = SymSpell(
        max_dictionary_edit_distance=2, # Distancia de búsqueda
        prefix_length=7 # Prefijos de palabras
    )

    load = spell.load_dictionary(
        dictionary_paths[lang],
        term_index=0, # posicion donde se encuentran los terminos
        count_index=1, # posicion donde se encuentran las frecuencias
        encoding="utf-8"
    )
    if not load:
        print("[sympellpy]: No ha sido posible cargar el diccionario")
        return None

    sugerencias = spell.lookup_compound(
        texto,
        max_edit_distance=2,
        ignore_non_words=True, # ignorar caracteres como números
    )
    return sugerencias[0].term

def corregir_pipe(documentos: list[str], lang="es"):
    """Corregir ortografia de una lista de textos

    :param documentos:  Lista de textos a corregir. Puede recibir df[column_name].tolist()
    :type documentos: list[str]
    :param lang: Idioma del texto a corregir
    :type lang: str
    :return: Devuelve la lista con los textos corregidos
    :rtype: list[str]
    """
    if lang not in dictionary_paths.keys():
        print(f"Idioma no soportado: {lang}")
        return None
    spell = SymSpell(
        max_dictionary_edit_distance=2, # Distancia de búsqueda
        prefix_length=7 # Prefijos de palabras
    )

    load = spell.load_dictionary(
        dictionary_paths[lang],
        term_index=0, # posicion donde se encuentran los terminos
        count_index=1, # posicion donde se encuentran las frecuencias
        encoding="utf-8"
    )
    if not load:
        print("[sympellpy]: No ha sido posiblecargar el diccionario")
        print(f"[symspellpy]: Ruta: {dictionary_paths[lang]}")
        return None

    return list(map(
        lambda documento: spell.lookup_compound(documento, max_edit_distance=2)[0].term,
        documentos
    ))
