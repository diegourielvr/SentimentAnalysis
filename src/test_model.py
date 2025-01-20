import os
from embedding import word2vec

path = os.path.join(os.getcwd(), "dataset","datos_prueba.txt")
print(path)
word2vec.crear_modelo(path)
word2vec.crear_modelo([["a", "v"], ["c", "d"]])
