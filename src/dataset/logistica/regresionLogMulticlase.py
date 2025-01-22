import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.estadisticas.estadisticas import mostrar_matriz_confusion


def pesos_iniciales(n, k):
    return np.zeros((n, k))

def sigmoid(w, X):
    """Funcion sigmoide
        X: (m, n)
        w: (n, 1) | (n, k)
        return (m, 1) | (m, k)
        """
    z = X @ w  # Multiplicacion matricial
    h = lambda x: 1 / (1 + np.exp(-x))  # Funcion sigmoide
    return h(z)

def funcionCosto(w, X, y, term_regularizacion):
    """Funcion de costo para la regresion logística
        Asume que X tiene una columna de 1 al inicio
        X: (m, n)
        w: (n, 1)
        y: (m, 1)
        """
    m = X.shape[0]

    h = sigmoid(w, X)  # dimension: (m, 1)
    costo = ((-1 / m) * ((y.T @ np.log(h)) + ((1 - y).T @ np.log(1 - h)))) + ((term_regularizacion / (2 * m)) * np.sum(w ** 2))

    # -- Calculo del gradiente
    gradiente = (1 / m) * (X.T @ (h - y))  # Para el bias
    gradiente[1:] = gradiente[1:] + ((term_regularizacion / m) * w[1:])  # (n, 1) para los demás pesos
    return gradiente, costo.item()


def descenso_gradiente(X, y, w, lr, epocas, term_regularizacion):
    """Rutina de minimizacion
       X: (m, n)
       w: (n, 1)
       y: (m, 1)
       Devuelve los pesos óptimos de w y el costo en cada iteracion
       """
    m = X.shape[0]  # Obtener el total de datos
    j_tot = []  # Costo en cada epoca
    for epoca in range(epocas):
        gradiente, costo = funcionCosto(w, X, y, term_regularizacion)
        j_tot.append(costo)
        w = w - (lr * gradiente)
    return w, j_tot  # dim de w (n, 1)

def oneVsAll(X, y, clases, lr, epocas, term_regularizacion):
    n = X.shape[1] # número de características
    matriz_pesos = pesos_iniciales(n, clases) # (n, k)

    historial_costos = []
    for k in range(clases):
        print(f"--> Entrenando modelo para la clase {k}")
        # vector columna de pesos para una clase
        w = matriz_pesos[:, k].reshape(-1, 1)
        # Convertir las etiquets a un vector logico para la clase actual
        y_bin = (y == k).astype(int)
        # Obtener parametros optimos
        w_optim, j_tot = descenso_gradiente(X, y_bin, w, lr, epocas, term_regularizacion)
        historial_costos.append(j_tot)
        # Guardar los pesos optimos en la matriz de pesos
        matriz_pesos[:, k] = w_optim.flatten()

    return matriz_pesos, historial_costos

def predictOneVsAll(W, X):
    """Prediccion usando un clasificador multiclase
        Asume que X tiene una columna de unos al inicio
        X: (m, n)
        W, (n, k)
        Devuelve la clase (1, ..., k) con la probabilidad más alta
        Devuelve una matriz de tamaño (m, 1)
        """
    predicciones = sigmoid(W, X)  # (m, k)
    # con np.argmax se obtiene el indice con mayor valor (probabilidad) de cada fila (axis=1)
    return np.argmax(predicciones, axis=1).reshape(-1, 1)


def regresionLogisticaMulticlase(df, epocas=100, lr=0.5):
    X = df["text"]
    Y = df["polaridad"]

    X = X.to_numpy()
    Y = Y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    tdidf_vectorizer = TfidfVectorizer(
        max_features=1000, # limite del vocabulario
        ngram_range=(1,2) # considerar unigramas y bigramas
    )

    X_train_tfidf = tdidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tdidf_vectorizer.transform(X_test)

    # Convertir las etiquetas en un vector columna
    train_y = y_train.reshape(-1, 1)
    test_y = y_test.reshape(-1, 1)

    # Agregar 1 al inicio de cada imagen
    X_train_tfidf = np.hstack((np.ones((X_train_tfidf.shape[0], 1)), X_train_tfidf.toarray()))
    X_test_tfidf = np.hstack((np.ones((X_test_tfidf.shape[0], 1)), X_test_tfidf.toarray()))

    print("x_train", X_train_tfidf.shape)
    print("x_test", X_test_tfidf.shape)
    print("train_y", train_y.shape)
    print("test_y", test_y.shape)

    term_regularizacion = 0.5
    # epocas = 500
    clases = 3

    print("Entrenando modelos....")
    W_optim, historial_costos = oneVsAll(X_train_tfidf, train_y, clases=clases, lr=lr,
                                         epocas=epocas,
                                         term_regularizacion=term_regularizacion)
    print(f"Tamaño de la matriz de pesos óptimos: {W_optim.shape}")

    print(f"lr: {lr}")
    print(f"term_regularizacion: {term_regularizacion}")
    print(f"epocas: {epocas}")

    list_epocas = np.arange(0, epocas)
    for idx, historial in enumerate(historial_costos):
        plt.plot(list_epocas, historial, label=f"modelo {idx}")
        plt.legend()
    plt.show()

    print("--- Predicciones con los datos de prueba ---")
    y_pred = predictOneVsAll(W_optim, X_test_tfidf)
    precision = np.sum(y_pred == test_y) / X_test_tfidf.shape[0] * 100.0
    print(f"Predicción del modelo: {precision}%")

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Reporte de clasificación")
    print(classification_report(y_test, y_pred))

    matriz_confusion = confusion_matrix(y_test, y_pred)
    mostrar_matriz_confusion(matriz_confusion)



