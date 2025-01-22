import plotly.express as px
import pandas as pd

def mostrar_matriz_confusion(matriz_confusion):
    # Convertir a un DataFrame para facilitar la visualización
    cm_df = pd.DataFrame(matriz_confusion,
                         index=["NEG","NEU", "POS"],
                         columns=["NEG", "NEU", "POS"])

    # Graficar con Plotly Express
    fig = px.imshow(cm_df,
                    text_auto=True,  # Muestra los valores en las celdas
                    color_continuous_scale='ylorbr',  # Escala de color
                    title='Matriz de Confusión',
                    labels=dict(
                        color='Número de muestras'))  # Etiqueta para la barra de color
    fig.update_layout(
        xaxis_title='Predicción',  # Título para el eje X (Predicción)
        yaxis_title='Real',  # Título para el eje Y (Real)
    )

    fig.update_layout(
        font=dict(size=14))  # Opcional: Ajustar el tamaño de fuente
    fig.show()


def mostrar_grafica(df):
    for col in df.columns:
        # print(col)
        # Variable numericas
        if df[col].dtype in ["int64", "float64"]:

            fig = px.scatter(df, x=col,
                             title=f"Scatter de {col}",
                             labels={col: col})
            fig.show()

            fig = px.box(df,x=col,
                         title=f"Boxplot de {col}",
                         labels={col: col})
            fig.show()

            fig = px.histogram(df, x=col,
                               title=f"Histograma de {col}",
                               labels={col: col})
            fig.show()

        elif df[col].dtype == "object":
            fig = px.bar(df, x=col,
                         title=f"Grafico de barras de {col}",
                         labels={col: col})
            fig.show()

        elif df[col].dtype == "datetime64[ns]":
            fig = px.line(df, x=col,
                          title=f"Grafico de lines de {col}",
                          labels={col: col})
            fig.show()

        else:
            print(f"No se puede graficar la columna: {col}")