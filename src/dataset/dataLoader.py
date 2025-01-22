"""
Cargar los datos recolectados y combinarlos
"""
import pandas as pd
from datetime import datetime, timedelta
import os
import pytz # zona horaria
import unicodedata

path_dataset = os.path.join(os.getcwd(), "TikTok")
extension = ".json"
encoding = "utf-8"
nombre_guardado = "tiktok_dataset.csv"

def reemplazar_nbsp(texto):
  """Reemplaza el símbolo &nbsp; o '\xa0' por un espacio en blanco
  """
  #return string.replace("\xa0", " ")
  return unicodedata.normalize("NFKD", str(texto))

def corregir_fecha_tiktok(fecha, fecha_referencia):
    """Tranformar fechas a un formato común yyyy-mm-dd
    """
    fecha = reemplazar_nbsp(fecha)
    zona_horaria = pytz.timezone('America/Mexico_City')
    #fecha_actual = datetime.now(tz=pytz.utc).astimezone(zona_horaria)
    fecha_actual = pd.to_datetime(fecha_referencia)
    unidad = ""
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

def combinar_datos():
    count = 0
    dataframe = []

    # Recuperar cada archivo
    for nombre_archivo in os.listdir(path_dataset):
        if nombre_archivo.endswith(extension):
            count += 1
            path = os.path.join(path_dataset, nombre_archivo)
            df = pd.read_json(path, encoding=encoding)
            termino = nombre_archivo.replace(".json", "")
            df["termino"] = "_".join(termino.lower().split())
            dataframe.append(df)

    df_combinado = pd.concat(dataframe, ignore_index=True)
    print(f"Total de datos recolectados: {len(df_combinado)}\n")

    # Eliminar filas duplicadas y conservar solo el primer registro por URL
    df_combinado= df_combinado.drop_duplicates(subset='url', keep='first')
    df_combinado = df_combinado.reset_index(drop=True)

    # Corregir fecha
    df_combinado["date"] = df_combinado.apply(
        lambda row: corregir_fecha_tiktok(row["date"], row["fecha_recuperacion"]),
        axis=1
    )

    # Eliminar columna de caption y fecha_recuperacion
    df_combinado = df_combinado.drop("caption", axis=1)
    df_combinado = df_combinado.drop("fecha_recuperacion", axis=1)

    # Reordenar columnas
    nuevo_orden = ["date", "termino", "views", "url", "title", "hashtags"]
    df_combinado = df_combinado[nuevo_orden]

    # Renombrar date
    df_combinado = df_combinado.rename(columns={
        "date": "fecha_publicacion",
        "views": "vistas",
        "title": "titulo"
    })


    print(f"Total de archivos combinados {count}")
    print(df_combinado.info())
    print(df_combinado.describe())
    df_combinado.to_csv(nombre_guardado, index=False)


if "__main__" == __name__:
    combinar_datos()

