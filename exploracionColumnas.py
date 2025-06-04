import pandas as pd

# Vamos a leer sólo la cabecera del archivo paraa obtener el nombre de las columnas
# sin tener problemas por el tamaño del dataframe
df_header = pd.read_csv("train.csv", nrows=0)

# Mostramos la cantidad de columnas y sus nombres
print(f"\nTotal de columnas: {len(df_header.columns)}\n")
print("Lista de columnas:")


columna =list(df_header.columns)

print(columna)

# Aquí haremos un estudio de cada columna fijándonos en:
# - Cantidad de valores nulos
# - Tipo manejado
# - Valores que puede asumir y cantidad por cada valor

for col in columna:
    print(f"\n*** Análisis para columna: {col} ***")
    
    value_counts = pd.Series(dtype="int")
    total_nulls = 0
    dtype_col = None

    for chunk in pd.read_csv("train.csv", usecols=[col], chunksize=500_000):
        value_counts = value_counts.add(chunk[col].value_counts(dropna=False), fill_value=0)
        total_nulls += chunk[col].isna().sum()
        if dtype_col is None:
            dtype_col = chunk[col].dtype
    
    print(value_counts.sort_values(ascending=False))
    print("La cantidad de valores nulos es", int(total_nulls))
    print("El tipo de datos es ->", dtype_col)
