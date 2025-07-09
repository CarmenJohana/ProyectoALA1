import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------- 1. Carga la muestra aleatoria ----------------------
def tomar_muestra_csv(ruta_csv, n_muestras):
    """Devuelve un DataFrame con una muestra aleatoria de filas del archivo CSV."""
    with open(ruta_csv, 'r', encoding='utf-8') as f:
        n_filas = sum(1 for _ in f) - 1
    if n_muestras > n_filas:
        n_muestras = n_filas
    filas_a_ignorar = random.sample(range(1, n_filas + 1), n_filas - n_muestras)
    filas_a_saltar = sorted(filas_a_ignorar)
    # Guardamos las filas ignoradas en un archivo
    with open("FilasIgnoradas.txt", 'w') as f_out:
        for fila in filas_a_saltar:
            f_out.write(f"{fila}\n")

    print(f"Se ignoraron {len(filas_a_saltar)} filas. Guardado en: {"FilasIgnoradas.txt"}")
    df = pd.read_csv(ruta_csv, skiprows=filas_a_saltar)
    return df

# ------------- 2. Codifica las variables categóricas ---------------
def codificar_categoricas(X, codificadores=None):
    categoricas = X.select_dtypes(include='object').columns
    if codificadores is None:
        codificadores = {}
        for col in categoricas:
            valores_unicos = X[col].dropna().unique()
            codificadores[col] = {val: idx for idx, val in enumerate(valores_unicos)}

    for col in categoricas:
        if col in codificadores:
            X[col] = X[col].map(codificadores[col])
        else:
            # Para evitar columnas inesperadas al tratar con test.csv:
            print(f"Columna inesperada en test: {col} — ignorando o asignando NaN.")
            X[col] = None  # o X.drop(columns=[col], inplace=True) si prefieres eliminarla

    return X, codificadores
# ------------- 3. Preprocesa los dataframes ---------------
def preprocesar_df(df, incluir_target=True):
    if incluir_target:
        y = df['HasDetections']
        X = df.drop(columns=['HasDetections',  'MachineIdentifier'], errors='ignore')
    else:
        y = None
        X = df.drop(columns=['MachineIdentifier'], errors='ignore')

    return X, y


def procesar_train(df):
    df_copy = df.copy()
    # Eliminamos columnas con más del 50% de valores nulos
    porcentaje_nulos = df_copy.isnull().mean()
    columnas_filtradas = porcentaje_nulos[porcentaje_nulos <= 0.5].index.tolist()
    df_copy = df_copy[columnas_filtradas]
    # Separamos columnas numéricas y categóricas
    df_num = df_copy.select_dtypes(include=['number'])
    df_cat = df_copy.select_dtypes(include='object')
    # Vamos a mostrar la matriz de correlación y las variables altamente correlacionadas antes de PCA
    mostrar_correlacion(df_num)
    mostrar_altamente_correlacionadas(df_num, umbral=0.8)
    # Reemplazamos nulos en numéricas
    medianas = df_num.median()
    df_num = df_num.fillna(medianas)
    # Escalado y PCA
    scaler = StandardScaler()
    df_num_scaled = scaler.fit_transform(df_num)
    pca_model = PCA(n_components=0.95)
    df_num_pca = pca_model.fit_transform(df_num_scaled)
    # Mostramos las componentes principales
    print("\n[ Componentes principales del PCA (varianza explicada >= 95%) ]")
    componentes = pca_model.components_
    for i, comp in enumerate(componentes, start=1):
        pesos = [(col, round(peso, 3)) for col, peso in zip(df_num.columns, comp)]
        pesos_ordenados = sorted(pesos, key=lambda x: abs(x[1]), reverse=True)[:5]  # top 5 por componente
        print(f"Componente {i}:")
        for variable, peso in pesos_ordenados:
            print(f"   {variable}: {peso}")
    # Codificamos variables categóricas
    df_cat_cod, codificadores = codificar_categoricas(df_cat)
    # Se concatenan las columnas transformadas numéricas (por PCA) y las categóricas codificadas en un solo array final, llamado X_train_final. 
    X_train_final = np.hstack([df_num_pca, df_cat_cod.values])
    
    return X_train_final, medianas, scaler, pca_model, codificadores, columnas_filtradas


# ========== 4. Visualización de y ==========
def graficar_distribucion_y(y, titulo='Distribución de HasDetections'):
    print("\nDistribución de HasDetections:")
    proporcion = y.value_counts(normalize=True).rename('proportion')
    print(proporcion)

    plt.figure(figsize=(6, 6))
    counts = y.value_counts()
    plt.pie(counts, labels=['No Detection (0)', 'Has Detection (1)'],
            autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])
    plt.axis('equal')
    plt.title(titulo)
    plt.show()

    
# ========== 5. Gráfico matriz de correlación ==========
def mostrar_correlacion(df):
    numericas = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numericas].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=True, fmt='.2f', cmap='coolwarm', square=True,
                annot_kws={"size": 5})  
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    plt.title("Matriz de correlación - Variables numéricas", fontsize=12)
    plt.tight_layout()
    plt.show()


# ========== 6. Aplicar preprocesamiento al test.csv ==========

def procesar_test(df, columnas_utilizadas, medianas, scaler, pca_model, codificadores):
    df_copy = df.copy()
    
    # Usa solo las columnas filtradas desde train
    df_copy = df_copy[columnas_utilizadas]
    
    # Separa columnas numéricas y categóricas
    df_num = df_copy.select_dtypes(include=['number'])
    df_cat = df_copy.select_dtypes(include='object')
    
    # Reemplazamos nulos con medianas aprendidas
    df_num = df_num.fillna(medianas)
    
    # Escalado y PCA
    df_num_scaled = scaler.transform(df_num)
    df_num_pca = pca_model.transform(df_num_scaled)
    
    # Codificación categórica
    df_cat_cod, _ = codificar_categoricas(df_cat, codificadores)
    
    # Concatenación
    X_test_final = np.hstack([df_num_pca, df_cat_cod.values])
    
    return X_test_final

# --- 7. Mostrar variables altamente correlacionadas
def mostrar_altamente_correlacionadas(df, umbral=0.8):
    numericas = df.select_dtypes(include=['number'])
    corr_matrix = numericas.corr().abs()
    correlaciones_altas = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= umbral:
                correlaciones_altas.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    if correlaciones_altas:
        print(f"\nVariables altamente correlacionadas (>|{umbral}|):")
        for var1, var2, valor in correlaciones_altas:
            print(f"{var1} <--> {var2} = {valor:.2f}")
    else:
        print(f"\nNo se encontraron variables con correlación mayor a {umbral}.")

# Main --------------------------------------------
# --- Paso 1: Cargamos los datos
df_train = tomar_muestra_csv('train.csv', 500)
df_train = df_train.set_index('MachineIdentifier')
# --- Paso 2: Preprocesamiento inicial de los datos de entrenamiento
X, y = preprocesar_df(df_train, incluir_target=True)
X_train_final, medianas, scaler, pca_model, codificadores, columnas_filtradas = procesar_train(X)
# Hacemos esta conversión para ver columnas:
X_train_final = pd.DataFrame(X_train_final, index=df_train.index)
# --- Paso 3: Visualización y original
graficar_distribucion_y(y, 'Distribución ORIGINAL')
# --- Paso 4: Recuperamos correspondencia entre filas
y_tratado = y.loc[X_train_final.index]
# --- Paso 5: Visualizamos la distribución de la variable objetivo y después de limpieza
graficar_distribucion_y(y_tratado, 'Distribución TRAS LIMPIEZA')
# --- Paso 6: Entrenamos un modelo RandomForestClassifier simple 
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train_final, y_tratado)

# --- Paso 7: Cargamos y preparamos el dataset de test. Como es muy grande, se usará una muestra aleatoria del mismo tamaño que la del train
test_df = tomar_muestra_csv('test.csv', 500)
test_df = test_df.set_index('MachineIdentifier')

X_test_final = procesar_test(test_df, columnas_filtradas, medianas, scaler, pca_model, codificadores)
# Hacemos esta conversión sobre el dataset de test ya preparado para ver columnas:
X_test_final = pd.DataFrame(X_test_final, index=test_df.index)

# Verificamos qué columnas hay en ambos dataframes
print("Lista X_train", list(X_train_final.columns))
print("Lista X_test", list(X_test_final.columns))
print(len(list(X_train_final.columns))==len(list(X_test_final.columns)))
print(clf.classes_)  # Esto dice qué clases aprendió el modelo

# --- Paso 8: Hacemos la predicción sobre el conjunto de test ya procesado
predicciones = clf.predict(X_test_final)

# --- Paso 9: Pasamos las predicciones hechas por el modelo a un archivo llamado resultados.csv
submission = pd.DataFrame({
    'MachineIdentifier': test_df.index,
    'HasDetections': predicciones
})

submission.to_csv('resultados.csv', index=False)
print("\nArchivo 'resultados.csv' generado correctamente")