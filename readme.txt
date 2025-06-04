README - Proyecto Predicción de Malware con PCA y Random Forest

Descripción

Este proyecto implementa un modelo de aprendizaje automático para predecir si un dispositivo será atacado por malware, usando un muestreo del dataset Microsoft  Malware Prediction (Kaggle). Se aplica reducción de dimensionalidad mediante PCA  y se entrena un clasificador Random Forest.

Requisitos

Instale las siguientes librerías de Python en caso de no tenerlas:

pip install numpy pandas matplotlib seaborn scikit-learn



Archivos necesarios

Los siguientes dos archivos son necesarios para poder correr el código:


    train.csv : Dataset de entrenamiento original.

    test.csv : Dataset para predicción (se toma una muestra aleatoria).

Para descargarlos, vaya a kaggle, regístrese o ingrese a su cuente y descargue los archivos que aparecen aquí: https://www.kaggle.com/competitions/microsoft-malware-prediction/data.
Para eso, en la parte final de la página, de click en Download All.

Uso

    - Coloque los archivos sample_submission.csv, train.csv y test.csv en la misma carpeta que el script PCA_Small_Sample.py

    - Ejecute el script de Python que contiene el código.

    - El script:

        Toma muestras aleatorias de 500 filas de ambos datasets para facilitar el manejo.

        Preprocesa datos (elimina columnas con muchos nulos, codifica variables categóricas).

        Aplica escalado y PCA para reducir dimensionalidad (95% varianza explicada).

        Visualiza la distribución de la variable objetivo HasDetections.

        Entrena un modelo Random Forest con la muestra procesada.

        Procesa el dataset de test de forma similar y predice con el modelo entrenado.

        Guarda las predicciones en un archivo resultados.csv.

Archivos generados

    - FilasIgnoradas.txt : Registro de las filas omitidas en la muestra aleatoria (para reproducibilidad).

    - resultados.csv : Predicciones del modelo para el dataset de test.


Notas importantes

    El dataset original es muy grande (>80 millones filas), por eso se usa muestreo aleatorio para manejarlo en memoria.

    Se detectaron variables altamente correlacionadas que justifican el uso de PCA.

    El código incluye visualizaciones de correlación y distribución para análisis exploratorio.

    Se requiere reproducir exactamente el proceso de codificación para el conjunto de test usando los codificadores aprendidos en train.