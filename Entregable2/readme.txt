El código para este entregable se encuentra en los siguientes notebooks:
	
	- Análisis exploratorio de los datos:
	https://colab.research.google.com/drive/193lAaLKf3FukSgMucQ3bhW3dWLniNmPi?usp=sharing

	- Entrenamiento0:
	https://colab.research.google.com/drive/1LhClHa4VGeJ_sgtinhiE2wPRx1EhuWdJ?usp=sharing

	- Entrenamiento1: 
	https://colab.research.google.com/drive/16Ak0bVABQ3s18FgMg7By4Dk5tbg3X0qw?usp=sharing


Antes de correr cualquier notebook, asegúrese subir los siguientes dos archivos en el entorno de Colab:

    dtypes_dict
    	
	Archivo con los tipos de datos originales del conjunto, requerido para la lectura optimizada del dataset.
	(Se encuentra en la carpeta de Github del segundo entregable).

    .env
    	Archivo de entorno necesario para descargar los datos desde Kaggle. Su contenido debe ser:

		KAGGLE_USERNAME="su_usuario_de_kaggle"
		KAGGLE_KEY="su_api_key_de_kaggle"
	
	Para obtener estos datos:

	    Cree una cuenta en https://www.kaggle.com/ (si aún no tiene una).

	    Vaya a "My Account" y seleccione Create New API Token.

	    Esto descargará un archivo kaggle.json con sus credenciales.

	    Copie los valores de "username" y "key" y colóquelos en el archivo .env como se muestra arriba entre comillas.
