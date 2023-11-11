<p align="center"><img src="https://github.com/natiqueija/MVP-PI-1-Henry/assets/109183272/c9ddc924-53f0-46d1-adfe-868ea8cfad6e" width="250" height="250"></p>



<H1 align="center">Proyecto de recomendaciones de videojuegos para la plataforma Steam</H1>

MVP para una api de recomendaciones de videojuegos como primer proyecto individual de [Soy Henry](https://www.soyhenry.com/?utm_source=google&utm_medium=cpc&utm_campaign=GADS_SEARCH_ARG_BRAND&utm_content=brand&gad=1&gclid=Cj0KCQjw1OmoBhDXARIsAAAYGSFC2WWyy_RYznNkc6QevI2HP7hhqdfUyI1W1CofKjIFQpAZhyfTYccaAt-fEALw_wcB)

Primer proyecto individual realizado en la etapa de labs del bootcamp 'Data Scientist' de Henry, en el cual se nos pidió la realización de un modelo de Machine Learning contemplando desde el tratamiento y recolección de los datos hasta el entrenamiento y mantenimiento del modelo según llegan nuevos datos.


## Contexto

Steam es una plataforma multinacional de distribución digital de videojuegos desarrollada por Valve Corporation lanzada en 2003 como una forma para Valve de proveer actualizaciones automáticas a sus juegos, pero finalmente se amplió para incluir juegos de terceros. En esta ocasión Steam se encuentra en la busqueda de solucionar un problema de negocio creando un sistema de recomendación de videojuegos para usuarios.

Se nos pide la entrega rápida de un MVP (Minimum Viable Product), basado en el desarrollo una API (Application Programming Interface) disponibilizando así los datos de la empresa para la realización de las consultas necesarias, además del sistema de recomendación de videojuegos, por lo que decidimos utilizar el siguiente esquema de trabajo.

![Flujo de procesos](https://github.com/HX-PRomero/PI_ML_OPS/raw/main/src/DiagramaConceptualDelFlujoDeProcesos.png)


## Fuente de datos

Se nos proporcionaron 3 archivos json y un diccionario de datos, a los cuales debimos realizar diferentes procesos de ETL para dejarlos preparados para nuestro futuro modelo.

- Steam_games: información relacionada a los juegos de Steam, como Título, Género, Publisher, Developer, Fecha de lanzamientos, etc.
- User_items: información sobre el usuario y un listado de items que se haya descargado.
- User_reviews: información acerca de las reseñas de usuarios sobre los videojuegos.
- Diccionario de datos: diccionario con algunas descripciones de las columnas disponibles en el dataset.

Debido a recursos limitados, dichos datasets se pueden encontrar en el siguiente [link](https://drive.google.com/drive/folders/18ubpDrUfChnage6gRNDTu68SxJlr4_xZ?usp=drive_link).

## Análisis exploratorio de los datos (EDA) y Transformaciones (ETL)

Se realizaron las transformaciones de cada dataset por separado para un mejor entendimiento de los datos. 

### Dataset "steam_games"
- Como primera observación vemos que los primeros 88310 registros están completamente vacíos por lo que se procede a eliminarlos y reindexar.
- Se realizan transformaciones relacionadas a tipos de datos, extracción del año en la columna *release_date*, eliminación de columnas con información innecesaria para este análisis.
- Se analizan los datos faltantes y duplicados, se eliminan 2 registros que no aportaban valor, pero en general se mantienen los valores vacíos ya que creemos que aportan valor a otro tipo de análisis.
- En la columna "price" decidimos reemplazar los valores NaN por el valor 0.00 para poder realizar los calculos correspondientes en cada consulta, además de reemplazar los valores que se presentaron en formato str por 0.00 en los casos de juegos gratuitos (Free) o Demos y por lo valores numéricos a aquellos que se presentaron como "A partir de $".
- Se realizan ciertos gráficos para visualizar la distribución de los datos:

1) Gráfico de barras del top 10 de desarrolladores con mas juegos y la cantidad de cada uno.
<div align="center">
    <img src="https://github.com/natiqueija/Henry-PI1/assets/109183272/44fa169b-99d4-469a-bf49-06d69e25bf6a" alt="Top 10 de desarrolladores con más juegos" width="70%">
</div>
<br>

2) Gráfico Boxplot de la columna "price" para buscar los valores outliers
<div align="center">
    <img src="https://github.com/natiqueija/Henry-PI1/assets/109183272/dbc6689f-d937-4c35-8acc-af9aeea572d1" alt="Outliers en el precio de los juegos" width="70%">
</div>
<br>

3) Así como también un gráfico de dispersión de la columna "price" para analizarlo mejor
<div align="center">
    <img src="https://github.com/natiqueija/Henry-PI1/assets/109183272/1a4ea80b-ff79-4665-817e-c7c5633b6d9f" alt="Dispersión en el precio de los juegos" width="70%">
</div>
<br>

4) Gráfico de torta para visualizar la diferencia entre la cantidad de juegos pagos vs la cantidad de juegos gratuitos que tiene la plataforma.
<div align="center">
    <img src="https://github.com/natiqueija/Henry-PI1/assets/109183272/6e40a324-98e9-43b0-b0ac-e8f633175674" alt="Juegos pagos vs juegos gratuitos" width="50%">
</div>
<br>

5) Por útlimo, un gráfico de barras para ver la cantidad de juegos lanzados por año por la plataforma.
<div align="center">
    <img src="https://github.com/natiqueija/Henry-PI1/assets/109183272/daf8ba9d-d555-4daf-ac0d-ac6a1a6dd964" alt="Cantidad de juegos lanzados por año" width="80%">
</div>
<br>

### Dataset "user_reviews"
- Como primera observación vemos que las reseñas están en un formato tipo json, por lo que, utilizamos primero un .explode() para lograr tener una reseña por fila y desanidamos el diccionario creando nuevas columnas para ver los valores.
- Encontramos que hay 28 registros con datos vacíos por lo que decidimos eliminarlos.
- Se realizan transformaciones relacionadas a tipos de datos, extracción del año en la columna *posted* y creamos la columna *year_posted*, eliminación de columnas con información innecesaria para este análisis.
- Por último se realizó un Análisis de sentimiento con NLP basado en la reseña escrita, en la que analizamos el texto y nos devuelve la siguiente escala: toma el valor '0' si es negativa, '1' si es neutral y '2' si es positiva. Este proceso se realizó a través de la función "SentimentIntensityAnalyzer()" de nltk.sentiment.vader. Tomando en consideración que de no ser posible este análisis por estar ausente la reseña escrita, debe tomar el valor de 1.
- Se realizan ciertos gráficos para visualizar la distribución de los datos:

1) Gráfico de torta para visualizar la cantidad de recomendaciones que hubieron en la plataforma.
<div align="center">
    <img src="https://github.com/natiqueija/Henry-PI1/assets/109183272/ccd1f5e9-dfb4-4127-8baa-5c03fea8799b" alt="Cantidad de recomendaciones en la plataforma" width="50%">
</div>
<br>

2) Gráfico de barras horizontales para visualizar el top 10 de juegos con mayor cantidad de recomendaciones.
<div align="center">
    <img src="https://github.com/natiqueija/Henry-PI1/assets/109183272/0e3f3cae-7530-4e0d-852d-7efdd793a300" alt="Top 10 juegos con mayor cantidad de recomendaciones" width="80%">
</div>
<br>

3) Gráfico de barras para analizar la cantidad de reviews por categoria del analisis de sentimiento.
<div align="center">
    <img src="https://github.com/natiqueija/Henry-PI1/assets/109183272/7b7124aa-a46b-4512-a55d-51660eb41c3b" alt="Cantidad de reviews por categoría" width="60%">
</div>
<br>

### Dataset "user_items"
- Como primera observación vemos que hay 1357 registros duplicados por lo que decidimos eliminarlos.
- Eliminamos los registros que no tengan ningún item descargado, es decir, que la columna *items_count* sea igual a 0.
- Vemos también que la columna items se encuentra anidada por lo que realizamos un .explode() y luego un json_normalize() para desanidarla.
- Se realizan transformaciones relacionadas a tipos de datos, extracción del año en la columna *posted* y creamos la columna *year_posted*, eliminación de columnas con información innecesaria para este análisis.


## Desarrollo de FastAPI

Se proponen las siguientes funciones para los endpoints que se consumirán en la API:

1) def developer( desarrollador : str ): Ingresando el nombre de un desarrollador, devuelve la cantidad de items lanzados y porcentaje de contenido Free por año según empresa desarrolladora.
Para esto, se decide utilizar directamente el dataset proporcionado, luego de haber pasado por el EDA y el ETL, pero quedandonos únicamente con la información relevante. 


2) def userdata( User_id : str ): Ingresando el id de un usuario devuelve la cantidad de dinero gastado por dicho usuario, el porcentaje de recomendación en base a la cantidad de items recomendados sobre el total de items que descargó y la cantidad de items descargados.
Para esto necesitamos hacer un merge entre los 3 datasets a través del id del item ya que encontramos la siguiente información por dataset:

- steam_games -> price (podemos calcular cuánto gasto el usuario)
- user_reviews -> recommend (podemos contar la cantidad de recomendaciones que realizó)
- user_items -> count (a traves de un count podemos saber la cantidad de items que descargó)


3) def UserForGenre( genero : str ): Ingresando un género, devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
Para esto necesitamos hacer un merge entre los datasets 'steam_games' y 'user_items' a través del id del item, ya que encontramos la siguiente información por dataset:

- steam_games -> genres (información sobre el género del videojuego) y release_year (información sobre el año de lanzamiento del juego)
- user_items -> user_id (id de usuario) y playtime_forever (cantidad de horas jugadas totales)


4) def best_developer_year( año : int ): Ingresando un año, devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. Es decir, que recomendaron el juego y además dejaron una reseña positiva. 

5) def developer_reviews_analysis( desarrolladora : str ): Ingresando el nombre de un desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.
Para estos últimos endpoints, decidimos realizar un merge entre el DataFrame 'steam_games' y 'user_reviews' a través del id del item, ya que encontramos la siguiente información por dataset:

- steam_games -> release_year (año de lanzamiento del videojuego) y developer (desarrollador)
- user_reviews -> recommend (verdadero o falso si es que recomendó el juego o no) y sentiment_analysis (categorización de la reseña en base a si fué positiva, negativa o neutra realizada a través de un análisis de sentimiento)


## Sistema de recomendación

Decidimos crear un sistema de recomendacion a traves de la similitud del coseno, en el cual, ingresando un item_id me devuelva una lista de los 5 juegos recomendados segun el genero del juego. Para realizar esto creamos un df con los géneros, item_id y título del juego.

!! Dada la limitación de espacio, tanto en github como en render, se deciden tomar para el análisis únicamente el top 100 de juegos con mayos cantidad de horas jugadas. 

- Creamos una tabla pivot con el índice como los item_id y las columnas de generos explotadas mediante un proceso de one-hot encoding para poder realizar el cálculo de la similitud del coseno utilizando la librería sklearn y evaluando cuáles son los 5 juegos más similares al id ingresado.

Se incluye en la API la siguiente función de recomendación:
- def recomendacion_juego( id de producto ): Ingresando el id de producto, recibimos una lista con 5 juegos recomendados similares al ingresado.


## Deployment

Se realiza el deployment en la plataforma de Render para disponibilizar las consultas a cualquier persona con internet. 
A continuación se encuentra el link a Render: [Henry-PI1]([https://mvp-pi-1-soyhenry.onrender.com/docs#/](https://henry-pi1.onrender.com/docs#/))

## Video explicativo

- Se realizó un video explicativo de todo el proyecto, el cual se encuentra alojado en el siguiente [Drive](https://drive.google.com/drive/folders/18ubpDrUfChnage6gRNDTu68SxJlr4_xZ?usp=sharing)

## Contacto

- Mail: natiqueija@gmail.com
- Linkedin: [Natalia Queija](https://www.linkedin.com/in/natalia-queija/)
