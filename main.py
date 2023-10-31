from fastapi import FastAPI, HTTPException
import pandas as pd
from typing import List

app = FastAPI()

df_dev = pd.read_csv('./sources/df_dev.csv')
df_userdata = pd.read_csv('./sources/df_userdata.csv')
df_genres = pd.read_csv('./sources/df_genres.csv')
df_dev_rev = pd.read_csv('./sources/df_dev_rev.csv')
df_ml = pd.read_csv('./sources/df_ml.csv')
df_sg = pd.read_csv('./sources/steam_games.csv')

@app.get('/developer/{developer}')
def developer(developer: str):
    # Para evitar la diferencia entre minusculas y mayusculas, transformamos el str en minuscula
    developer = developer.lower()

    df = df_dev.copy()
    df['developer'] = df['developer'].str.lower()
    
    if any(df['developer'] == developer):
        # Primero filtramos por la empresa desarrolladora
        df_dev_free = df[df['developer'] == developer]

        # Filtrarmos para obtener solo los ítems con precio igual a 0.00 (contenido Free)
        free_items_df = df_dev_free[df_dev_free['price'] == 0.00]

        # Calculamos la cantidad de ítems por año
        items_per_year = df_dev_free.groupby('release_year')['item_id'].count()

        # Calculamos la cantidad de ítems Free por año
        free_items_per_year = free_items_df.groupby('release_year')['item_id'].count()

        # Combinar ambas series en un DataFrame
        stats_df = pd.DataFrame({
            'Cantidad de Items': items_per_year,
            'Contenido Free': free_items_per_year,
        }).reset_index()

        # Reemplazamos los NaN por 0
        stats_df['Contenido Free'] = stats_df['Contenido Free'].fillna(0)

        # Calcular el porcentaje de contenido Free
        stats_df['Contenido Free'] = (stats_df['Contenido Free'] / stats_df['Cantidad de Items'] * 100).round(2).astype(str) + '%'
    
        return stats_df.to_dict(orient="records")
    
    return 'No se encontro el desarrollador indicado'


@app.get('/user/{user_id}')
def userdata(user_id: str):
    # Filtramos el DataFrame por el usuario especificado
    user_data = df_userdata[df_userdata['user_id'] == user_id]

    if user_data.empty:
        raise HTTPException(status_code=404, detail='No se encontraron datos para el usuario especificado')

    # Calculamos la cantidad de dinero gastado por el usuario
    spent_money = user_data['price'].sum()

    # Calculamos el porcentaje de recomendación en base a la columna 'recommend'
    # Se asume que True es una recomendación positiva y se calcula el porcentaje de True en 'recommend'
    recommend_percentage = (user_data['recommend'].sum() / len(user_data)) * 100

    # Calculamos la cantidad de items
    num_items = len(user_data)

    return {
        "Cantidad de dinero gastado": f'{spent_money:.2f} USD',
        "Porcentaje de recomendación": f"{recommend_percentage:.2f}%",
        "Cantidad de items": num_items
    }


@app.get('/genre/{genero}')
def UserForGenre(genero: str):
    # Para evitar la diferencia entre minusculas y mayusculas, transformamos el str en minuscula
    genero = genero.lower()
    
    if any(df_genres['genres'] == genero):
        # Filtramos el df por el genero indicado
        df_genero = df_genres[df_genres['genres'] == genero]

        # Eliminamos las columnas innecesarias y agrupamos por el 'user_id' con la sumatoria
        # de las horas jugadas por usuario
        df_user_cum = df_genero.drop(columns=['genres', 'release_year'], axis=1)
        df_user_cum = df_user_cum.groupby('user_id')['playtime_forever'].sum().reset_index()

        # Buscamos el usuario con mayor cantidad de horas jugadas
        usuario_mayor_horas = df_user_cum.loc[df_user_cum['playtime_forever'].idxmax(), 'user_id']

        # Filtramos el df por el usuario
        df_user_years = df_genero[df_genero['user_id'] == usuario_mayor_horas]

        # Agrupar por año y sumar las horas jugadas
        acumulacion_por_anio = df_user_years.groupby('release_year')['playtime_forever'].sum()

        return f'Usuario con más horas jugadas para Género {genero}: {usuario_mayor_horas}, Horas jugadas:{acumulacion_por_anio.to_dict()}'
    
    else:
        return 'El genero ingresado es inválido'
    
@app.get('/best_developer_year/{anio}')
def best_developer_year(anio: int):
    # Filtramos el DataFrame por las reviews cuya recomendacion es True y el sentiment_analysis igual a 2
    df_rec = df_dev_rev[df_dev_rev['recommend'] == True]
    df_rec = df_rec[df_rec['sentiment_analysis'] == 2]

    # Filtramos el DataFrame por el anio indicado
    df_rec = df_rec[df_rec['release_year'] == anio]

    if len(df_rec) > 0:
        # Eliminamos las columnas innecesarias
        df_rec = df_rec.drop(columns=['user_id', 'item_id', 'sentiment_analysis', 'price', 'release_year'])

        # Agrupamos por el desarrollador
        df_rec = df_rec.groupby('developer', as_index=False).sum()

        # Ordenamos el DataFrame de mayor a menor
        top_developers = df_rec.sort_values(by='recommend', ascending=False)

        # Crear una lista de diccionarios con el formato deseado
        top_developers_list = [{"Puesto {} : {}".format(i+1, developer)} for i, developer in enumerate(top_developers.head(3)['developer'])]

        return top_developers_list
    
    else:
        return 'No hubo recomendaciones positivas para dicho anio'
    
@app.get('/developer_reviews/{developer}')
def developer_reviews_analysis(developer: str):
    # Convertimos la columna 'sentiment_analysis' en dummies
    df_dummies = pd.get_dummies(df_dev_rev['sentiment_analysis'])
    df_dummies = pd.concat([df_dev_rev, df_dummies], axis=1)

    # Filtramos por el desarrollador indicado
    df_developer = df_dummies[df_dummies['developer'] == developer.lower()]

    if len(df_developer) > 0:
        # Eliminamos las columnas innecesarias
        df_developer = df_developer.drop(columns=['user_id','item_id','recommend','item_id','sentiment_analysis','price','release_year'], axis=1)

        # Agrupamos segun el desarrollador
        df_developer = df_developer.groupby('developer', as_index=False).sum()

        negativos = df_developer.iloc[0, 1]
        positivos = df_developer.iloc[0, 3]

        resultado = {developer: [f"Negativos = {negativos}", f"Positivos = {positivos}"]}

        return resultado

    else:
        return 'No se encontraron recomendaciones para dicho desarrollador'


def get_top_similar_games(item_id: int) -> List[str]:
    # Corregimos el index para que sea el item_id
    df = df_ml.set_index('item_id')

    # Buscamos los juegos similares para el id dado
    juegos_similares = df.loc[item_id].sort_values(ascending=False)

    # Filtramos por los primeros 5, desde la posicion 1 ya que el primer lugar corresponde a uno mismo
    top_juegos_similares = juegos_similares.iloc[1:].nlargest(5)

    # Buscamos los id's de dichos juegos y los convertimos en una lista
    lista_de_ids = top_juegos_similares.index.astype(int).tolist()

    # Buscamos los titulos de los juegos
    titulos_top = df_sg.loc[df_sg['item_id'].isin(lista_de_ids), 'title'].values.tolist()

    return titulos_top

@app.get('/recomendacion_juego/{item_id}')
def recomendacion_juego_endpoint(item_id: int) -> List[str]:
    return get_top_similar_games(item_id)