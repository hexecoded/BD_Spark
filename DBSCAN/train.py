"""
Autores: Grupo 4NN

Este script ha sido diseñado para realizar pruebas de escalado con
un 20, 30, 40, 50 y 60 % de los datos del dataset de Spotify.
Recibe como parámetro de entrada el número de núcleos que se desea
emplear, e internamente, se pueden encontrar algunas variables globales:


CORES  # Número de núcleos de CPU a usar
PARTS  # Número de particiones del dataset (para hacer repartition)
FRACS # Fracción de los datos aleatoriamente seleccionados
DIMS  # Número de componentes PCA seleccionadas en base a mayor varianza explicada
REPS # Repeticiones de ejecución por experimento
"""
# %%
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from pyspark.ml.feature import PCA
from pyspark.sql.functions import monotonically_increasing_id
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pyspark.sql.functions as sql_f
import time

from graphframes import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from scipy.spatial import distance
from pyspark.sql.functions import col, substring
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import substring
from pyspark.sql.functions import size, col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import Row
from itertools import combinations

import sys

num = int(sys.argv[1])

# Variable de entorno. Modificar en caso de disponer de varias
# versiones de Java instaladas.
os.environ["JAVA_HOME"] = r"/home/cristhian/Descargas/OpenJDK8U-jdk_x64_linux_hotspot_8u452b09/jdk8u452-b09"

# Preparamos donde almacenar los resultados
mediciones = pd.DataFrame(columns=["Porcentaje", f"T{num}"])


# %%
# Definición del entorno de ejecución
CORES = num  # Número de núcleos de CPU a usar
PARTS = num  # Número de particiones del dataset (para hacer repartition)
# Fracción de los datos aleatoriamente seleccionados
FRACS = [0.2, 0.3, 0.4, 0.5, 0.6]
DIMS = 14  # Número de componentes PCA seleccionadas en base a mayor varianza explicada
REPS = 3  # Repeticiones de ejecución por experimento
spark = (
    SparkSession.builder.master(f"local[{CORES}]")
    .appName(f"Local DT con {CORES} particiones")
    .config("spark.driver.memory", "20g")
    .config("spark.executor.memory", "20g")
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12")

    .getOrCreate()
)
sc = spark.sparkContext
sc.setCheckpointDir("/tmp/graphframes-checkpoints")

# # Carga de datos con Spark

for FRAC in FRACS:
    print("##################################################################")
    print(f"Tamaño de datos: {FRAC * 100}%")
    print("##################################################################")

    df = spark.read.format('csv') \
        .option('header', 'true') \
        .option('delimiter', ',') \
        .load('spotify_songs.csv') \
        .repartition(PARTS)

    # %% [markdown]
    # Mostramos los 10 primeros individuos del conjunto de datos. Observamos como la carga se ha completado, en principio, sin ningún problema aparente ni en los individuos ni en los atributos.

    # %% [markdown]
    # Observamos como tenemos para la variable playlist_genre algunos valores inesperados, tras revisar el conjunto de datos observamos algunos individuos en el conjuntos de datos que tienen valores erróneos. En la siguiente sección, el preprocesado, lidiamos con este problema.

    # %% [markdown]
    # # Preprocesado

    # %%
    # Lista de columnas a convertir
    cols_to_convert = [
        "track_popularity", "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
        "tempo", "duration_ms"
    ]

    # Castea las columnas especificadas a DoubleType
    for c in cols_to_convert:
        df = df.withColumn(c, col(c).cast(DoubleType()))

    # %% [markdown]
    # Observamos de nuevo el esquema de DataFrame donde ahora sí comprobamos que están las variables con su tipo correcto.

    # %% [markdown]
    # ## Eliminación de instancias corruptas

    # %% [markdown]
    # Tras el análisis se han comprobado la existencia de ciertas instancias con valores corruptos en el conjunto de datos. Proponemos una solución directa, que en el contexto de Big Data y nuestro dataset (que tiene una suficiente cantidad de individuos) vemos viable y razonable: la eliminación directa. Para ello, se filtran únicamente las instancias que cumplen con alguno de los posibles valores de la variable categórica: playlist_genre (vista en el análisis).

    # %% [markdown]
    # Utilizamos la construcción de una lista manual de los valores correctos y la función filter() para su correcto filtrado.

    # %%

    valid_genres = ['pop', 'rock', 'rap', 'edm', 'r&b', 'latin']
    df = df.filter(col('playlist_genre').isin(valid_genres))

    # %% [markdown]
    # A modo de comprobación, mostramos y contamos los valores únicos de esa variable para comprobar que ya no existen los valores erróneos que obteníamos antes. Ahora sí, tenemos los verdaderos valores y su verdadera cantidad. Como no hemos detectado ningunos otros valores erróneos, asumimos que se han filtrado el resto de valores correctamente, y ya tenemos el conjunto de datos limpio.

    # %% [markdown]
    # ## Eliminación de variables con valores únicos

    # %% [markdown]
    # Se eliminarán las variables que tienen un valor identificador para cada canción. Las variables como: track_id, track_name, track_artist, track_album_id, track_album_name, playlist_name y playlist_id son eliminadas en este apartado, porque actúan como identificadores únicos y no aportan valor predictivo ni patrones útiles al análisis; al tener un valor distinto para cada fila, no permiten generalización, pueden introducir ruido y aumentarían la complejidad del modelo sin beneficio real.

    # %%
    cols_to_drop = [
        'track_id', 'track_name', 'track_artist',
        'track_album_id', 'track_album_name',
        'playlist_name', 'playlist_id'
    ]

    df = df.drop(*cols_to_drop)

    # %% [markdown]
    # ## Inclusión de un ID propio

    # %% [markdown]
    # A pesar de haber eliminado las variables identificadoras como track_id o track_name, se incluirá una variable id (de construcción propia) que no será utilizada en el algoritmo de clustering, pero que servirá como referencia para realizar visualizaciones pertinentes. Esta variable nos permitirá vincular cada observación con sus valores originales de las variables categóricas playlist_genre y playlist_subgenre, facilitando así la interpretación y comparación de los resultados obtenidos por el clustering con las categorías iniciales de las playlists.

    # %% [markdown]
    # Para ello, utilizamos la función monotonically_increasing_id. Esta función genera valores enteros largos que aumentan de forma monótona, aunque no necesariamente consecutiva ni ordenada estrictamente, lo que garantiza unicidad sin necesidad de una columna de índice previamente definida.

    # %%

    # Añadimos la ID al comienzo de la línea
    df = df.withColumn("id", monotonically_increasing_id())

    # %% [markdown]
    # ## Aplicación de one-hot enconding de las variables categóricas

    # %% [markdown]
    # Se aplicará one-hot enconding a las variables: playlist_genre, playlist_subgenre. Y a la variable: track_album_release_date se le aplica un one-hot encoding por decada.

    # %%

    def one_hot_preprocess(df, categorical_cols):

        indexers = [StringIndexer(inputCol=col, outputCol=col +
                                  '_idx', handleInvalid='keep') for col in categorical_cols]
        encoders = [OneHotEncoder(
            inputCol=col + '_idx', outputCol=col + '_ohe') for col in categorical_cols]

        pipeline = Pipeline(stages=indexers + encoders)
        return pipeline.fit(df).transform(df)

    # %%
    categorical_cols = ['playlist_genre']
    df = one_hot_preprocess(df, categorical_cols=categorical_cols)

    # Extraer el año de la fecha (formato 'YYYY-MM-DD')
    df = df.withColumn("release_year", substring(
        "track_album_release_date", 1, 4).cast("int")).cache()

    # %%
    # Guardamos las variables 'playlist_genre' y 'playlist_subgenre' en otro dataframe
    # Nota: serán útiles más tarde para comparar con los resultados del Clustering
    df_var_saved = df.select('id', 'playlist_genre', 'playlist_subgenre')

    # Eliminación de variables transformadas y retiradas
    df = df.drop('playlist_genre', 'playlist_subgenre',
                 'track_album_release_date')

    # %%
    idx_cols = [col + '_idx' for col in categorical_cols]
    df = df.drop(*idx_cols)

    # %% [markdown]
    # ### Modificar la representación de one-hot Sparse Vector a columnas convencionales

    # %% [markdown]
    # Explicar muy bien porque hemos pensado en cierto punto de la implemetación cambiar a columnas convencionales.

    # %% [markdown]
    # ## Estandarización

    # %%
    columnas = df.columns

    if "id" in columnas:
        columnas.remove("id")

    # %%

    # Ensamblar todas los atributos en un único vector dentro de una sola columna
    assembler = VectorAssembler(inputCols=columnas, outputCol="features")
    df = assembler.transform(df)

    # Ajuste de la estandarización
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    # Eliminar la columna 'features'
    df = df.drop('features')

    # %% [markdown]
    # ## Aplicación de reducción de la dimensionalidad

    # %% [markdown]
    # ### Análisis de la varianza explicada de las componentes

    # %%

    # Obtener el tamaño del vector en la columna 'features'
    vector_size = df.select("scaled_features").first()[0].size

    pca = PCA(k=vector_size, inputCol="scaled_features",
              outputCol="pca_features")
    pca_model = pca.fit(df)

    # %% [markdown]
    # ### Reducción de la dimensionalidad con PCA

    # %%
    # Cantidad de componentes principales a conservar

    # Ajustar el modelo de PCA
    pca = PCA(k=DIMS, inputCol="scaled_features", outputCol="pca_features")
    pca_model = pca.fit(df)

    # Transformar el dataframe
    df = pca_model.transform(df)

    # Eliminar la variable de 'scaled_features'
    pca_data = df.drop('scaled_features')
    ###
    from pyspark.sql.functions import col, floor, sqrt, collect_set
    from pyspark.ml.functions import vector_to_array
    from graphframes import GraphFrame
    from itertools import combinations

    def euclidean_distance_expr(vec1, vec2, dim):
        """
        Calcula la distancia euclidiana entre dos vectores en un espacio de dimensión dada.

        :param vec1: Lista o vector (tipo iterable) con coordenadas del primer punto.
        :param vec2: Lista o vector (tipo iterable) con coordenadas del segundo punto.
        :param dim: Entero que representa la cantidad de dimensiones a considerar para el cálculo.
        :return: Valor flotante correspondiente a la distancia euclidiana entre los dos vectores.
        """
        return sqrt(sum([(vec1[i] - vec2[i]) ** 2 for i in range(dim)]))

    # Etiquetar núcleos y puntos base
    def label_points(row, min_pts):
        """
        Función que etiqueta puntos como núcleos si supera un número minimo de vecinos. Los puntos base son 
        los vecinos de un punto núcleo

        :param min_pts: Número mínimo de puntos (incluyendo el mismo) para ser considerado punto núcleo.
        :return: Función que toma una tupla (id, vecinos) y retorna una lista de tuplas (id, [(etiqueta, es_núcleo)]).
        """
        id = row["src"]
        neighbors = row["neighbors"]
        if len(neighbors) + 1 >= min_pts:
            out = [(id, (id, True))]
            out.extend([(n, (id, False)) for n in neighbors])
            return out
        else:
            return []

    # Combinar etiquetas por punto
    def combine_labels(x):
        """
        Combina múltiples etiquetas de clúster para un punto y determina si es un punto núcleo.

        :param x: Tupla donde el primer elemento es el id del punto, y el segundo es una lista de (etiqueta, es_núcleo).
        :return: Tupla (id, etiquetas de clúster, es_núcleo) con todas las etiquetas si es núcleo, o solo una si no lo es.
        """
        point, tags = x
        core = any(tag[1] for tag in tags)
        clusters = [tag[0] for tag in tags]
        return point, clusters if core else [clusters[0]], core

    def process_dataframe(spark, df, epsilon, min_pts, dim, checkpoint_dir):
        """
        Procesa un dataframe de entrada con el algoritmo DBSCAN
        :param spark: sesión de Spark
        :param df: dataframe de entrada donde cada fila tiene un id y valores 
        :param epsilon: parámetro de distancia de vecindad para DBSCAN
        :param min_pts: parámetro de mínimo de vecinos para DBSCAN
        :param dim: dimensiones de los datos de entrada
        :param checkpoint_dir: directorio requerido por Graphframe para almacenar datos
        :return: Un dataframe con el id de punto,su etiqueta de cluster y si es núcleo
        """
        # Convertimos el vector a array para poder acceder a los valores
        df = df.withColumn("value_array", vector_to_array(col("value")))

        # Seleccionamos un pivote aleatorio
        pivot_vector = df.select("value_array").limit(1).collect()[0][0]

        # Calcular la distancia al pivote
        dist_expr = sum([
            (col("value_array")[i] - float(pivot_vector[i])) ** 2 for i in range(dim)
        ])
        # Calcular la
        df = df.withColumn("pivot_dist", sqrt(dist_expr))

        # Asignación de la partición original
        df_primary = df.withColumn("partition", floor(
            col("pivot_dist") / epsilon).cast("int"))

        # Replicar cada punto para asignarlo a la partición adyacente
        df_adjacent = df.withColumn(
            "partition", (floor(col("pivot_dist") / epsilon) + 1).cast("int"))

        # Unir ambos DataFrames para que cada punto aparezca en ambas particiones
        df = df_primary.union(df_adjacent)

        # Alias para join consigo mismo
        df_a = df.alias("a")
        df_b = df.alias("b")

        # Join por partición y evitando duplicados (id_a < id_b)
        df_pairs = df_a.join(df_b, (col("a.partition") == col(
            "b.partition")) & (col("a.id") < col("b.id")))

        # Calcular distancia
        df_pairs = df_pairs.withColumn(
            "distance",
            euclidean_distance_expr(col("a.value_array"),
                                    col("b.value_array"), dim)
        )
        # Filtrar vecinos
        df_neighbors = df_pairs.filter(col("distance") < epsilon) \
            .select(col("a.id").alias("src"), col("b.id").alias("dst"))

        # Agrupar vecinos por punto
        adjacency = df_neighbors.groupBy("src").agg(
            collect_set("dst").alias("neighbors"))

        # Etiquetar núcleos y puntos base
        labeled_rdd = adjacency.rdd.flatMap(lambda x: label_points(x, min_pts))

        # Combinar etiquetas
        combined = labeled_rdd.groupByKey().mapValues(list).map(combine_labels).cache()

        # Crear dataframe seleccionando una sola etiqueta de cluster
        df_initial = combined.map(lambda x: (x[0], x[1][0], x[2])) \
            .toDF(["point", "cluster_label", "core_point"])

        # Crear vértices y aristas
        vertices = combined.flatMap(
            lambda x: [(cid,) for cid in x[1]]).distinct().toDF(["id"])
        edges = combined.flatMap(lambda x: combinations(x[1], 2)) \
            .map(lambda x: (x[0], x[1])) \
            .distinct().toDF(["src", "dst"])

        # Crear grafo
        spark.sparkContext.setCheckpointDir(checkpoint_dir)
        g = GraphFrame(vertices, edges)
        components = g.connectedComponents()

        result = df_initial.join(components, df_initial.cluster_label == components.id) \
            .select("point", "component", "core_point")

        return result
    ###

    df_pca = pca_data.select("id", "pca_features")
    df_preproc = df_pca.sample(
        withReplacement=False, fraction=FRAC, seed=123456)

    # Renombramos la columna por un nombre genérico como value
    df_preproc = df_preproc.withColumnRenamed("pca_features", "value").cache()

    # Mostrar el resultado

    # Ejecución del algoritmo
    EPSILON = 2
    MINPTS = 15

    tiempos = np.zeros(REPS)

    for i in range(REPS):
        start = time.perf_counter()
        df_clusters = process_dataframe(
            spark, df_preproc, EPSILON, MINPTS, 14, "checkpoints")

        end = time.perf_counter()

        elapsed = (end - start)
        print(
            f"Duración ejecición {i} con {FRAC * 100}% datos: {elapsed}")

        tiempos[i] = elapsed
        # Mostrar el resultado por cluster
        # df_counts = df_clusters.groupBy("component").count()
        # df_counts.show()

    nueva_fila = pd.DataFrame(
        [{"Porcentaje": FRAC, f"T{num}": np.mean(tiempos)}])
    mediciones = pd.concat([mediciones, nueva_fila], ignore_index=True)

mediciones.to_csv(f"meds_{CORES}")
