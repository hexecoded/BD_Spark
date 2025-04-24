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

    # %% [markdown]
    # # Hola - Esto deberíamos moverlo arriba para seguir con el esquema que se espera del notebook :)

    # %% [markdown]
    #     Contexto
    #     Objetivos
    #     Descripción detallada de la soluciones propuestas  <--- Poner aquí
    #         Breve discusión de las decisiones clave que hacen que tu algoritmo sea escalable
    #     Experimentos
    #         Estudio breve de escalabilidad (scale-up, size-up, speedup)
    #         Resultados acorde a métricas relevantes
    #     Discusión de resultados
    #     Resumen de la contribución de cada miembro del equipo

    # %%

    def __distance_from_pivot(pivot, dist, epsilon, operations):
        """
        Genera una función que asigna un punto a una partición según su distancia al pivote.

        :param pivot: Valor del pivote para calcular distancias.
        :param dist: Función de distancia que toma dos valores y retorna una distancia numérica.
        :param epsilon: Umbral de distancia para crear particiones.
        :param operations: (Opcional) Objeto para contar operaciones de distancia, con método `add()`.
        :return: Función que toma un objeto `x` y retorna una lista de tuplas con índice de partición y lista de `Row`s.
        """
        def distance(x):
            pivot_dist = dist(x.value, pivot)
            if operations is not None:
                operations.add()
            partition_index = math.floor(pivot_dist / epsilon)
            rows = [Row(id=x.id, value=x.value,
                        pivot_dist=dist(x.value, pivot))]
            out = [(partition_index, rows),
                   (partition_index + 1, rows)]
            return out
        return distance

    def __scan(epsilon, dist, operations):
        """
        Genera una función que identifica vecinos dentro de una partición que están a menos de `epsilon` de distancia.

        :param epsilon: Distancia máxima para considerar dos puntos como vecinos.
        :param dist: Función de distancia entre puntos.
        :param operations: (Opcional) Objeto para contar operaciones de distancia.
        :return: Función que toma una tupla con índice de partición y datos, y retorna lista de `Row`s con vecinos.
        """

        def scan(x):
            # El diccionario de salida tiene un ID de punto como clave y un conjunto de IDs de los puntos a una distancia
            # menor a epsilon. value contiene a los vecinos
            out = {}
            # El índice 0 es el índice de partición
            # El índice 1 son los datos
            partition_data = x[1]
            partition_len = len(partition_data)
            for i in range(partition_len):
                for j in range(i + 1, partition_len):
                    if operations is not None:
                        operations.add()
                    if dist(partition_data[i].value, partition_data[j].value) < epsilon:
                        # Tanto i como j están a una distancia menor a epsilon
                        if partition_data[i].id in out:
                            out[partition_data[i].id].add(partition_data[j].id)
                        else:
                            out[partition_data[i].id] = set(
                                [partition_data[j].id])
                        if partition_data[j].id in out:
                            out[partition_data[j].id].add(partition_data[i].id)
                        else:
                            out[partition_data[j].id] = set(
                                [partition_data[i].id])
            # Devuelve un punto y sus vecinos como tupla
            return [Row(item[0], item[1]) for item in out.items()]

        return scan

    def __label(min_pts):
        """
        Genera una función que etiqueta puntos como núcleos si supera un número minimo de vecinos. Los puntos base son 
        los vecinos de un punto núcleo

        :param min_pts: Número mínimo de puntos (incluyendo el mismo) para ser considerado punto núcleo.
        :return: Función que toma una tupla (id, vecinos) y retorna una lista de tuplas (id, [(etiqueta, es_núcleo)]).
        """
        def label(x):
            if len(x[1]) + 1 >= min_pts:
                # Usar ID como etiqueta de cluster
                cluster_label = x[0]
                # Se devuelve True para los puntos núcleo
                out = [(x[0], [(cluster_label, True)])]
                for idx in x[1]:
                    # Se devuelve False para los puntos base
                    out.append((idx, [(cluster_label, False)]))
                return out
            return []

        return label

    def __combine_labels(x):
        """
        Combina múltiples etiquetas de clúster para un punto y determina si es un punto núcleo.

        :param x: Tupla donde el primer elemento es el id del punto, y el segundo es una lista de (etiqueta, es_núcleo).
        :return: Tupla (id, etiquetas de clúster, es_núcleo) con todas las etiquetas si es núcleo, o solo una si no lo es.
        """
        # El elemento 0 es el ID del punto
        # El elemento 1 es una lista de tuplas con cluster y estiqueta de núcleo
        point = x[0]
        core_point = False
        cluster_labels = x[1]
        clusters = []
        for (label, point_type) in cluster_labels:
            if point_type is True:
                core_point = True
            clusters.append(label)
        # Si es núcleo se mantienen todas las etiquetas de cluster, si no solo una
        return point, clusters if core_point is True else [clusters[0]], core_point

    def process(spark, df, epsilon, min_pts, dist, dim, checkpoint_dir, operations=None):
        """
        Process given dataframe with DBSCAN parameters
        :param spark: spark session
        :param df: input data frame where each row has id and value keys
        :param epsilon: DBSCAN parameter for distance
        :param min_pts: DBSCAN parameter for minimum points to define core point
        :param dist: method to calculate distance. Only distance metric is supported.
        :param dim: number of dimension of input data
        :param checkpoint_dir: checkpoint path as required by Graphframe
        :param operations: class for managing accumulator to calculate number of distance operations
        :return: A dataframe of point id, cluster component and boolean indicator for core point
        """
        # Se elige el pivote aleatoriamente
        zero = df.rdd.takeSample(False, 1)[0].value

        # Se obtienen dos tuplas (partition_id, [point]) para cada punto en particiones contiguas
        step1 = df.rdd.flatMap(__distance_from_pivot(
            zero, dist, epsilon, operations))

        # Se obtienen tuplas (partition_id, [point, point, ...]) con los puntos de cada partición
        step2 = step1.reduceByKey(lambda x, y: x + y)

        # Se obtienen tuplas (point_id, {point_id, point_id, ...}) con los IDs de vecinos de cada punto en una partición
        step3 = step2.flatMap(__scan(epsilon, dist, operations))

        # Se obtienen tuplas (point_id, {point_id, point_id, ...}) uniendo los vecinos de particiones distintas
        step4 = step3.reduceByKey(lambda x, y: x.union(y))

        # Se obtienen tuplas (point_id, [(cluster_id, is_core)]) con una misma etiqueta de clúster
        # para los puntos núcleo y vecinos de un núcleo y un booleano para identificar si es punto núcleo
        step5 = step4.flatMap(__label(min_pts))

        # Se obtienen tuplas (point_id, [(cluster_id, is_core), ...]) con los etiquetados de cada punto
        step6 = step5.reduceByKey(lambda x, y: x + y)

        # Se obtienen tuplas (point_id, [cluster_id, cluster_id, ...], is_core) manteniendo:
        #     - Todas las etiquetas de clúster si es núcleo (al menos un is_core es True)
        #     - Solo una etiqueta de cluster si no es núcleo
        combine_cluster_rdd = step6.map(__combine_labels).cache()
        # Se crea un RDD que selecciona la primera etiqueta de cluster para cada punto

        id_cluster_rdd = combine_cluster_rdd.\
            map(lambda x: Row(point=x[0],
                cluster_label=x[1][0], core_point=x[2]))
        try:
            id_cluster_df = id_cluster_rdd.toDF()
            # Se crean los vértices del grafo extrayendo las etiquetas de clúster de cada punto
            vertices = combine_cluster_rdd.\
                flatMap(lambda x: [Row(id=item)
                        for item in x[1]]).toDF().distinct().cache()
            # Se generan las aristas del grafo entre todos los pares de etiquetas del mismo punto
            edges = combine_cluster_rdd. \
                flatMap(lambda x: [Row(src=item[0], dst=item[1])
                                   for item in combinations(x[1], 2)]). \
                toDF().distinct().cache()
            # Se establece el directorio de checkpoints, requisito para el procesamiento en GraphFrames
            spark.sparkContext.setCheckpointDir(checkpoint_dir)
            # Creación del grafo a partir de vértices y aristas
            g = GraphFrame(vertices, edges)
            # Se ejecuta el algoritmo de componentes conexas para consolidar las etiquetas de clúster
            connected_df = g.connectedComponents()

            # Se une el DataFrame original (con la primera etiqueta asignada) con los resultados
            # de componentes conexas, utilizando la etiqueta de clúster como llave.
            # La unión permite asignar a cada punto su clúster final (campo 'component').
            id_cluster_df = id_cluster_df.\
                join(connected_df, connected_df.id == id_cluster_df.cluster_label). \
                select("point", "component", "core_point")
            return id_cluster_df
        except ValueError:
            return None

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
        df_clusters = process(spark, df_preproc, EPSILON, MINPTS,
                              distance.euclidean, DIMS, "checkpoint").cache()
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
