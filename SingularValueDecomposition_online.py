import sys
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import lit,col
from functools import reduce
from pyspark.sql import SparkSession
from math import sqrt as sqrt
from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRowMatrix, IndexedRow
from pyspark.mllib.linalg import  Vectors
from pyspark.sql.functions import row_number,lit
from pyspark.sql.window import Window

if __name__ == "__main__":
    if len(sys.argv) !=3:
        print("Usage: spark-submit lr_assignment.py <movieID>", file=sys.stderr)
        sys.exit(-1)

spark = (SparkSession
        .builder
        .appName("Linear Regression using Spark")
        .getOrCreate())

sc = spark.sparkContext
sc.setLogLevel('WARN')

hdfs_path = str(sys.argv[2])
movie_id = int(sys.argv[1])
k = 150

df_ratings = spark.read.format("csv").load(hdfs_path+'/ratings.csv',header='true',inferSchema = 'true')
df_movies = spark.read.format("csv").load(hdfs_path+'/movies.csv',header='true',inferSchema = 'true')
df_movies_filter = spark.read.format("csv").load(hdfs_path+'/movies_filtered.csv',header='true',inferSchema = 'true')
df_bridge = spark.read.format("csv").load(hdfs_path+'/Bridge_SVD_online.csv',header='true',inferSchema = 'true')

V = np.load('V.npy')
U = np.load('U.npy')

def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

translated_movie_id = df_bridge.filter(col('movieId') == movie_id).select('row_num').rdd.map(lambda x : x[0]).collect()
translated_movie_id = translated_movie_id[0]

top_n = 20
sliced = V.T[:, :k]
indexes = top_cosine_similarity(sliced, translated_movie_id, top_n)

df_bridge_recc = df_bridge.filter(df_bridge.row_num.isin(indexes.tolist()))

recc_movies_list = df_bridge_recc.select('movieId').rdd.map(lambda x : x[0]).collect()

print("-----Input Movie------")
df_movies.filter(col('movieId') == movie_id).show(truncate = False)


df_movies.filter(df_movies.movieId.isin(recc_movies_list)).show(truncate = False)



