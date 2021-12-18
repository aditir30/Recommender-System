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
        print("Usage: spark-submit SingularValueDecomposition.py <movieID> </user/<username>/dataset>", file=sys.stderr)
        sys.exit(-1)

spark = (SparkSession
        .builder
        .appName("Singular Value Decomposition")
        .getOrCreate())

sc = spark.sparkContext
sc.setLogLevel('WARN')

def matrix_multiply(row):
  row = row.asDict()
  for i in independent_columns:
    (ki, vi) = (i, row[i])
    for j in independent_columns:
        (kj, vj) = (j, row[j])
        yield ((ki,kj), vi * vj)

def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

k = 150
hdfs_path = str(sys.argv[2])
movie_id = int(sys.argv[1])

df_ratings = spark.read.format("csv").load(hdfs_path+'/ratings.csv',header='true',inferSchema = 'true')
df_movies = spark.read.format("csv").load(hdfs_path+'/movies.csv',header='true',inferSchema = 'true')
df_movies_filter = spark.read.format("csv").load(hdfs_path+'/movies_filtered.csv',header='true',inferSchema = 'true')

df_umr = df_ratings['userId','movieId','rating']

movie_count = np.sqrt(df_movies.count())

#Impute df_umr
df_um_cross_join = df_umr.select('userId').distinct().crossJoin(df_umr.select('movieId').distinct())
df_umr = df_umr.join(df_um_cross_join,['movieId','userId'],'right').na.fill(0)


df_m_mean = df_umr.select('movieId','rating').groupBy('movieId').avg('rating')
df_m_mean = df_m_mean.withColumnRenamed('avg(rating)','rating_avg')


df_join_m_mean_umr = df_umr.join(df_m_mean, on='movieId')
df_join_m_mean_umr_normalized = df_join_m_mean_umr.withColumn('delta', df_join_m_mean_umr['rating']- df_join_m_mean_umr['rating_avg'])

df_join_m_mean_umr_normalized = df_join_m_mean_umr_normalized.withColumn('delta', df_join_m_mean_umr_normalized['delta'] / movie_count)

spark.conf.set("spark.sql.pivotMaxValues", 60000)
df_bridge = df_movies_filter.select("movieId").distinct()
w = Window().orderBy(lit('A'))

df_bridge = df_bridge.withColumn("row_num", row_number().over(w)).cache()
df_join_m_mean_umr_normalized = df_join_m_mean_umr_normalized.join(df_bridge,'movieId','inner').cache()

df_pivot_mu = df_join_m_mean_umr_normalized.groupBy("movieId").pivot("userId").agg(F.first("delta")).cache()
df_pivot_um = df_join_m_mean_umr_normalized.groupBy("userId").pivot("movieId").agg(F.first("delta")).cache()
um_col = df_pivot_um.columns[1:]    #This gives movieIds
mu_col = df_pivot_mu.columns[1:]      #This gives userIds

independent_columns = mu_col
	
		
aat_data = ( df_pivot_mu.rdd
    .flatMap(matrix_multiply)
    .reduceByKey(lambda a, b: a + b)
    .collect()
  )
  
lst = []
lst_inner = []
for i in range(0,len(aat_data)):
    if(len(lst_inner) < len(independent_columns)):
        lst_inner.append(aat_data[i][1])
    else:
        lst.append(lst_inner)
        lst_inner = [aat_data[i][1]]

lst.append(lst_inner)
aat = np.array(lst)

independent_columns = um_col
ata_data = ( df_pivot_um.rdd
    .flatMap(matrix_multiply)
    .reduceByKey(lambda a, b: a + b)
    .collect()
  )
  
lst = []
lst_inner = []
for i in range(0,len(ata_data)):
    if(len(lst_inner) < len(independent_columns)):
        lst_inner.append(ata_data[i][1])
    else:
        lst.append(lst_inner)
        lst_inner = [ata_data[i][1]]

lst.append(lst_inner)
ata = np.array(lst)


eighVals_aat, eighVecs_aat = np.linalg.eigh(aat)
eighVals_ata, eighVecs_ata = np.linalg.eigh(ata)


idx_aat = eighVals_aat.argsort()[::-1]
eighVals_aat = eighVals_aat[idx_aat]
eighVecs_aat = eighVecs_aat[:,idx_aat]

idx_ata = eighVals_ata.argsort()[::-1]
eighVals_ata = eighVals_ata[idx_ata]
eighVecs_ata = eighVecs_ata[:,idx_ata]

U = eighVals_aat

V = eighVecs_ata.T

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