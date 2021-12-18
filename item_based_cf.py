import sys
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, SparkSession
from pyspark.sql.functions import lit,col,sum,avg,max,count,when
from functools import reduce
from math import sqrt as sqrt

if __name__ == "__main__":
    if len(sys.argv) !=3:
        print("Usage: spark-submit item_based_cf.py <movieID> </user/<username>/dataset>", file=sys.stderr)
        sys.exit(-1)

spark = (SparkSession
        .builder
        .appName("Recommender System using Spark")
        .getOrCreate())

sc = spark.sparkContext
sc.setLogLevel('WARN')

scoreThreshold = 0.90
coOccurenceThreshold = 50.0
top_n = 10
movieID = int(sys.argv[1])
hdfs_path = str(sys.argv[2])
df = spark.read.format("csv").load(hdfs_path+'/ratings.csv',header='true',inferSchema = 'true')
df_movies = spark.read.format("csv").load(hdfs_path+'/movies.csv',header='true',inferSchema = 'true')
df2 = df['userId','movieId','rating']

# Find every movie pair rated by same user, we are achieving this by using a “self-join” operation on DF
joinedData_uniq = df2.alias("left").join(df2.alias("right"),(col("left.userId") == col("right.userId")) & (col("left.movieId") != col("right.movieId")) & (col("left.movieId") < col("right.movieId"))).select("left.userId",col("left.movieId").alias("movieId1"),col("left.rating").alias("rating1"),col("right.movieId").alias("movieId2"),col("right.rating").alias("rating2"))

# cosine similarity on the relevant columns of dataframe - rating1 and rating2
joinedData_uniq=joinedData_uniq.select("*",F.pow(col("rating1"),2).alias("rating1_sq"),F.pow(col("rating2"),2).alias("rating2_sq"),(col("rating1")*col("rating2")).alias("numerator"))
joinedData_uniq=joinedData_uniq.groupBy("movieId1","movieId2").agg(sum("numerator").alias("numerator"),sum("rating1_sq").alias("rating1_sq"),sum("rating2_sq").alias("rating2_sq"),count(F.lit(1)).alias("num_pairs"))
joinedData_uniq=joinedData_uniq.select("*",F.pow(col("rating1_sq"),1/2).alias("rating1_sqrt"),F.pow(col("rating2_sq"),1/2).alias("rating2_sqrt"))
joinedData_uniq=joinedData_uniq.select("*",(col("rating1_sqrt")*col("rating2_sqrt")).alias("denominator"))
joinedData_uniq=joinedData_uniq.select("*",(col("numerator")/col("denominator")).alias("cosine_sim"))

# cache the dataframe that has cosine similarity score. This will help in faster computations of operations on the dataframe
joinedData_uniq = joinedData_uniq.cache()

# filter out the movie IDs that are greater than our scoreThreshold and coOccurenceThreshold values
matched_data = joinedData_uniq.filter(((col("movieId2") == movieID) | (col("movieId1") == movieID)) & (col("num_pairs") > coOccurenceThreshold) & (col("cosine_sim") > scoreThreshold)).sort(col("cosine_sim").desc())

# using the movie IDs find the movie titles by joining matched_data and df_movies on movieID
final_recc = matched_data.withColumn("movieId", when(matched_data.movieId1 != movieID, matched_data.movieId1).otherwise(matched_data.movieId2)).select("movieId","cosine_sim","num_pairs").join(df_movies.select("movieId","title","genres"),"movieId","inner").cache()

# print the input movie title
print("----- Input Movie -----")
df_movies.filter(col('movieId')== movieID).show(truncate = False)

# print recommended movie titles
print("---- Similar Movies -----")

final_recc.show(top_n,truncate=False)
