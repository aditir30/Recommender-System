# Recommender System using Collaborative Filtering

## Authors
- [Aditi Raghuwanshi](https://github.com/aditir30)
- [Pratik Chaudhari](https://github.com/pratiknc)


## Introduction
The aim of this project is to implement a recommender system for movies. The system is implemented using Python that can be executed on a cluster that has Hadoop and Spark installed on it.

## Dataset Information
The dataset is selected from https://grouplens.org/datasets/movielens/ site.
Dataset size – The dataset has 3 files – ratings.csv, users.csv and movies.csv. Files have data present in the following format –

File name    | Available attributes | Project specific attributes
:----------: | :----------: | :----------:
ratings.csv | userID, movieID, rating, timestamp   | userID, movieID, rating 
movies.csv  | movieID, title, genres    | movieID, title, genres  
user.csv   | userID, gender, age, occupation, zip-code     | Not Applicable to this project[^1]  

[^1]: Since we only need movie names and their rating for collaborative filtering, we do not use users.csv file

The ratings file has approximately 100,000 ratings – each user has rated at least 20 movies.

For implementing SVD, some data analysis and filtering task had to be performed since decomposing matrices and performing eigen computations was very memory intensive.

## Run Instructions

Ensure all input csv files are present on HDFS. I have used `dataset` as the name where all input csv files are stored.

#### For *item_based_cf.py* use command

```
spark-submit item_based_cf.py <movieID> '/user/<username>/dataset'
```

Example -
```
spark-submit item_based_cf.py 50 '/user/xyz/dataset'
```


#### For *SingularValueDecomposition.py* use command

```
spark-submit --conf spark.kryoserializer.buffer.max=1024 --conf spark.driver.memory=15g SingularValueDecomposition.py <movieID> '/user/<username>/dataset'
```

Example -
```
spark-submit --conf spark.kryoserializer.buffer.max=1024 --conf spark.driver.memory=15g SingularValueDecomposition.py 2628 '/user/xyz/dataset'
```
The driver memory and buffer size can be changed according to the cluster on which it runs.

#### For running only the online component of SVD, with pre computed matrices, use command
```
spark-submit SingularValueDecomposition_online.py <movieID> '/user/<username>/dataset'
```

Example -
```
spark-submit SingularValueDecomposition_online.py 2628 '/user/xyz/dataset'
```
