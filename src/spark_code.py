#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark

from pandas import Series, DataFrame
import pandas as pd

findspark.init(r"C:\Users\monda\Desktop\Spark\spark-3.0.0-bin-hadoop2.7")


# In[2]:


import pyspark
from pyspark.sql import SparkSession


# In[3]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

#create session
appName = "Clustering in Customers"
spark = SparkSession     .builder     .appName(appName)     .config("config.option", "v.1.1")     .getOrCreate()


# In[5]:


principal_df = spark.read.csv(
    'input/train_new.csv', inferSchema=True, header=True)
principal_df.show(5)


# In[6]:


from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Word2Vec

tokenizer = Tokenizer(inputCol="ingredients_string", outputCol="words")
finalData = tokenizer.transform(principal_df)


hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(finalData)

idf = IDF(inputCol="rawFeatures", outputCol="ingredients")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("id", "ingredients", "cuisine").show()


# In[7]:


#define assembler

assembler = VectorAssembler(inputCols = [
    "id", "ingredients"], 
                            outputCol="features")
data = assembler.transform(rescaledData).select('cuisine', 'features')
data.show(truncate = False, n=3)


# In[8]:


kmeans = KMeans(
    featuresCol=assembler.getOutputCol(), 
    predictionCol="cluster", k=5)
model = kmeans.fit(data)
print ("Model is successfully trained!")


# In[9]:


centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[10]:


prediction = model.transform(data)
prediction.groupBy("cluster").count().orderBy("cluster").show()
prediction.select('Cuisine', 'cluster').show(10)


# In[11]:


prediction.show(10)


# In[ ]:




