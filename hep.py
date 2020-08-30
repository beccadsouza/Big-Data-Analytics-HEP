import findspark
findspark.init()
import pyspark
import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('hep-analysis').getOrCreate()
data_location = 'HIGGS_subsampled_20k.csv'
df = spark.read.load(data_location,format="csv", sep=",", inferSchema="true", header="true")
(training, test) = df.randomSplit([0.7, 0.3])
print(training.count(), test.count())
print(training.columns)
print(training.describe(training.columns[1]).show())

from pyspark.sql.functions import *
from pyspark.ml.linalg import DenseVector

training_dense = training.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
training_dense = spark.createDataFrame(training_dense, ["label", "features"])

test_dense = test.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
test_dense = spark.createDataFrame(test_dense, ["label", "features"])

from pyspark.ml.feature import StandardScaler
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled", withMean=True)

scaler = standardScaler.fit(training_dense)
scaled_training = scaler.transform(training_dense)
print(scaled_training.head(2))


scaled_test = scaler.transform(test_dense)
print(scaled_test.head(2))


from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib import linalg as mllib_linalg
from pyspark.ml import linalg as ml_linalg

def as_old(v):
    if isinstance(v, ml_linalg.SparseVector):
        return mllib_linalg.SparseVector(v.size, v.indices, v.values)
    if isinstance(v, ml_linalg.DenseVector):
        return mllib_linalg.DenseVector(v.values)
    raise ValueError("Unsupported type {0}".format(type(v)))
    

scaled_labelPoint_train = scaled_training.rdd.map(lambda row: LabeledPoint(row.label, as_old(row.features_scaled)))
print(scaled_labelPoint_train.take(2))

labelPoint_train = training_dense.rdd.map(lambda row: LabeledPoint(row.label, as_old(row.features)))
print(labelPoint_train.take(2))



print('Learned classification GBT model:')
import time
train_start = time.time()
GBTmodel = GradientBoostedTrees.trainClassifier(labelPoint_train,categoricalFeaturesInfo={}, numIterations=30)
train_end = time.time()
print(f'Time elapsed training model: {train_end - train_start} seconds')

# Evaluate model on test instances and compute test error
predictions = GBTmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)


testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() /float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))




from pyspark.mllib.tree import RandomForest, RandomForestModel

print('Learned classification RF model:')
train_start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)
train_end = time.time()
print(f'Time elapsed training model: {train_end - train_start} seconds')

predictions = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


spark.stop()
