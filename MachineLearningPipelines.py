import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

sc = pyspark.SparkContext()

# Create my spark
spark = SparkSession.builder.getOrCreate()

flights_file_path = "/Users/sez/Spark/flights_small.csv"
planes_file_path = "/Users/sez/Spark/planes.csv"

# Read the data
flights_df = spark.read.csv(flights_file_path,header=True)
planes_df = spark.read.csv(planes_file_path,header=True)

# Add DataFrames to the catalog
flights_df.createOrReplaceTempView('flights')
planes_df.createOrReplaceTempView('planes')

# Create the DataFrames
flights = spark.table("flights")
planes = spark.table("planes")

############ Join the DataFrames ###########

# Rename year column to avoid duplicate column names
planes = planes.withColumnRenamed('year','plane_year')

# Join the DataFrames using the tailnum column as the key.
model_data = flights.join(planes, on='tailnum', how="leftouter")

######## Convert all the appropriate columns string to integer #########
# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.\
arr_delay.\
cast('integer'))

model_data = model_data.withColumn("air_time", model_data.air_time.\
cast('integer'))

model_data = model_data.withColumn("month", model_data.month.\
cast('integer'))

model_data = model_data.withColumn("plane_year",model_data.plane_year.\
cast('integer'))

# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)

# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.\
cast('integer'))

# Remove missing values
model_data = model_data.\
    filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")

# Create a StringIndexer for carrier column
carr_indexer = StringIndexer(inputCol='carrier', outputCol='carrier_index')

# Create a OneHotEncoder for carrier column
carr_encoder = OneHotEncoder(inputCol='carrier_index', outputCol='carrier_fact')

# Create a StringIndexer for Destination column
dest_indexer = StringIndexer(inputCol='dest',outputCol='dest_index')

# Create a OneHotEncoder for Destination column
dest_encoder = OneHotEncoder(inputCol='dest_index',outputCol='dest_fact')

""" The last step in the Pipeline is to combine all of the columns containing our features into a single column. """

######## Assemble a vector  #######

# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])

###### Transform the data #########

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])  # split the data into two pieces, training with 60% of the data, and test with 40% of the data