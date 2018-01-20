import pyspark
from pyspark.sql import SparkSession

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