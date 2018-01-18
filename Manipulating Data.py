import pyspark
from pyspark.sql import SparkSession

sc = pyspark.SparkContext()

# Create my spark
spark = SparkSession.builder.getOrCreate()

print(spark)
print(sc)
print(sc.version)

########## Selecting ##############

flights_file_path = "/Users/sez/Spark/flights_small.csv"

# Read the data
flights_df = spark.read.csv(flights_file_path,header=True)

# Add flights_df to the catalog
flights_df.createOrReplaceTempView('flights')

print(spark.catalog.listTables())

# Create the DataFrame flights
flights = spark.table("flights")

# Filter flights with a SQL string
long_flights1 = flights.filter("distance > 1000")

# Filter flights with a boolean column
# long_flights2 = flights.filter(flights.distance > 1000)

# Examine the data to check they're equal
print(long_flights1.show())
# print(long_flights2.show())

