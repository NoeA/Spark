import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

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

avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")


######### Aggregating ########

# Find the shortest flight from PDX in terms of distance
# flights.filter(flights.origin == 'PDX').groupBy().min('distance').show()

# Find the longest flight from SEA in terms of duration
# flights.filter(flights.origin == 'SEA').groupBy().max('air_time').show()

# Average duration of Delta flights
# flights.filter(flights.carrier == 'DL').filter(flights.origin == 'SEA').groupBy().avg('air_time').show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum('duration_hrs').show()

############  Grouping and Aggregating ################

# Group by tailnum
by_plane = flights.groupBy("tailnum")

# Number of flights each plane made
by_plane.count()  # Add .show() to see


# Group by origin
by_origin = flights.groupBy("origin")

# Average duration of flights from PDX and SEA
by_origin.avg("air_time").show()


# Group by month and dest
by_month_dest = flights.groupBy("month", "dest")

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation
by_month_dest.agg(F.stddev("dep_delay")).show()


######## Joining ##########

airports_file_path = "/Users/sez/Spark/airports.csv"

# Read the data
airports_df = spark.read.csv(flights_file_path,header=True)

# Add airports_df to the catalog
airports_df.createOrReplaceTempView('airports')

# print(spark.catalog.listTables())

# Create the DataFrame airports
airports = spark.table("airports")

# Examine the data
print(airports.show())

# Rename the faa column
airports = airports.withColumnRenamed("faa", "dest")

# Join the DataFrames
flights_with_airports = flights.join(airports, on="dest", how="leftouter")

# Examine the data
print(flights_with_airports.show())
