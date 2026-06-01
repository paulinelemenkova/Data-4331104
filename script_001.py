# PySpark ingestion and windowing of continuous seismic waveform data from AFAD-compatible MiniSEED streams

from pyspark.sql import SparkSession
from pyspark.sql.functions import window, col, from_json
from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType

spark = SparkSession.builder \
    .appName("MarmaraSeismicIngest") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

schema = StructType() \
    .add("station_id", StringType()) \
    .add("timestamp", TimestampType()) \
    .add("channel", StringType()) \
    .add("amplitude", DoubleType()) \
    .add("sampling_rate", DoubleType())

# Read streaming data from Apache Kafka (Bronze Layer)
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka-broker:9092") \
    .option("subscribe", "marmara.seismic.raw") \
    .load() \
    .selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .select("data.*")

# Apply 10-s tumbling windows for real-time analysis
windowed = raw_stream \
    .groupBy(window(col("timestamp"), "10 s"), col("station_id")) \
    .agg({"amplitude": "max"}) \
    .withColumnRenamed("max(amplitude)", "peak_amplitude")

query = windowed.writeStream \
    .outputMode("append") \
    .format("delta") \
    .option("checkpointLocation", "/mnt/checkpoints/seismic_ingest") \
    .start("/mnt/bronze/seismic_windows")
query.awaitTermination()
