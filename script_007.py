# End-to-end EEW alert pipeline using Spark Structured Streaming, monitoring processing latency to maintain the ``Golden Seconds'' constraint

from pyspark.sql.functions import current_timestamp, unix_timestamp, expr

alert_stream = picked_df \
    .filter(col("p_confidence") > 0.85) \
    .withColumn("ingestion_time", current_timestamp()) \
    .withColumn("latency_ms",
        (unix_timestamp(col("ingestion_time")) -
         unix_timestamp(col("timestamp"))) * 1000)

# Flag records exceeding 100 ms EEW latency budget
alert_stream = alert_stream \
    .withColumn("latency_ok", col("latency_ms") < 100)

# Write alerts to dashboard and trigger automated responses
alert_query = alert_stream.writeStream \
    .outputMode("append") \
    .format("delta") \
    .option("checkpointLocation", "/mnt/checkpoints/eew_alerts") \
    .trigger(processingTime="1 s") \
    .start("/mnt/gold/eew_alerts")

# Log latency SLA compliance
sla_df = spark.read.format("delta").load("/mnt/gold/eew_alerts")
sla_pct = sla_df.filter(col("latency_ok")).count() / sla_df.count() * 100
print(f"EEW latency SLA compliance: {sla_pct:.1f}%")
alert_query.awaitTermination()
