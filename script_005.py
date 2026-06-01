# Distributed ambient noise cross-correlation for surface wave tomography using PySpark's Cartesian join across all Marmara station pairs

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
import numpy as np

@udf(returnType=ArrayType(DoubleType()))
def cross_correlate(sig1, sig2):
    """Normalized cross-correlation for ambient noise interferometry."""
    a, b = np.array(sig1), np.array(sig2)
    a = a / (np.std(a) + 1e-10)
    b = b / (np.std(b) + 1e-10)
    xcorr = np.correlate(a, b, mode='full')
    return (xcorr / len(a)).tolist()

stations_df = spark.read.format("delta").load("/mnt/silver/daily_noise")
pairs_df = stations_df.alias("s1").crossJoin(stations_df.alias("s2")) \
    .filter(col("s1.station_id") < col("s2.station_id"))

xcorr_df = pairs_df.withColumn(
    "xcorr",
    cross_correlate(col("s1.waveform"), col("s2.waveform"))
)
xcorr_df.write.format("delta").mode("append") \
    .save("/mnt/gold/noise_xcorr_marmara")
print("Cross-correlation complete for all Marmara station pairs.")
