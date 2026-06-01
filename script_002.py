#Application of a Butterworth bandpass filter to seismic waveforms using ObsPy within a PySpark UDF to remove cultural noise

import numpy as np
from scipy.signal import butter, sosfilt
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

def butterworth_bandpass(signal_list, lowcut=1.0, highcut=10.0, fs=100.0, order=4):
    """Apply a Butterworth bandpass filter to isolate tectonic signal."""
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, np.array(signal_list)).tolist()

bandpass_udf = udf(butterworth_bandpass, ArrayType(DoubleType()))

silver_df = spark.read.format("delta").load("/mnt/bronze/seismic_windows")
filtered_df = silver_df.withColumn(
    "filtered_signal",
    bandpass_udf(col("raw_samples"), col("sampling_rate"))
)
filtered_df.write.format("delta") \
    .mode("overwrite") \
    .save("/mnt/silver/filtered_waveforms")
print(f"Processed {filtered_df.count()} waveform records into Silver layer.")
