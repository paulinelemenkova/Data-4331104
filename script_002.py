# =============================================================================
# Artificial Intelligence and Big Data Analytics for Seismic Hazard Assessment — Marmara Region, Türkiye
#
# Authors: Polina Lemenkova and Abdullah Can Zülfikar
# ORCID:   https://orcid.org/0000-0002-5759-1089
# Paper:   Lemenkova, P.; Zülfikar, A.C. (2026). Artificial Intelligence and Big Data Analytics for Seismic Hazard Assessment: Methodological Advances and Computational Frameworks for the Marmara Region, Türkiye. Data, 11(6), 131. ISSN 2306-5729.
# DOI:     https://doi.org/10.3390/data11060131
# License: MIT (see LICENSE)
# =============================================================================
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
