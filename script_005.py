# =============================================================================
# Artificial Intelligence and Big Data Analytics for Seismic Hazard Assessment — Marmara Region, Türkiye
#
# Authors: Polina Lemenkova and Abdullah Can Zülfikar
# ORCID:   https://orcid.org/0000-0002-5759-1089
# Paper:   Lemenkova, P.; Zülfikar, A.C. (2026). Artificial Intelligence and Big Data Analytics for Seismic Hazard Assessment: Methodological Advances and Computational Frameworks for the Marmara Region, Türkiye. Data, 11(6), 131. ISSN 2306-5729.
# DOI:     https://doi.org/10.3390/data11060131
# License: MIT (see LICENSE)
# =============================================================================
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
