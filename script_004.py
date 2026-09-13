# =============================================================================
# Artificial Intelligence and Big Data Analytics for Seismic Hazard Assessment — Marmara Region, Türkiye
#
# Authors: Polina Lemenkova and Abdullah Can Zülfikar
# ORCID:   https://orcid.org/0000-0002-5759-1089
# Paper:   Lemenkova, P.; Zülfikar, A.C. (2026). Artificial Intelligence and Big Data Analytics for Seismic Hazard Assessment: Methodological Advances and Computational Frameworks for the Marmara Region, Türkiye. Data, 11(6), 131. ISSN 2306-5729.
# DOI:     https://doi.org/10.3390/data11060131
# License: MIT (see LICENSE)
# =============================================================================
# Local magnitude estimation using peak amplitude and statistical validation (RMSE, precision, recall) against the AFAD earthquake catalog

import math
from pyspark.sql.functions import log10, sqrt, mean, stddev

# Local magnitude: Ml = log10(A) + 1.11*log10(R) + 0.00189*R - 2.09
# where A = peak amplitude (nm), R = hypocentral distance (km)
catalog_df = spark.read.format("delta").load("/mnt/gold/afad_catalog")
predicted_df = events_df \
    .withColumn("log_amp", log10(col("peak_amplitude"))) \
    .withColumn("Ml_estimated",
        col("log_amp") + 1.11 * log10(col("distance_km"))
        + 0.00189 * col("distance_km") - 2.09)

# Join with catalog for validation
joined = predicted_df.join(catalog_df, on="event_id", how="inner")
validation = joined.withColumn("sq_error",
    (col("Ml_estimated") - col("Ml_catalog"))**2)

rmse = validation.agg(sqrt(mean(col("sq_error"))).alias("RMSE")).collect()[0]["RMSE"]
print(f"Magnitude estimation RMSE: {rmse:.3f}")

# Precision / Recall for event detection (TP threshold: |dM| < 0.5)
tp = joined.filter((col("Ml_estimated") - col("Ml_catalog")).between(-0.5, 0.5)).count()
fp = predicted_df.count() - tp
fn = catalog_df.count() - tp
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
