# =============================================================================
# Artificial Intelligence and Big Data Analytics for Seismic Hazard Assessment — Marmara Region, Türkiye
#
# Authors: Polina Lemenkova and Abdullah Can Zülfikar
# ORCID:   https://orcid.org/0000-0002-5759-1089
# Paper:   Lemenkova, P.; Zülfikar, A.C. (2026). Artificial Intelligence and Big Data Analytics for Seismic Hazard Assessment: Methodological Advances and Computational Frameworks for the Marmara Region, Türkiye. Data, 11(6), 131. ISSN 2306-5729.
# DOI:     https://doi.org/10.3390/data11060131
# License: MIT (see LICENSE)
# =============================================================================
# Gutenberg--Richter b-value estimation and exceedance probability computation from the AFAD catalog using PySpark SQL and NumPy

import numpy as np
from pyspark.sql.functions import count, col, log10 as spark_log10

# Gutenberg-Richter: log10(N) = a - b*M
catalog = spark.read.format("delta").load("/mnt/gold/afad_catalog")

mag_counts = catalog \
    .filter(col("magnitude") >= 2.0) \
    .groupBy("magnitude_bin") \
    .agg(count("*").alias("N")) \
    .withColumn("log_N", spark_log10(col("N")))

# Collect for numpy regression
data = mag_counts.orderBy("magnitude_bin").toPandas()
M = data["magnitude_bin"].values
logN = data["log_N"].values
b, a = np.polyfit(M, logN, 1)   # b-value (negative slope)
print(f"Gutenberg-Richter: a={a:.2f}, b={-b:.3f}")

# Exceedance probability for M >= 7.0 in 50 years
# P(exceedance) = 1 - exp(-lambda * T)
lambda_annual = 10 ** (a + b * 7.0)  # events per year
T = 50.0
prob_50yr = 1 - np.exp(-lambda_annual * T)
print(f"P(M>=7.0 in 50 yr) = {prob_50yr:.3f}")

# Uncertainty on b-value (Aki 1965 estimator)
N_events = int(catalog.filter(col("magnitude") >= 2.0).count())
M_mean   = float(catalog.filter(col("magnitude") >= 2.0)
                 .agg({"magnitude": "mean"}).collect()[0][0])
M_min    = 2.0
sigma_b  = b / np.sqrt(N_events)
print(f"b-value uncertainty (1-sigma): {sigma_b:.4f}")
