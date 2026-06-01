# Short-term average/long-term average (STA/LTA) event detection implemented as a distributed PySpark operation across the AFAD station network

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BooleanType
import pandas as pd

@pandas_udf(BooleanType())
def sta_lta_trigger(signal_series: pd.Series) -> pd.Series:
    """
    Detect seismic events using the STA/LTA ratio.
    Returns True where STA/LTA exceeds threshold (typically 3.0).
    """
    results = []
    for signal in signal_series:
        arr = np.array(signal)
        sta_len, lta_len, threshold = 5, 100, 3.0
        sta = np.convolve(arr**2, np.ones(sta_len)/sta_len, 'same')
        lta = np.convolve(arr**2, np.ones(lta_len)/lta_len, 'same')
        ratio = np.where(lta > 0, sta / lta, 0)
        results.append(bool(np.max(ratio) > threshold))
    return pd.Series(results)

events_df = filtered_df.withColumn(
    "event_detected", sta_lta_trigger(col("filtered_signal"))
).filter(col("event_detected") == True)

print(f"Detected {events_df.count()} candidate seismic events.")
events_df.write.format("delta").mode("append") \
    .save("/mnt/silver/detected_events")
