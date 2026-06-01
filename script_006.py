# Automated P- and S-wave phase picking using PhaseNet integrated into a PySpark Gold-layer pipeline with pick uncertainty estimation

import torch
import numpy as np
from pyspark.sql.functions import struct, col

# Load pre-trained PhaseNet model (seisbench interface)
# seisbench.models.PhaseNet.from_pretrained("stead")
def phasenet_pick(waveform_list):
    """
    Apply PhaseNet to 3-component waveform and return
    (p_pick_sample, s_pick_sample, p_prob, s_prob).
    """
    wf = np.array(waveform_list).reshape(3, -1)
    # Normalize per channel
    wf = (wf - wf.mean(axis=1, keepdims=True)) / (wf.std(axis=1, keepdims=True) + 1e-8)
    with torch.no_grad():
        probs = model(torch.tensor(wf, dtype=torch.float32).unsqueeze(0))
    p_prob = probs[0, 1, :].numpy()
    s_prob = probs[0, 2, :].numpy()
    p_pick = int(np.argmax(p_prob))
    s_pick = int(np.argmax(s_prob))
    return (p_pick, s_pick, float(p_prob[p_pick]), float(s_prob[s_pick]))

schema_pick = "struct<p_pick:int, s_pick:int, p_prob:double, s_prob:double>"
phasenet_udf = udf(phasenet_pick, schema_pick)

picked_df = filtered_df \
    .withColumn("picks", phasenet_udf(col("filtered_signal"))) \
    .select("station_id", "timestamp",
            col("picks.p_pick").alias("p_sample"),
            col("picks.s_pick").alias("s_sample"),
            col("picks.p_prob").alias("p_confidence"),
            col("picks.s_prob").alias("s_confidence"))

# Uncertainty: sigma_t = (1 - confidence) * dt, dt = 1/sampling_rate
picked_df = picked_df.withColumn(
    "p_uncertainty_s", (1.0 - col("p_confidence")) / 100.0)
picked_df.write.format("delta").mode("append") \
    .save("/mnt/gold/phase_picks")
