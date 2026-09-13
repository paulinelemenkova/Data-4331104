# Artificial Intelligence and Big Data Analytics for Seismic Hazard Assessment — Marmara Region, Türkiye

This repository contains the computational framework accompanying the peer-reviewed article below. It implements a distributed big-data and artificial-intelligence pipeline, built on Apache Spark (PySpark), for seismic hazard assessment of the Marmara Region, Türkiye. The workflow follows a medallion (bronze/silver/gold) data-lake architecture and processes continuous seismic waveforms together with the AFAD (Disaster and Emergency Management Presidency of Türkiye) earthquake catalog, covering signal conditioning, event detection, deep-learning phase picking, magnitude estimation, earthquake early warning and probabilistic seismic-hazard statistics.

## Associated publication

Lemenkova, P.; Zülfikar, A.C. (2026). Artificial Intelligence and Big Data Analytics for Seismic Hazard Assessment: Methodological Advances and Computational Frameworks for the Marmara Region, Türkiye. *Data*, 11(6), 131. ISSN 2306-5729.

Indexed in Scopus and Web of Science.

## Links

- Published paper (DOI): https://doi.org/10.3390/data11060131
- Publisher (MDPI): https://www.mdpi.com/2306-5729/11/6/131
- Archived (Zenodo): https://doi.org/10.5281/zenodo.20513886
- Preprint (HAL): https://hal.science/hal-05641902v1
- LifeScience.net: https://www.lifescience.net/publications/2043658/artificial-intelligence-and-big-data-analytics-for/
- Author ORCID (Polina Lemenkova): https://orcid.org/0000-0002-5759-1089

## Computational framework

Eight PySpark scripts implement the end-to-end pipeline:

- `script_001.py` — PySpark ingestion and windowing of continuous seismic waveform data from AFAD-compatible MiniSEED streams
- `script_002.py` — Butterworth bandpass filtering of seismic waveforms (SciPy/ObsPy) inside a PySpark UDF to remove cultural noise
- `script_003.py` — Distributed STA/LTA (short-term/long-term average) event detection across the AFAD station network via PySpark pandas UDFs
- `script_004.py` — Local magnitude (Ml) estimation from peak amplitude with statistical validation (RMSE, precision, recall) against the AFAD catalog
- `script_005.py` — Distributed ambient-noise cross-correlation for surface-wave tomography (PySpark Cartesian join over Marmara station pairs)
- `script_006.py` — Automated P- and S-wave phase picking with PhaseNet (PyTorch/SeisBench) in a PySpark Gold-layer pipeline, with pick-uncertainty estimation
- `script_007.py` — End-to-end Earthquake Early Warning (EEW) alert pipeline with Spark Structured Streaming, monitoring latency for the 'Golden Seconds' constraint
- `script_008.py` — Gutenberg-Richter b-value estimation and exceedance-probability computation from the AFAD catalog (PySpark SQL + NumPy)

## Data

The scripts consume continuous seismic waveforms (MiniSEED) and the AFAD earthquake catalog, organised as Delta Lake tables across bronze/silver/gold layers. The raw seismological data are distributed by AFAD and are not bundled here; the `/mnt/bronze`, `/mnt/silver` and `/mnt/gold` paths refer to a Spark + Delta Lake environment and should be adapted to your own cluster.

## Requirements

Python 3 with Apache Spark (PySpark) and Delta Lake; NumPy, SciPy and pandas; PyTorch with SeisBench/PhaseNet for deep-learning phase picking; and ObsPy for seismological I/O. A Spark cluster is assumed for distributed execution.

## Authors

Polina Lemenkova and Abdullah Can Zülfikar

Polina Lemenkova — ORCID: https://orcid.org/0000-0002-5759-1089

## License

MIT — see the LICENSE file (Copyright Polina Lemenkova and Abdullah Can Zülfikar).
