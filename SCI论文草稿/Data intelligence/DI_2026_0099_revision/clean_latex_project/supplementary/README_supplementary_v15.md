# Supplementary files for DI-2026-0099 v15

This folder contains machine-readable versions of the manuscript tables, the expected figure-file manifest, the parameter/provenance manifest, and metric-recalculation templates.

Important: the CSV files reproduce the summary values stated in the revised manuscript or provide templates for trial-level audit. The raw robot logs and high-resolution figure files were not available in this conversation. Before final journal upload, replace or complement the templates with the true raw logs, high-resolution figures, policy checkpoints, URDF/model versions, and any proprietary files that can be shared under the journal policy.

Files:
- `figure_file_manifest_v15.md/csv`: expected figure filenames, labels, captions, and source-reference counts.
- `table_data/actuator_conversion_v15.csv`: values used in Table 1.
- `table_data/joint_identification_v15.csv`: values used in Table 2.
- `table_data/domain_randomization_ranges_v15.csv`: values used in Table 3.
- `table_data/ablation_matrix_v15.csv`: values used in Table 5.
- `table_data/hardware_statistics_v15.csv`: values used in Table 6.
- `table_data/hardware_reliability_v15.csv`: values used in Table 7.
- `table_data/trace_ranges_main_v15.csv`: values used in Table 8.
- `table_data/parameter_provenance_v15.csv`: labels quantities as identified, manufacturer/URDF-derived, fixed controller setting, or evaluation-derived.
- `table_data/trial_level_metrics_template_v15.csv`: header template for independent trial-level metrics.
- `scripts/metric_recalculation_template_v15.py`: template for aggregating trial-level metrics.
- `parameter_manifest_v15.yaml`: actuator constants, simulator synchronization settings, and RTI-bound conventions.
- `raw_logs/README_raw_logs.md`: recommended raw-log structure.
