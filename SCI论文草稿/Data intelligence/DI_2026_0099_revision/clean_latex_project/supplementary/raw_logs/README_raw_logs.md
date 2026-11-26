# Raw log placement for DI-2026-0099 v15

The raw hardware logs were not uploaded in this conversation because of file-size constraints. Before final journal upload, place one aligned or raw log file per independent trial in this folder, or provide a controlled-access link if proprietary driver data cannot be redistributed publicly.

Recommended minimum columns: timestamp_s, joint_position_rad, joint_velocity_rad_s, desired_joint_position_rad, driver_current_A, IMU_pitch_deg, IMU_angular_velocity_deg_s, command_velocity_m_s, battery_soc_percent, safety_flag, intervention_flag, fall_flag.

The trial-level CSV in ../table_data/trial_level_metrics_template_v15.csv should be generated from these logs before public release.
