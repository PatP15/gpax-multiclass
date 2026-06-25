# Calibration study (GPP vs LPE) -- scripts behind docs/CALIBRATION_STUDY.md
Run from repo root in the gpax-multiclass conda env: python experiments/calibration_study/<script>.py
- calib_analysis.py : ECE/Brier, AnnoMI GPP vs LPE (committed annoMI embeddings)
- calib_why.py      : per-n_obs Pearson + rational-uncertainty (reads results/csv/results_jax_fuzziness_p{1,2}.csv)
- rescue_calib.py   : strength x alpha_eps sweep (no rescue)
- rescue_temp.py    : softmax temperature sweep (no rescue)
- kernel_rescue.py  : RBF lengthscale sweep + marginal-likelihood selection (RESCUE); needs results/embeddings/embeddings_M1.npy
