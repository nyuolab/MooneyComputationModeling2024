import numpy as np
from nilearn.maskers import MultiNiftiMasker

eps = 1e-6
masker = MultiNiftiMasker(smoothing_fwhm=5)
gt, baseline, normal = masker.fit_transform([
    snakemake.input.groundtruth,
    snakemake.input.baseline_predictions,
    snakemake.input.normal_predictions,
])

indices = {
    "pre": lambda x: x[:33],
    "post": lambda x: x[33:66],
    "gray": lambda x: x[66:],
}

gt = indices[snakemake.wildcards.brain_phase](gt)
norm_diff = np.abs(normal - gt)
baseline_diff = np.abs(baseline - gt)

# Difference
perf_diff = np.clip((baseline_diff + 1) / (norm_diff + 1) - 1, 0, None)
masker.inverse_transform(perf_diff).to_filename(snakemake.output.error_ratio)