# This file is part of Mooney computational modeling project.
#
# Mooney computational modeling project is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Mooney computational modeling project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Mooney computational modeling project. If not, see <https://www.gnu.org/licenses/>.


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