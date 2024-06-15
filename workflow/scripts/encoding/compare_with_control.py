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


import pandas as pd
from nilearn.image import math_img, threshold_img
from nilearn.reporting import get_clusters_table
from nilearn.glm.second_level import non_parametric_inference
dmat = pd.DataFrame({
    "difference": [1] * len(snakemake.input.pos_maps) + [-1] * len(snakemake.input.neg_maps),
    "subject": list(range(len(snakemake.input.pos_maps))) * 2
})
dmat = pd.get_dummies(dmat, columns=["subject"], drop_first=False)
outputs = non_parametric_inference(
    design_matrix=dmat,
    second_level_input=snakemake.input.pos_maps + snakemake.input.neg_maps,
    second_level_contrast="difference",
    model_intercept=True,
    smoothing_fwhm=5.0,
    n_perm=10000,
    tfce=True,
    n_jobs=8,
    two_sided_test=False,
    verbose=100
)
outputs["t"].to_filename(snakemake.output.tmap)
outputs["logp_max_tfce"].to_filename(snakemake.output.pmap)
math_img(
    "np.where(p>1.3, score1 - score2, 0)",
    p=outputs["logp_max_tfce"],
    score1=snakemake.input.avg_map,
    score2=snakemake.input.avg_map_control
).to_filename(snakemake.output.thresholded_score)
threshold_img(outputs["logp_max_tfce"], 1.3).to_filename(snakemake.output.thresholded_pmap)
get_clusters_table(outputs["logp_max_tfce"], 1.3, 500).to_csv(snakemake.output.cluster_record)
