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
from nilearn.image import smooth_img, load_img
from nilearn.reporting import get_clusters_table
from nilearn.glm.second_level import non_parametric_inference

# Pre smooth since nc_map is already smoothed
maps = [smooth_img(m, 5.0) for m in snakemake.input.maps]
if snakemake.params.lower:
    dm = pd.DataFrame([-1] * len(maps)+[1], columns=["intercept"])
else:
    dm = pd.DataFrame([1] * len(maps)+[-1], columns=["intercept"])

outputs = non_parametric_inference(
    second_level_input=maps + [load_img(snakemake.input.nc_map)],
    design_matrix=dm,
    model_intercept=True,
    smoothing_fwhm=None,
    n_perm=10000,
    threshold=0.05,
    n_jobs=8,
    two_sided_test=False,
    verbose=100
)
outputs["t"].to_filename(snakemake.output.tmap)
outputs["logp_max_mass"].to_filename(snakemake.output.pmap)
get_clusters_table(outputs["logp_max_mass"], 1.3, 500).to_csv(snakemake.output.cluster_record)
