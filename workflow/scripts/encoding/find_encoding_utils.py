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


import pickle
import numpy as np

from nilearn.image import load_img, math_img
from nilearn.maskers import NiftiMasker

# First load the data and find the mask
mask = math_img("~(np.isnan(m).any(axis=-1))", m=snakemake.input.betas)
masker = NiftiMasker(mask)
Y = masker.fit_transform(load_img(snakemake.input.betas))

# Save masked data
np.save(snakemake.output.masked_data, Y)

# Save mask
mask.to_filename(snakemake.output.mask)

# Save the masker
with open(snakemake.output.masker, "wb") as f:
    pickle.dump(masker, f)