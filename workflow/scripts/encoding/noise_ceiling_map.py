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
from nilearn.maskers import NiftiMasker
from sklearn.model_selection import LeaveOneOut

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import batched_pearsonr


# Create a non-nan, non-constant mask
masker = NiftiMasker(smoothing_fwhw=5.0).fit(snakemake.input.maps[0])

# Load the data in [subject, condition, voxels]
masked_data = [masker.transform(d) for d in snakemake.input.maps]
masked_data = np.stack(masked_data, axis=0)

# Find the phase
if snakemake.wildcards.phase == "pre":
    masked_data = masked_data[:, :33]
elif snakemake.wildcards.phase == "gray":
    masked_data = masked_data[:, 66:]
elif snakemake.wildcards.phase == "post":
    masked_data = masked_data[:, 33:66]
else:
    raise NotImplementedError

# Calculate average of all data for upper ceiling
all_trained = masked_data.mean(axis=0)

# Leave one out
rzs_lower = np.zeros(masked_data.shape[2])
rzs_upper = np.zeros(masked_data.shape[2])
for train_idx, test_idx in LeaveOneOut().split(masked_data):
    train_data, test_data = masked_data[train_idx], masked_data[test_idx].squeeze(0)

    # Calculate lower ceiling by averaging train data and evaluating on test data
    train_data = train_data.mean(axis=0)

    # Take the correlation along conditions
    rzs_lower += np.arctanh(batched_pearsonr(train_data, test_data))

    # Calculate upper ceiling by averaging all data
    rzs_upper += np.arctanh(batched_pearsonr(all_trained, test_data))

# Average the results and convert back to r
rzs_lower /= masked_data.shape[0]
rzs_upper /= masked_data.shape[0]

rzs_lower = np.tanh(rzs_lower)
rzs_upper = np.tanh(rzs_upper)

# Save the results
masker.inverse_transform(rzs_lower).to_filename(snakemake.output.lower)
masker.inverse_transform(rzs_upper).to_filename(snakemake.output.upper)