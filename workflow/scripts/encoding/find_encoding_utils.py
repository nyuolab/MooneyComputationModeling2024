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