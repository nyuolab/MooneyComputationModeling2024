import numpy as np
import pandas as pd

from nilearn.maskers import MultiNiftiMasker
from skdim.id import FisherS

from sklearn.utils import resample

eps = 1e-6

masker = MultiNiftiMasker(smoothing_fwhm=5.0).fit(snakemake.input.error)
error_masked = masker.transform(snakemake.input.error)
error_masked = np.concatenate(error_masked, axis=0)

# Remove almost all zero voxels
all_zero = (error_masked == 0).all(axis=0)
error_masked = error_masked[:, ~all_zero]

estimator = FisherS()

if snakemake.params.boot:
    error_masked = resample(error_masked)

pd.DataFrame({
    "dimensions": [estimator.fit_transform(error_masked)],
}).to_csv(snakemake.output.record, index=False)