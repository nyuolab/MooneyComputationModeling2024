import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from nilearn.maskers import MultiNiftiMasker

eps = 1e-6
n_components = 4

masker = MultiNiftiMasker(smoothing_fwhm=2.0).fit(snakemake.input.error)
error_masked = masker.transform(snakemake.input.error)
error_masked = np.concatenate(error_masked, axis=0)

# Remove almost all zero voxels
all_zero = (error_masked == 0).all(axis=0)
error_masked = error_masked[:, ~all_zero]

# Regularize the components, but not coefficients
learner = NMF(n_components=n_components, max_iter=1000)
coef = learner.fit_transform(error_masked)

for component_idx, individual_component in enumerate(learner.components_):
    new_data = np.zeros(all_zero.shape)
    new_data[~all_zero] = individual_component
    individual_component = new_data

    individual_component = masker.inverse_transform(individual_component)
    individual_component.to_filename(snakemake.output.components[component_idx])

with open(snakemake.output.learner, "wb") as f:
    pickle.dump(learner, f)

# save the coefficients and data
np.save(snakemake.output.coefficients_raw, coef)
np.save(snakemake.output.data, error_masked)

counter = 0
n_imgs = 33
subjects = snakemake.config["fmri_subjects"]
container = {"subject": [], "component_idx": [], "component_value": [], "image_index": []}
for subject_idx, subject in enumerate(subjects):
    for image_idx in range(n_imgs):
        for component_idx in range(n_components):
            container["component_value"].append(coef[counter, component_idx])
            container["subject"].append(subject)
            container["component_idx"].append(component_idx)
            container["image_index"].append(image_idx)
        counter += 1

# Finished
assert counter == len(coef)

pd.DataFrame(container).to_csv(snakemake.output.coefficients, index=False)
