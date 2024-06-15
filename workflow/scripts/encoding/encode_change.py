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
import pandas as pd
import numpy as np

from nilearn.maskers import NiftiMasker

from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, LeaveOneOut

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import get_model_features, batched_pearsonr

X = get_model_features(
    sequence_record=pd.read_csv(snakemake.input.sequence),
    features=np.load(snakemake.input.feat),
    phase_limit="all",
)

if len(X.shape) == 3:
    X = X.reshape(X.shape[0], -1)

# Get the Y
Y = np.load(snakemake.input.betas)

# Linear model
estimator = Pipeline([
    ("scaler", StandardScaler()),
    ("linear", KernelRidge())
])

indices = {
    "pre": lambda x: x[:33],
    "post": lambda x: x[33:66],
    "gray": lambda x: x[66:],
}

if hasattr(snakemake.wildcards, "model_phase"):
    model_phase = snakemake.wildcards.model_phase
else:
    # This is when we use baselines
    model_phase = snakemake.wildcards.brain_phase

X = indices[model_phase](X)
Y = indices[snakemake.wildcards.brain_phase](Y)
pred_Y = cross_val_predict(estimator, X, Y, cv=LeaveOneOut())

# Calculate the score
masker: NiftiMasker = pickle.load(open(snakemake.input.masker, "rb"))

scores = np.arctanh(batched_pearsonr(Y, pred_Y))
masker.inverse_transform(scores).to_filename(snakemake.output.score)

# Also save the predictions themselves
masker.inverse_transform(pred_Y).to_filename(snakemake.output.pred)