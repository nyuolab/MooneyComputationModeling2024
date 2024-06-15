import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


feature_source = pd.read_csv(snakemake.input.feature_source)
features =  np.load(snakemake.input.features)

if (limit := snakemake.config["limit_prediction_dimensions"]) > 0:
    features = features[:, :limit]

# Get models for each case
model = Pipeline([("transformer", StandardScaler()), ("classifier", SVC())])
dummy_model = DummyClassifier(strategy="most_frequent")

# Determine the phase
phase = snakemake.wildcards.phase

# Container for features
n_images = len(feature_source["Image index"].unique())
aggregated_features = np.zeros((n_images, features.shape[1]))
counter = np.zeros(n_images)

# Look at image index
for idx, row in feature_source[feature_source["Image phase"] == phase].iterrows():
    aggregated_features[row["Image index"]] += features[idx]
    counter[row["Image index"]] += 1

# Average
X = aggregated_features / counter[:, None]

# Take the subject we want
sub = int(snakemake.wildcards.sub)
y = loadmat(snakemake.input.target_pred, struct_as_record=True)['correct_diff'][:, sub]

# Preprocess y
new_y = []
for idx in range(len(y)):
    try:
        tmp = y[idx].item()
        new_y.append(tmp)
    except ValueError:
        # Garbage matlab gives you literal not a number
        new_y.append(np.nan)
y = np.array(new_y)

# Remove nan
nan_mask = ~np.isnan(y)
y = y[nan_mask]
X = X[nan_mask]

# Train model and get best performance
score = cross_val_score(model, X, y, scoring="roc_auc", cv=10).mean()

# Get dummy performance
dummy_score = cross_val_score(dummy_model, X, y, scoring="roc_auc", cv=10).mean()

perf_name = f"{snakemake.wildcards.feature_type}_perf"
data = {
    "perf_type": ["CV", "Chance"],
    "subject": [sub] * 2,
    perf_name: [score, dummy_score]
}

# Save
pd.DataFrame(data).to_csv(snakemake.output[0], index=False)
