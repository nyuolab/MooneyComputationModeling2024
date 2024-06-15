import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import torch
import pandas as pd
import numpy as np

import json

from PIL import Image
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoImageProcessor
from src.model import TopDownPerceiver

config = snakemake.config
sequence = pd.read_csv(snakemake.input.behavior_sequence)

if hasattr(snakemake.params, "no_save"):
    no_save = snakemake.params.no_save
else:
    no_save = False

# Get the device
if torch.cuda.is_available() and not config["use_cpu"]:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Get the model's class names
class_names = load_dataset(config["dataset"], split=config["val_split_name"], cache_dir=config["cache_dir"]).features["label"].names

# Load the checkpoint
model = TopDownPerceiver.load_from_checkpoint(snakemake.input.checkpoint)
image_processor = AutoImageProcessor.from_pretrained("/".join([config["model_prefix"], snakemake.wildcards.backbone]))
model.eval().to(device)

# List to keep track of outputs
image_step = []
global_step = []

# Keep image index and phase
image_index = []
image_phase = []

predicted_class = []
groundtruth_class = []

# Add the recurrent states
state_container = {k: [] for k in config['feature_types']}
layer_container = {k: [] for k in config['layer_feature_types']}

prev_state = None
for idx, row in tqdm(sequence.iterrows(), total=len(sequence)):
    # Turn it into a dictionary
    row = row.to_dict()

    # If file doesn't exist, remove the trailing 1 before .bmp
    image_path = row["image_path"]
    if not os.path.isfile(image_path):
        image_path = image_path.replace("1.bmp", ".bmp")
    loaded_img = Image.open(image_path).convert("L")

    # Get the images
    image = image_processor(
        loaded_img,
        return_tensors="pt",
    )

    # View video
    with torch.no_grad():
        out = model(
            image['pixel_values'].to(device),
            prev_state=prev_state,
        )

    # Get the previous states
    prev_state = out.state

    # Get the predicted class
    class_indices = out.logits.argmax(dim=-1).flatten().tolist()
    class_indices = [min(idx, len(class_names) - 1) for idx in class_indices]
    predicted_class.extend(class_names[idx] for idx in class_indices)
    groundtruth_class.extend([row["class"]])

    # Then get the register features
    for feat_name in config['feature_types']:
        state_container[feat_name].append(getattr(out, feat_name).reshape(-1).cpu().numpy())

    for feat_name in config['layer_feature_types']:
        layer = feat_name.split("_")[1]
        layer_container[feat_name].append(out.hidden_states[int(layer)].reshape(-1).cpu().numpy())

    # Keep track of the image index and phase
    image_index.extend([row["image_index"]])
    image_phase.extend([row["phase"]])

# Make a dataframe
results = pd.DataFrame({
    "Image index": image_index,
    "Image phase": image_phase,
    "Predicted class": predicted_class,
    "Ground truth class": groundtruth_class,
})

# Save the results
results.to_csv(snakemake.output[0], index=False)

if no_save:
    exit()

layer_container = {k: np.stack(v, axis=0) for k, v in layer_container.items()}
state_container = {k: np.stack(v, axis=0) for k, v in state_container.items()}

# Make the accumulated and static features
static_container = {k: [] for k in config['static_feature_types']}
for idx, row in tqdm(sequence.iterrows(), total=len(sequence), desc="Static features"):
    # Turn it into a dictionary
    row = row.to_dict()

    # If file doesn't exist, remove the trailing 1 before .bmp
    image_path = row["image_path"]
    if not os.path.isfile(image_path):
        image_path = image_path.replace("1.bmp", ".bmp")
    loaded_img = Image.open(image_path).convert("L")

    # Get the images
    image = image_processor(
        loaded_img,
        return_tensors="pt",
    )

    # View video
    with torch.no_grad():
        out = model(image['pixel_values'].to(device))

    # Get the static features
    for feat_name in config['static_feature_types']:
        source_feat_name = feat_name.split("_")[1]
        static_container[feat_name].append(getattr(out, source_feat_name).reshape(-1).cpu().numpy())

static_container = {k: np.stack(v, axis=0) for k, v in static_container.items()}

# Then make the accumulated features
rate = model.model.accumulate_state.cpu().item()
accumulate_container = {k: [] for k in config['accumulate_feature_types']}
for feat_name in config['accumulate_feature_types']:
    source_feat_name = feat_name.split("_")[1]

    last_rep = model.model.init_state.detach().cpu().numpy()

    n_conditions = snakemake.config["n_condition_tokens"]
    if source_feat_name == "vis":
        last_rep = last_rep[:, 1+n_conditions:, :].reshape(-1)
    elif source_feat_name == "cls":
        last_rep = last_rep[:, 1:, :].reshape(-1)

    for t in range(static_container[f"static_{source_feat_name}"].shape[0]):
        last_rep = rate * last_rep + (1 - rate) * static_container[f"static_{source_feat_name}"][t].reshape(-1)
        accumulate_container[feat_name].append(last_rep)

accumulate_container = {k: np.stack(v, axis=0) for k, v in accumulate_container.items()}

# Save
all_dict = accumulate_container | layer_container | state_container | static_container
for idx, feature_name in enumerate(config["feature_types"]+config["layer_feature_types"]+config["static_feature_types"]+config["accumulate_feature_types"]):
    np.save(snakemake.output.representations[idx], all_dict[feature_name])