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


import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os

import json
import torch
import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoImageProcessor

from src.model import TopDownPerceiver
from src.datamodules.standard import transform

config = snakemake.config

# Number of grayscales to take
n_images = 210

# Take this number of images from the validation set
val_set = load_dataset(config["dataset"], split=config["val_split_name"], cache_dir=config["cache_dir"])

# Data
data = pd.read_csv(snakemake.input.sequence)
image_index = data["Image index"].tolist()
image_class = data["Ground truth class"].tolist()
image_phase = data["Image phase"].tolist()

# Get the device
if torch.cuda.is_available() and not config["use_cpu"]:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the checkpoint
model = TopDownPerceiver.load_from_checkpoint(snakemake.input.checkpoint)
image_processor = AutoImageProcessor.from_pretrained("/".join([config["model_prefix"], snakemake.wildcards.backbone]))
model.eval().to(device)

# Get performance
predicted_class = []
prev_state = None
for image_idx, label, phase in tqdm(zip(image_index, image_class, image_phase), total=len(image_index), desc="Evaluating model in pre and post"):
    image_dict = transform(image_processor, val_set[image_idx])

    if phase in ["pre", "post"]:
        image = image_dict["binarized_pixel_values"]
    else:
        image = image_dict["pixel_values"]

    # View video
    with torch.no_grad():
        out = model(
            torch.tensor(image).to(device).unsqueeze(0),
            prev_state=prev_state,
        )

    # Get the previous states
    prev_state = out.state

    # Get the predicted class
    predicted_class.append(out.logits.argmax(-1).item())

# Then get the repetition results
phase_to_add = []
label_to_add = []
predicted_class_to_add = []
prev_state = None
for image_idx, label, phase in tqdm(zip(image_index, image_class, image_phase), total=len(image_index), desc="Evaluate model in repetition"):
    image_dict = transform(image_processor, val_set[image_idx])

    if phase == "gray":
        continue
    elif phase == "post":
        phase = "repetition"

    image = image_dict["binarized_pixel_values"]

    # View video
    with torch.no_grad():
        out = model(
            torch.tensor(image).to(device).unsqueeze(0),
            prev_state=prev_state,
        )

    # Get the previous states
    prev_state = out.state

    if phase != "pre":
        # Get the predicted class
        label_to_add.append(label)
        phase_to_add.append(phase)
        predicted_class_to_add.append(out.logits.argmax(-1).item())

# Add the repetition results
image_class += label_to_add
image_phase += phase_to_add
predicted_class += predicted_class_to_add

# Count the number of same phase before
time_index = []
pre_count = 0
gray_count = 0
post_count = 0
repetition_count = 0
for p in image_phase:
    if p == "pre":
        time_index.append(pre_count)
        pre_count += 1
    elif p == "gray":
        time_index.append(gray_count)
        gray_count += 1
    elif p == "post":
        time_index.append(post_count)
        post_count += 1
    elif p == "repetition":
        time_index.append(repetition_count)
        repetition_count += 1

pd.DataFrame({
    "Image phase": image_phase,
    "Time index": time_index,
    "Ground truth class": image_class,
    "Predicted class": predicted_class,
}).to_csv(snakemake.output.records, index=False)