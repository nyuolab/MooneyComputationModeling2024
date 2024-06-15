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



import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm, trange

from transformers import AutoImageProcessor

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.model import TopDownPerceiver

config = snakemake.config
sequence = pd.read_csv(snakemake.input.behavior_sequence)

# Get the device
if torch.cuda.is_available() and not config["use_cpu"]:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the checkpoint
model = TopDownPerceiver.load_from_checkpoint(snakemake.input.checkpoint)
image_processor = AutoImageProcessor.from_pretrained("/".join([config["model_prefix"], snakemake.wildcards.backbone]))
model.eval().to(device)

# Get a list of grayscale image index -> filepath also
grayscale_img_paths = [None for _ in range(33)]

states = []
prev_state = None
for idx, row in tqdm(sequence.iterrows(), total=len(sequence)):
    # Turn it into a dictionary
    row = row.to_dict()

    # If file doesn't exist, remove the trailing 1 before .bmp
    image_path = row["image_path"]
    if not os.path.isfile(image_path):
        image_path = image_path.replace("1.bmp", ".bmp")

    # Add the grayscale image index to the list
    if grayscale_img_paths[row["image_index"]] is None:
        grayscale_img_paths[row["image_index"]] = image_path
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

    # Store the current state
    states.append(prev_state)

states = torch.concatenate(states, dim=0)

# Store a new state
new_state = torch.zeros(len(states), *out.latent_vis.shape[1:], device=states.device)

for grayscale_idx in range(33):
    # Randomly select a grayscale image
    for to_replace_idx in trange(31, desc=f"Grayscale {grayscale_idx}"):
        # Replace the image path of the grayscale image with a different one
        sequence_copy = sequence.copy()
        sequence_copy.loc[sequence_copy["image_index"] == grayscale_idx, "image_path"] = snakemake.input.catch_images[to_replace_idx]

        # Find when first grayscale image with this index appears
        first_idx = sequence_copy[(sequence_copy["image_index"] == grayscale_idx) & (sequence_copy["phase"] == "gray")].index[0]
        last_idx = sequence_copy[sequence_copy["image_index"] == grayscale_idx].index[-1] + 1
        sub_sequence = sequence_copy.loc[first_idx:last_idx]

        # Start running as if we are restarting
        prev_state = states[first_idx - 1][None]
        for idx, row in sub_sequence.iterrows():
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

            # Optionally store the new state
            if row["phase"] in ["post", "gray"] and row["image_index"] == grayscale_idx:
                new_state[idx] += out.latent_vis[0]

            # Get the previous states
            prev_state = out.state

# Average the new state
new_state = new_state.reshape(len(new_state), -1)
new_state /= 31

# Combine with repetition feat
# rep_feat = np.load(snakemake.input.repetition_feat)

# Save the new state
np.save(snakemake.output.state, new_state.cpu().numpy())
# np.save(snakemake.output.state, np.concatenate([new_state.cpu().numpy(), rep_feat], axis=1))