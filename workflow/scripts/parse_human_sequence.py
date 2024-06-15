import re
import pandas as pd

# Load the class name for each image
curr_subject = int(snakemake.wildcards.subject)
class_names = open(str(snakemake.input.classes), "r").read().split("\n")

# Load the order of image presented to a subject
img_name_order = open(str(snakemake.input.image_order), "r").read().split("\n")

# Operating mode
mode = snakemake.params.mode

# Make the convertion from image name to image index
if mode == "fmri":
    # Load the filenames
    img_filenames = open(str(snakemake.input.image_filenames), "r").read().split("\n")

    if img_name_order[0] == "ImagePresented":
        img_name_order = img_name_order[1:]

    pattern = re.compile(r'--(.*?)\.')
    conversion = {}
    for idx, image_file in enumerate(img_name_order):
        suffix = pattern.search(image_file).group(1)
        if suffix not in img_filenames:
            suffix = f"{suffix}1"
        conversion[image_file] = img_filenames.index(suffix)
elif mode == "behavior":
    # Note that some data we receive are 1-indexed
    conversion = {k: int(k[1:].split('.')[0])-1 for k in set(img_name_order)}
else:
    raise ValueError(f"Unknown mode {mode}")

# Indicator of image phase
def guess_phase(image_name):
    if "filter" in image_name:
        return "gray"
    elif image_name.startswith("1"):
        return "gray"
    else:
        return ["pre", "post"]

# Loop through the order
subjects = []
phases = []

image_path = []
image_index = []
image_class = []

# Some states
previous_grays = []
prev_pre_indices = []

# Loop through the images
for present_idx, image_name in enumerate(img_name_order):
    # Get image index
    curr_image_index = conversion[image_name]
    guess_phase_curr = guess_phase(image_name)

    # Guess the phase
    if guess_phase_curr == "gray":
        previous_grays.append(curr_image_index)
        phase = "gray"
    elif len(guess_phase_curr) == 1:
        phase = guess_phase_curr[0]
    else:
        phase = "pre" if curr_image_index not in previous_grays else "post"

    # If phase is post, delete it from the previous grays
    if phase == "post":
        previous_grays.remove(curr_image_index)

    # Keep track of image index, block, and subject
    image_index.append(curr_image_index)
    phases.append(phase)

    # Keep track of image path and class
    if mode == "behavior":
        if phase == "gray":
            image_path.append(f"data/behavior_images/1_grayscale/{image_name}")
        else:
            image_path.append(f"data/behavior_images/0_Mooney/{image_name}")
    elif mode == "fmri":
        image_path.append(f"data/fmri_images/{image_name}")
    image_class.append(class_names[conversion[image_name]])

# Write the dataframe
out = pd.DataFrame({
    "phase": phases,
    "image_index": image_index,
    "image_path": image_path,
    "class": image_class,
}).to_csv(str(snakemake.output), index=False)