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

import json
import os
from tqdm import tqdm
from datasets import load_dataset
from nltk.corpus import wordnet as wn

"""
In this file we get the model class names, and map them to the behavior classes. 
The behavior classes are simply a text file list of names that might contain
repetitions.

Model class name is nxxxxx wordnet synset offsets. We want to convert it to match
the ones used by some studies indicated by the input file. We thus produce a json dictionary file
that maps from model label to the target label. Some classes there
are not in the imagenet, in which case we simply assign the word "other" to them.

An appropriate mapping is when the target class name is a hypernym of the model class name.
For example, if the target label has dogs, we can map model classes like basenji to it.

An easy way to test this is by testing for lowest common hypernyms. If the lowest common hypernym
turns out to be the target label, then we have a match. Otherwise, we don't.
"""

# Get the original model class names
config = snakemake.config

# Get the model's class names
model_class_names = load_dataset(config["dataset"], split=config["val_split_name"], cache_dir=config["cache_dir"]).features["label"].names

# Get the target labels
with open(str(snakemake.input), "r") as f:
    target_labels = f.read().splitlines()

# Get the unique labels for both
unique_model_class_names = list(set(model_class_names))
unique_target_labels = list(set(target_labels))

# Check which pair of [target_label, model_class_name] have a hypernym relationship
# TODO: A few false negatives: DOG & Chihuahua? Trolleybus & bus?
mapping = {k: "other" for k in unique_model_class_names}
for model_class in tqdm(unique_model_class_names):

    # Sometimes we don't get synset id but name. We instead directly get the synset from name
    if model_class[1:].isnumeric():
        model_class_synset = [wn.synset_from_pos_and_offset("n", int(model_class[1:]))]
    else:
        model_class_synset = [wn.synsets(model_class_sub, pos=wn.NOUN) for model_class_sub in model_class.split(",")]
        new_list = []
        for synset in model_class_synset:
            if isinstance(synset, list) and len(synset) == 0:
                continue
            elif isinstance(synset, list):
                new_list.extend(synset)
            else:
                new_list.append(synset)
        model_class_synset = new_list

    for target_label in unique_target_labels:
        target_label_synset = wn.synsets(target_label, pos=wn.NOUN)

        to_break = False
        for l in target_label_synset:
            for m in model_class_synset:
                lch = m.lowest_common_hypernyms(l)

                # Check if lch is same as target_label
                if any(x == l for x in lch):
                    mapping[model_class] = target_label
                    to_break = True
                    break
            if to_break:
                break

# Which keys still have "other"? Save as a text file
na_keys = [k for k, v in mapping.items() if v == "other"]
with open(str(snakemake.output.na_classes), "w") as f:
    f.write("\n".join(na_keys))

# Which target class doesn't have a match? Save as a text file
no_match = [k for k in unique_target_labels if k not in mapping.values()]
with open(str(snakemake.output.no_match_classes), "w") as f:
    f.write("\n".join(no_match))

# Save the dictionary as json
with open(str(snakemake.output.conversion), "w") as f:
    json.dump(mapping, f)
