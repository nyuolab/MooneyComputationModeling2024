import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from datasets import load_dataset

from src.datamodules.standard import transform
from src.utils import mean_diff_pixels

import torch
from tqdm import trange

import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor
from src.model import TopDownPerceiver
from src.utils import get_attention_map

from scipy.stats import pearsonr

mode = snakemake.params.mode
config = snakemake.config

if mode == "single":
    val_set = load_dataset(config["dataset"], split=config["val_split_name"], cache_dir=config["cache_dir"])

    # Shuffle the dataset
    val_set = val_set.shuffle()

    # Get the device
    if torch.cuda.is_available() and not config["use_cpu"]:
        device = torch.device("cuda")
        limit = 5000
    else:
        device = torch.device("cpu")
        limit = 10

    n_conds = config["n_condition_tokens"]
    model = TopDownPerceiver.load_from_checkpoint(snakemake.input.model_checkpoint)
    image_processor = AutoImageProcessor.from_pretrained("/".join([config["model_prefix"], snakemake.wildcards.backbone]))
    model.eval().to(device)

    # Check similarity between gray and pre, gray and post
    pre_correlation = []
    post_correlation = []

    # The other one tracks the change for each grayscale
    token_idx = []
    gray_rank = []
    gray_score = []
    increase_score = []
    pre_moran_score = []
    post_moran_score = []
    correct = []
    pre_correct = []

    for idx in trange(len(val_set)):
        if limit > 0 and idx > limit:
            break

        image_dict = transform(image_processor, val_set[idx])

        mooney_image = torch.tensor(image_dict["binarized_pixel_values"]).to(device).unsqueeze(0)
        grayscale_img = torch.tensor(image_dict["pixel_values"]).to(device).unsqueeze(0)

        containers, out_pre, out_gray, out_post = get_attention_map(model, mooney_image, grayscale_img, mooney_image, n_conds)
        gray_score_curr, pre_score_curr, post_score_curr = containers["gray"].reshape(-1), containers["pre"], containers["post"]

        # Sum to 1
        gray_score_curr = gray_score_curr / gray_score_curr.sum()
        pre_score_curr = pre_score_curr / pre_score_curr.sum()
        post_score_curr = post_score_curr / post_score_curr.sum()

        pre_correlation.append(pearsonr(gray_score_curr, pre_score_curr.flatten())[0])
        post_correlation.append(pearsonr(gray_score_curr, post_score_curr.flatten())[0])

        for idx, (pre, post, gray) in enumerate(zip(pre_score_curr.flatten(), post_score_curr.flatten(), gray_score_curr)):
            token_idx.append(idx)
            gray_score.append(gray)
            increase_score.append(post - pre)

            # The percentile of the gray score
            gray_rank.append(
                (gray_score_curr < gray).sum().item() / len(gray_score_curr)
            )

        # Add the moran's I
        pre_moran_score.append(mean_diff_pixels(pre_score_curr).item())
        post_moran_score.append(mean_diff_pixels(post_score_curr).item())
        pre_correct.append(image_dict["label"] == out_pre.logits.argmax().item())
        correct.append(image_dict["label"] == out_post.logits.argmax().item())

    # Save to a txt file as well
    with open(snakemake.output.pre_correlation, "w") as f:
        f.write("\n".join([str(item) for item in pre_correlation]))
    with open(snakemake.output.post_correlation, "w") as f:
        f.write("\n".join([str(item) for item in post_correlation]))

    with open(snakemake.output.token_idx, "w") as f:
        f.write("\n".join([str(item) for item in token_idx]))
    with open(snakemake.output.gray_score, "w") as f:
        f.write("\n".join([str(item) for item in gray_score]))
    with open(snakemake.output.gray_rank, "w") as f:
        f.write("\n".join([str(item) for item in gray_rank]))
    with open(snakemake.output.increase_score, "w") as f:
        f.write("\n".join([str(item) for item in increase_score]))

    with open(snakemake.output.pre_moran_score, "w") as f:
        f.write("\n".join([str(item) for item in pre_moran_score]))
    with open(snakemake.output.post_moran_score, "w") as f:
        f.write("\n".join([str(item) for item in post_moran_score]))
    with open(snakemake.output.correct, "w") as f:
        f.write("\n".join([str(item) for item in correct]))
    with open(snakemake.output.pre_correct, "w") as f:
        f.write("\n".join([str(item) for item in pre_correct]))

elif mode == "all":

    pre_correlation = []
    post_correlation = []

    # The other one tracks the change for each grayscale
    token_idx = []
    gray_score = []
    gray_rank = []
    increase_score = []

    pre_moran_score = []
    post_moran_score = []

    # Load the text files
    for file in snakemake.input.pre_correlation:
        with open(file, "r") as f:
            pre_correlation += [float(line.strip()) for line in f.readlines()]

    for file in snakemake.input.post_correlation:
        with open(file, "r") as f:
            post_correlation += [float(line.strip()) for line in f.readlines()]
    
    for file in snakemake.input.token_idx:
        with open(file, "r") as f:
            token_idx += [int(line.strip()) for line in f.readlines()]

    for file in snakemake.input.gray_score:
        with open(file, "r") as f:
            gray_score += [float(line.strip()) for line in f.readlines()]

    for file in snakemake.input.gray_rank:
        with open(file, "r") as f:
            gray_rank += [float(line.strip()) for line in f.readlines()]

    for file in snakemake.input.increase_score:
        with open(file, "r") as f:
            increase_score += [float(line.strip()) for line in f.readlines()]
        
    for file in snakemake.input.pre_moran_score:
        with open(file, "r") as f:
            pre_moran_score += [float(line.strip()) for line in f.readlines()]
    for file in snakemake.input.post_moran_score:
        with open(file, "r") as f:
            post_moran_score += [float(line.strip()) for line in f.readlines()]

    # Make a dataframe
    df = pd.DataFrame({
        "Visual attention pattern similarity": pre_correlation+post_correlation,
        "Phase": ["Pre"] * len(pre_correlation) + ["Post"] * len(post_correlation)
    })

    # Plot and annotate difference
    ax = sns.barplot(data=df, x="Phase", y="Visual attention pattern similarity")
    pairs = [["Pre", "Post"]]
    annotator = Annotator(ax, pairs, data=df, x="Phase", y="Visual attention pattern similarity")
    annotator.configure(test='Mann-Whitney', text_format='star', comparisons_correction="BH")
    annotator.apply_and_annotate()
    plt.title("Alignment of attention with grayscale phase")

    # Save the plot
    plt.savefig(snakemake.output.fig_out)
    plt.clf()
    del df

    # Other df
    df = pd.DataFrame({
        "Token idx": token_idx,
        "Gray score": gray_score,
        "Gray rank": gray_rank,
        "Probability of increase in score": [s>0 for s in increase_score],
    })

    # Plot
    sns.lineplot(data=df, x="Gray rank", y="Probability of increase in score")
    plt.savefig(snakemake.output.fig_out2)
    plt.clf()

    # Just plot grayscale score disribution
    sns.histplot(data=df, stat="probability", x="Gray score", bins=30, kde=True)
    plt.savefig(snakemake.output.fig_out3)
    plt.clf()
    del df

    # Finally plot the 
    df = pd.DataFrame({
        "Token idx": token_idx,
        "Gray score": gray_score,
        "Gray rank": gray_rank,
        "Change": increase_score,
    })
    pos_df = df[df["Change"] > 0]
    neg_df = df[df["Change"] < 0]

    pos_df["Change"] = pos_df["Change"].abs()
    neg_df["Change"] = neg_df["Change"].abs()

    pos_df["Post phase change"] = "Increase"
    neg_df["Post phase change"] = "Decrease"
    df = pd.concat([pos_df, neg_df])

    # TODO: Annotate this
    ax = sns.lineplot(data=df, x="Gray rank", y="Change", hue="Post phase change", hue_order=["Increase", "Decrease"])
    plt.savefig(snakemake.output.fig_out4)
    plt.clf()

    # Plot magnitude of change
    ax = sns.lineplot(data=df, x="Gray rank", y="Change")
    plt.savefig(snakemake.output.fig_out5)
    plt.clf()
    del df

    # Plot the Moran's score of pre and post
    df = pd.DataFrame({
        "Moran's I": pre_moran_score + post_moran_score,
        "Phase": ["Pre"] * len(pre_moran_score) + ["Post"] * len(post_moran_score)
    })

    # Plot the Moran's I
    ax = sns.barplot(data=df, x="Phase", y="Moran's I")
    pairs = [["Pre", "Post"]]
    annotator = Annotator(ax, pairs, data=df, x="Phase", y="Moran's I")
    annotator.configure(test='Mann-Whitney', text_format='star', comparisons_correction="BH")
    annotator.apply_and_annotate()
    plt.title("Change of spatial autocorrelation")

    # Save the plot
    plt.savefig(snakemake.output.fig_out6)
    plt.clf()
    del df
