import torch
import numpy as np

from scipy.stats import rankdata

from einops import reduce, rearrange
from sklearn.feature_extraction.image import grid_to_graph

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.rsa import get_phase_img


def batched_pearsonr(x, y):
    """
    Compute the Pearson correlation coefficient for each dimension across N pairs.
    
    Args:
    - x (numpy.ndarray): Input array of shape (N, D) where N is the number of samples and D is the number of dimensions.
    - y (numpy.ndarray): Input array of shape (N, D) where N is the number of samples and D is the number of dimensions.
    
    Returns:
    - numpy.ndarray: An array of Pearson correlation coefficients of shape (D,).
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Compute means
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    
    # Compute deviations
    dev_x = x - mean_x
    dev_y = y - mean_y
    
    # Compute covariance for each dimension
    covariance = np.sum(dev_x * dev_y, axis=0) / (x.shape[0] - 1)
    
    # Compute standard deviations
    std_x = np.sqrt(np.sum(dev_x**2, axis=0) / (x.shape[0] - 1))
    std_y = np.sqrt(np.sum(dev_y**2, axis=0) / (y.shape[0] - 1))
    
    # Compute Pearson correlation coefficient for each dimension
    pearson_correlation = covariance / (std_x * std_y)
    
    return pearson_correlation


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def get_attention_map(model, pre_img, gray_img, post_img, n_conds):
    # Pass images through the model
    with torch.no_grad():
        out_pre = model(pre_img)
        out_gray = model(
            gray_img,
            prev_state=out_pre.state,
        )
        out_post = model(
            post_img,
            prev_state=out_gray.state,
        )

    # Look at attention shape
    container = {}
    for out, phase in [(out_pre, "pre"), (out_gray, "gray"), (out_post, "post")]:
        attn = out.attentions

        # First average over heads
        attn = torch.stack(attn, dim=1)
        attn = reduce(attn, "time l h f t -> time l f t", "mean")

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = rearrange(torch.eye(attn.size(-1), device=attn.device), "f t -> () () f t")
        aug_att_mat = attn + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size(), device=attn.device)
        joint_attentions[:, 0] = aug_att_mat[:, 0]

        for n in range(1, aug_att_mat.size(1)):
            joint_attentions[:, n] = torch.matmul(aug_att_mat[:, n], joint_attentions[:, n-1])

        # Take the last layer
        joint_attentions = joint_attentions[:, -1]

        # Take cls token (1st token)
        joint_attentions = joint_attentions[:, 0, :]

        # Remove cls, and condition tokens from attention
        joint_attentions = joint_attentions[:, 1+n_conds:]

        # Reshape
        grid_size = int(joint_attentions.shape[1] ** 0.5)
        joint_attentions = joint_attentions.reshape(1, grid_size, grid_size)

        joint_attentions = np.asarray(joint_attentions.cpu()).transpose(1, 2, 0)
        mask = joint_attentions / joint_attentions.max()

        container[phase] = mask

    # Make the difference image between pre and post
    diff_mask = container["post"] - container["pre"]

    # Scale to 0-1
    container["diff_inc"] = np.clip(diff_mask, 0, None) / np.max(diff_mask)
    container["diff_dec"] = np.clip(-diff_mask, 0, None) / (-np.min(diff_mask))

    return container, out_pre, out_gray, out_post


def get_model_features(sequence_record, features, phase_limit=None):
    # Iterate through the image index and get the average features
    representations = []

    all_phases = ["pre", "post", "gray"]

    if phase_limit is not None:
        if phase_limit == "all":
            phase_limit = all_phases
        else:
            phase_limit = [phase_limit]

    for p in phase_limit:
        for i in sorted(sequence_record["Image index"].unique()):
            index_to_choose = sequence_record[(sequence_record["Image index"] == i) & (sequence_record["Image phase"] == p)].index
            representations.append(np.mean(features[index_to_choose], axis=0))

    return np.stack(representations, axis=0)


def mean_diff_pixels(image):
    if len(image.shape) == 3:
        assert image.shape[2] == 1
        image = image[:, :, 0]

    horizontal_diffs = np.abs(np.diff(image, axis=1))
    vertical_diffs = np.abs(np.diff(image, axis=0))

    average_horizontal_diff = np.mean(horizontal_diffs)
    average_vertical_diff = np.mean(vertical_diffs)

    # Compute overall average difference
    return (average_horizontal_diff + average_vertical_diff) / 2


def batched_partial_corr(x, y, z, method="pearson", semi=False):
    # Adapted from https://github.com/raphaelvallat/pingouin/blob/f4593c5ae898410aa5b46f639a0a44e2979c3a68/src/pingouin/correlation.py#L678

    # x, y, z shape: (n, p)
    data = np.stack([x, y, z], axis=2)

    # If use spearman, convert the data to rank
    if method == "spearman":
        data = rankdata(data, axis=1, method="average")

    # Remove mean
    mean = np.mean(data, axis=1, keepdims=True)
    data_centered = data - mean

    # Compute covariance matrix: (N, 3, 3)
    V = np.einsum('ijk,ijl->ikl', data_centered, data_centered) / (data.shape[1] - 1)

    # Inverting V can be directly done batched: (N, 3, 3)
    Vi = np.linalg.pinv(V, hermitian=True)  # Inverse covariance matrix

    # Take batched diagonal: (N, 3)
    Vi_diag = Vi.diagonal(offset=0, axis1=1, axis2=2)

    # Broadcast magic to get the diagonal matrix instead of np.diag
    D = np.eye(3)[np.newaxis] * np.sqrt(1 / Vi_diag)[:, :, np.newaxis]
    pcor = -1 * (D @ Vi @ D)

    if not semi:
        return pcor[:, 0, 1]
    else:
        spcor = (
            pcor
            / np.sqrt(V.diagonal(offset=0, axis1=1, axis2=2))[..., None]
            / np.sqrt(np.abs(Vi_diag[:, np.newaxis] - Vi**2 / Vi_diag[..., None])).transpose(0, 2, 1)
        )
        return spcor[:, 1, 0]