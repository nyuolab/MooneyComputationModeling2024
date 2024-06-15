import numpy as np
import itertools

def get_phase(rdm, phase):
    assert phase in ["pre", "gray", "post", "pre-post", "pre-gray", "post-gray", "all"]
    if phase == "pre":
        res = rdm[:33, :33]
    elif phase == "post":
        res = rdm[33:66, 33:66]
    elif phase == "gray":
        res = rdm[66:, 66:]
    elif phase == "pre-post":
        res = np.block([
            [rdm[:33, :33], rdm[:33, 33:66]],
            [rdm[33:66, :33], rdm[33:66, 33:66]],
        ])
    elif phase == "pre-gray":
        res = np.block([
            [rdm[:33, :33], rdm[:33, 66:]],
            [rdm[66:, :33], rdm[66:, 66:]],
        ])
    elif phase == "post-gray":
        res = np.block([
            [rdm[33:66, 33:66], rdm[33:66:, 66:]],
            [rdm[66:, 33:66], rdm[66:, 66:]]
        ])
    elif phase == "all":
        res = rdm
    else:
        raise ValueError(f"Unrecognized phase: {phase}")
    return res


def get_phase_data_sample_first(data, phase):
    assert phase in ["pre", "gray", "post", "pre-post", "pre-gray", "post-gray", "all"]
    if phase == "pre":
        return data[:33]
    elif phase == "post":
        return data[33:66]
    elif phase == "gray":
        return data[66:]
    elif phase == "pre-post":
        return np.concatenate([data[:33], data[33:66]], axis=0)
    elif phase == "pre-gray":
        return np.concatenate([data[:33], data[66:]], axis=0)
    elif phase == "post-gray":
        return np.concatenate([data[33:66], data[66:]], axis=0)
    elif phase == "all":
        return data
    else:
        raise ValueError(f"Unrecognized phase: {phase}")


def get_phase_data(data, phase):
    assert phase in ["pre", "gray", "post", "pre-post", "pre-gray", "post-gray", "all"]
    if phase == "pre":
        return data[..., :33]
    elif phase == "post":
        return data[..., 33:66]
    elif phase == "gray":
        return data[..., 66:]
    elif phase == "pre-post":
        return np.concatenate([data[..., :33], data[..., 33:66]], axis=-1)
    elif phase == "pre-gray":
        return np.concatenate([data[..., :33], data[..., 66:]], axis=-1)
    elif phase == "post-gray":
        return np.concatenate([data[..., 33:66], data[..., 66:]], axis=-1)
    elif phase == "all":
        return data
    else:
        raise ValueError(f"Unrecognized phase: {phase}")


from nilearn.image import index_img, concat_imgs
def get_phase_img(img, phase):
    if phase == "pre":
        return index_img(img, slice(0, 33))
    elif phase == "post":
        return index_img(img, slice(33, 66))
    elif phase == "gray":
        return index_img(img, slice(66, 99))
    elif phase == "pre-post":
        return index_img(img, slice(0, 66))
    elif phase == "pre-gray":
        return concat_imgs([index_img(img, slice(0, 33)), index_img(img, slice(66, 99))])
    elif phase == "post-gray":
        return concat_imgs([index_img(img, slice(33, 66)), index_img(img, slice(66, 99))])
    elif phase == "all":
        return img
    else:
        raise ValueError(f"Unrecognized phase: {phase}")


def calculate_within_block_rank(block):
    """
    Calculate the within-block rank for a given block.
    """
    if np.allclose(block, block.T):
        # Calculating rank for the upper triangular part
        upper_triangular = np.triu(block, k=1)
        rank_upper = np.zeros_like(upper_triangular)

        # Argsort twice gives the ranks
        ranks = np.argsort(np.argsort(upper_triangular, axis=None), axis=None).reshape(upper_triangular.shape)
        rank_upper = np.triu(ranks, k=1)
        block = rank_upper + rank_upper.T

    else:
        vec = block.flatten()
        block = np.argsort(np.argsort(vec, axis=None), axis=None).reshape(block.shape)

    # Normalizing the rank within the block
    min_rank = np.min(block)
    max_rank = np.max(block)
    return (block - min_rank) / (max_rank - min_rank)

def adjust_percentile_rdm(rdm):
    block_size = 33
    assert rdm.shape[0] % block_size == 0

    n_blocks = rdm.shape[0] // block_size
    result_matrix = np.zeros(rdm.shape)
    for i, j in itertools.product(range(n_blocks), range(n_blocks)):
        if i <= j:
            # Extracting block
            block = rdm[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]

            # Calculating within-block rank
            rank_block = calculate_within_block_rank(block)

            # Putting block back into result matrix
            result_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = rank_block

    return result_matrix
