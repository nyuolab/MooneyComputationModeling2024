import numpy as np

from tqdm import tqdm
from rsatoolbox.util.searchlight import get_volume_searchlight

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import batched_pearsonr


def evaluate_similarity_sl(map_1, map_2, mask, radius, threshold):
    # Implementation mostly taken from rsatoolbox's searchlight
    centers, neighbors = get_volume_searchlight(mask, radius, threshold)

    centers = np.array(centers)
    n_centers = centers.shape[0]

    # Flatten both data
    if len(map_1.shape) == 4:
        assert map_1.shape[-1] == 1, "Maps must be 3D"

    map_1, map_2 = map_1.reshape(-1), map_2.reshape(-1)

    # we can't run all centers at once, that will take too much memory
    # so lets to some chunking
    chunked_center = np.split(np.arange(n_centers), np.linspace(0, n_centers, 101, dtype=int)[1:-1])

    # loop over chunks
    scores = np.zeros((n_centers,))
    for chunks in tqdm(chunked_center, desc='Computing scores...'):
        center_data = []
        for c in chunks:
            center_neighbors = neighbors[c]
            d1 = map_1[center_neighbors]
            d2 = map_2[center_neighbors]
            center_data.append(batched_pearsonr(d1, d2))

        scores[chunks] = center_data

    # Return the score restored to the original shape
    scores_original = np.zeros(mask.flatten().shape)
    scores_original[list(centers)] = scores
    scores_original = scores_original.reshape(mask.shape)

    return scores_original