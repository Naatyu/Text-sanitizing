import math
from multiprocessing import Pool

import numpy as np
from tqdm.auto import tqdm


def calculate_shannon_entropy_numpy(text: str) -> float:
    """Compute Shannon entropy using NumPy."""
    counts = np.unique(list(text), return_counts=True)[1]
    probs = counts / len(text)
    return -np.sum(probs * np.log2(probs))


def filter_by_entropy(
    texts: list[str],
    min_entropy: float = 2,
    max_entropy: float = 5,
    num_workers: int = 4,
) -> list[str]:
    """Filter texts with shanon entropy.

    Args:
        texts (listr[str]): list of texts to filter.
        min_entropy (float, optional): minimum entropy for filtering.
                                       Defaults to 2.
        max_entropy (float, optional): maximum entropy for filtering.
                                       Defaults to 5.

    Returns:
        list[str]: _description_

    """
    with Pool(processes=num_workers) as pool:
        results = pool.map(
            calculate_shannon_entropy_numpy,
            tqdm(texts, desc="Calculating Entropy", unit_scale=True),
        )

    filtered_text = []
    for entropy, text in tqdm(
        zip(results, texts, strict=True),
        total=len(results),
        desc="Entropy filtering",
        unit_scale=True,
    ):
        if min_entropy <= entropy <= max_entropy:
            filtered_text.append(text)

    return filtered_text
