import math

from tqdm.auto import tqdm


def calculate_shannon_entropy(text: str) -> float:
    """Compute shanon entropy of a text.

    Args:
        text (str): input text.

    Returns:
        float: entropy of the text.

    """
    # Count character frequencies
    freq = {}
    for char in text:
        if char in freq:
            freq[char] += 1
        else:
            freq[char] = 1

    # Calculate entropy
    entropy = 0
    for count in freq.values():
        prob = count / len(text)
        entropy -= prob * math.log2(prob)

    return entropy


def filter_by_entropy(
    texts: list[str],
    min_entropy: float = 2,
    max_entropy: float = 5,
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
    return [
        text
        for text in tqdm(texts, desc="Entropy Filtering")
        if min_entropy <= calculate_shannon_entropy(text) <= max_entropy
    ]
