import os

from datasets import load_dataset

from modules.entropy import filter_by_entropy
from modules.line_deduplication import (
    get_duplicated_hashed_lines,
    remove_duplicated_lines,
)
from modules.min_hash import find_similar_texts

if __name__ == "__main__":
    # Load data
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr")

    # --- Entropy --- #

    filtered_texts = filter_by_entropy(
        ds["train"][0:10000]["text"],
        num_workers=os.cpu_count() - 1,
    )

    # --- MinHashing --- #
    similar_docs = find_similar_texts(
        filtered_texts,
        threshold=0.9,
        return_only_idx=True,
    )

    # Remove similard docs
    ds["train"] = ds["train"].select(
        idx for idx in range(len(ds["train"])) if idx not in similar_docs
    )

    # --- Line-level deduplication --- #
    hashes = get_duplicated_hashed_lines(ds["train"][0:1000]["text"])
    line_deduplicated_txts = remove_duplicated_lines(
        ds["train"][0:1000]["text"],
        to_remove=hashes,
    )

    print("Done !")
