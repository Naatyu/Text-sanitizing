from datasets import load_dataset

from line_deduplication import (
    get_duplicated_hashed_lines,
    remove_duplicated_lines,
)
from min_hash import find_similar_texts

if __name__ == "__main__":
    # Load data
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr")

    # MinHashing
    similar_docs = find_similar_texts(
        ds["train"][0:1000]["text"],
        threshold=0.9,
        return_only_idx=True,
    )
    # similar_docs = sorted(
    #     similar_docs,
    #     key=lambda d: d["similarity"],
    #     reverse=True,
    # )

    # Remove similard docs
    ds["train"] = ds["train"].select(
        idx for idx in range(len(ds["train"])) if idx not in similar_docs
    )

    # Line-level deduplication
    hashes = get_duplicated_hashed_lines(ds["train"][0:1000]["text"])
    line_deduplicated_txts = remove_duplicated_lines(
        ds["train"][0:1000]["text"],
        to_remove=hashes,
    )

    print("Done !")
