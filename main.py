import polars as pl

from min_hash import find_similar_texts

if __name__ == "__main__":
    # Load data
    splits = {
        "train": "data/0124/toxic-chat_annotation_train.csv",
        "test": "data/0124/toxic-chat_annotation_test.csv",
    }
    df = pl.read_csv("hf://datasets/lmsys/toxic-chat/" + splits["train"])

    data = df["user_input"].to_list()

    # MinHashLSH
    similar_docs = find_similar_texts(data, threshold=0.9)
    similar_docs = sorted(
        similar_docs,
        key=lambda d: d["similarity"],
        reverse=True,
    )
    print("Done !")
