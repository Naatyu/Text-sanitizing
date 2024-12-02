import regex
from datasketch import MinHash, MinHashLSH
from tqdm.auto import tqdm


def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters
    compiled_pattern = regex.compile(r"[^\w\s\'-]")
    text = regex.sub(compiled_pattern, " ", text)

    # Remove extra whitespace
    compiled_pattern = regex.compile(r"\s+")
    text = regex.sub(compiled_pattern, " ", text)

    return text


def get_shingles(text: str, k: int = 3) -> set[str]:
    """Convert text into k-shingles (k-grams)."""
    text = preprocess_text(text)

    return {text[i : i + k] for i in range(len(text) - k + 1)}


def create_minhash(text: str, num_perm: int = 128, k: int = 3) -> MinHashLSH:
    """Create a MinHash object from text."""
    minhash = MinHash(num_perm=num_perm)
    shingles = get_shingles(text, k)
    for shingle in shingles:
        minhash.update(shingle.encode("utf-8"))

    return minhash


def find_similar_texts(
    data: list[str],
    threshold: float = 0.5,
    num_perm: int = 128,
    k: int = 3,
    return_only_idx: bool = False,
):
    # Initialize LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Store MinHash objects
    minhashes = {}

    # Add documents to LSH index TODO: multiprocess
    for idx, text in enumerate(tqdm(data, desc="Hashing")):
        minhash = create_minhash(text, num_perm=num_perm, k=k)
        minhashes[idx] = minhash
        lsh.insert(f"doc_{idx}", minhash)

    # Find similars pairs using LSH
    similar_pairs = []
    texts_to_remove = set()

    for idx, text in enumerate(tqdm(data, desc="Searching similarities")):
        if idx in texts_to_remove:
            continue

        query_minhash = minhashes[idx]
        similar_docs = lsh.query(query_minhash)

        # Process results
        for similar_doc in similar_docs:
            similar_idx = int(similar_doc.split("_")[1])
            # Only keep pairs where idx < similar_idx to avoid duplicates
            if idx < similar_idx:
                # Calculate actual similarity
                similarity = minhashes[idx].jaccard(minhashes[similar_idx])

                if similarity > threshold:
                    if return_only_idx:
                        texts_to_remove.add(similar_idx)
                        continue
                    similar_pairs.append(
                        {
                            "text1_index": idx,
                            "text2_index": similar_idx,
                            "text1": text,
                            "text2": data[similar_idx],
                            "similarity": similarity,
                        },
                    )

    if return_only_idx:
        return texts_to_remove

    return similar_pairs
