import hashlib
from collections import defaultdict

from tqdm.auto import tqdm


def normalize_line(line: str) -> str:
    """Normalize line of text."""
    return " ".join(line.strip().lower().split())


def hash_line(line: str) -> str:
    """Create sha256 hash of a line."""
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def process_document(doc: str) -> dict:
    """Normalize and hash lines of a document."""
    lines = doc.split("\n")
    return [hash_line(normalize_line(line)) for line in lines]


def count_lines_in_bucket(hashed_docs: list):
    line_counts = defaultdict(int)
    for doc in hashed_docs:
        for line_hash in doc:
            line_counts[line_hash] += 1
    return line_counts


def get_duplicated_hashed_lines(docs: list[str], threshold: int = 6) -> list:
    processed_docs = [
        process_document(doc) for doc in tqdm(docs, desc="Normalize and hash")
    ]
    line_counts = count_lines_in_bucket(processed_docs)

    return [k for k, v in line_counts.items() if v >= threshold]


def remove_duplicated_lines(docs: list[str], to_remove: list) -> list[str]:
    processed_docs = []
    removed_lines = 0
    for doc in tqdm(docs, desc="Removing lines"):
        cleaned_txt = []
        normalized_lines = [normalize_line(line) for line in doc.split("\n")]
        for line in normalized_lines:
            if hash_line(line) in to_remove:
                removed_lines += 1
                continue
            cleaned_txt.append(line)
        processed_docs.append("\n".join(cleaned_txt))

    print(f"Removed {removed_lines} lines.")

    return processed_docs
