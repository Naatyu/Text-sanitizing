# Pipeline to santize text data

Here are the different steps (from Llama 3.1 paper https://arxiv.org/pdf/2407.21783). The goal is to implement a full pipeline to sanitize a text dataset from duplication and poor content.

The project is managed with `uv`.

## Entropy

This is not in the original paper but I wanted to implement it, plus it's a good first filtering to remove anything that is not language.
For that we can use the shannon entropy.

## De-duplication

If we scrap from web, we can start the deduplication by removing similar urls. We are not interested in this here.

### Document-level de-duplication

The first process that we will make is to de-duplicate documents. For that we will also use the MinHash. This will allow to remove near duplicate documents. MinHash allow to quickly estimate how similar two sets are. 

### Line-level de-duplication

They removed lines that appeared more than 6 times in a bucket in 30M documents. We don't know exactly how it was done but we can assume that they normalized lines and then hash it, and remove similar hashes.
We can also wonder what is called a line ? it seems that it is not a line that end with a dot but more like a paragraph that ends with `\n`. It is more like paragraph de-duplication than line de-duplication.