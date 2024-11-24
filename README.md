# Pipeline to santize text data

Here are the different steps (from Llama 3.1 paper https://arxiv.org/pdf/2407.21783)

## De-duplication

If we scrap from web, we can start the deduplication by removing similar urls. We are not interested in this here.

### Document-level de-duplication

The first process that we will make is to de-duplicate documents. For that we will also use the MinHash. This will allow to remove near duplicate documents. MinHash allow to quickly estimate how similar two sets are. 

