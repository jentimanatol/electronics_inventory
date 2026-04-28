# Inventory AI Assistant — Project Outline

## Problem
A personal electronics inventory grows quickly, and searching by exact text is not enough. The project adds an AI assistant that answers questions such as "Do I have sensors?", "What should I restock?", and "What parts can I use for a smart garden build?"

## Baseline
The baseline is normal keyword search over SQLite inventory rows.

## Method
The deployed method is a private RAG-style workflow:
1. Convert SQLite inventory rows into text documents.
2. Tokenize and normalize terms.
3. Retrieve relevant rows with TF-IDF / cosine similarity.
4. Generate a grounded answer only from retrieved rows.
5. Show retrieved evidence so the answer is explainable.

The optional class component is a tiny decoder-only Transformer in `app/ai_transformer_decoder.py`.
The additional `app/ai_decoder_lab.py` page logic demonstrates greedy decoding, beam search, temperature/top-k/top-p sampling, repetition penalty, KV-cache intuition, continuous batching, quantization estimates, and model-cascade routing.

## Evaluation
- Retrieval precision@k
- Correct no-match behavior
- Low-stock warning correctness
- Duplicate detection correctness
- Error taxonomy: no-match, wrong component type, synonym failure, plural/singular failure, stale database row
- Ablations: keyword baseline vs synonym-expanded retrieval; greedy vs sampling output

## Responsible AI
- No paid API required
- No inventory data leaves the app
- The assistant refuses to invent parts when no evidence is retrieved
- Credentials and database backups must not be committed to GitHub
