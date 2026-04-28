# General Inventory AI Assistant

A deployed general-purpose inventory web app with a private local AI assistant.  
The app started as an electronics inventory system and was upgraded into a broader inventory manager for electronics, tools, materials, books, documents, furniture, and other item categories.

Live deployment:

```text
https://electronicsinventory-production.up.railway.app/
```

AI page:

```text
https://electronicsinventory-production.up.railway.app/ai
```

---

## Project Summary

This project solves a practical inventory-management problem: a user may have many parts or objects stored in different boxes, drawers, or locations, but manually searching the inventory can become slow and error-prone.

The solution is a web-based inventory system with an AI assistant that can answer natural-language questions such as:

```text
Show all items
Show categories
Do I have resistors?
Do I have resistor in store?
Do we have sensors?
What should I restock?
What is my inventory worth?
Check duplicates
Find stepper motor driver parts
```

The AI assistant is designed to be private, lightweight, and deployable. It does not require a paid external API. The live production app uses grounded retrieval over the local SQLite inventory database, while optional Transformer/decoder files are included for class experimentation and documentation.

---

## Main Features

- Add, edit, delete, and search inventory items
- Upload item photos
- Generate QR labels
- Scan QR codes
- Backup and restore inventory database
- General inventory categories, not only electronics
- Unit price field for each item
- Total value calculation using:

```text
total value = quantity × unit price
```

- AI assistant page at `/ai`
- Natural-language inventory question handling
- Show all items
- Show categories/types
- Search for item names, types, tags, and locations
- Singular/plural normalization, for example:
  - resistor / resistors
  - capacitor / capacitors
  - sensor / sensors
  - category / categories
  - item / items
- Low-stock warnings
- Duplicate detection
- Inventory value estimate
- Retrieved evidence display so the answer is explainable

---

## AI Class Track

Primary track:

```text
Track D — Systems / Deployment
```

Secondary track:

```text
Track A — LLM Applications
```

The project fits the systems/deployment track because it focuses on building and deploying an AI-enabled application under realistic compute and hosting limits. It also includes LLM-style components such as retrieval-augmented generation, intent classification, decoder concepts, and hallucination/error analysis.

---

## Problem Framing

### Task

Build an AI-assisted inventory system that can answer user questions from a private inventory database.

### Why It Matters

Manual inventory search becomes inefficient when items are spread across many physical locations. A natural-language AI assistant makes the system easier to use because the user can ask questions directly instead of remembering exact search terms.

### Success Criterion

The system is successful if it can:

1. Correctly identify inventory-related user intent.
2. Retrieve relevant items from the database.
3. Handle singular and plural wording.
4. Show all items or categories when requested.
5. Warn about low stock.
6. Detect duplicate item groups.
7. Estimate total inventory value.
8. Avoid hallucinating items that are not in the database.
9. Show evidence rows used for the answer.

---

## Data Statement

### Dataset Source

The dataset is the user's own inventory stored in a local SQLite database.

The main data table contains item records such as:

- Item ID
- Category
- Type
- Value/model
- Normalized value
- Quantity
- Location
- Tags
- Notes
- Photo filename
- Unit price

### Data Size

The data size depends on how many inventory items the user adds. The system works with small personal inventories and can scale to larger local collections.

### Licensing and Terms

The inventory data is user-created private data. No external dataset is required for the production app.

### Privacy

Inventory records, prices, locations, and photos may be private. The app is designed so the AI assistant uses local database records and does not send inventory data to a paid third-party AI API.

### Preprocessing

The AI assistant preprocesses text by:

- Lowercasing text
- Tokenizing words
- Normalizing simple plurals
- Matching singular/plural variants
- Expanding common inventory terms
- Combining item fields into searchable text documents

Example:

```text
resistors → resistor
sensors → sensor
categories → category
```

### Split Plan

For the deployed app, there is no supervised train/validation/test split because the inventory data is live user data.

For evaluation, a small fixed test set of inventory questions can be created manually:

- Item lookup questions
- Category questions
- Low-stock questions
- Duplicate questions
- Inventory value questions
- No-match questions

### Leakage Risks

The main leakage risk is exposing private inventory data, database backups, uploaded photos, or credentials. These should not be committed to GitHub.

Recommended `.gitignore` entries:

```text
app/uploads/
*.db
*.sqlite
.env
__pycache__/
results/*.pt
```

---

## Baseline

The baseline is normal keyword search over the inventory database.

Example baseline behavior:

```text
Query: Do I have resistors?
Baseline: Search for exact word matches in item fields.
```

This baseline is appropriate because most inventory apps use basic search by category, name, tag, or location.

Limitations of the baseline:

- It may miss singular/plural variants.
- It may not understand intent.
- It may not know when the user asks for all items.
- It may not group categories.
- It may not provide low-stock or duplicate diagnostics.

---

## Method

The improved method combines four AI-style components:

1. Lightweight trained intent classifier
2. RAG-style retrieval over SQLite rows
3. Singular/plural and keyword normalization
4. Optional Transformer decoder and decoding/deployment lab files for class evidence

---

## AI Architecture

### 1. Intent Classifier

File:

```text
app/ai_intent_model.py
```

The intent model is a small local Naive Bayes style classifier trained from example inventory questions.

It recognizes intents such as:

```text
show_all
show_categories
search
low_stock
duplicates
inventory_value
```

Example supported questions:

```text
show all items
list everything
show categories
group by type
do I have resistors?
do I have resistor in store?
do we have sensors?
what should I restock?
duplicate check
what is my inventory worth?
```

This makes the assistant more reliable than pure keyword search.

---

### 2. RAG-Style Retrieval

The production AI assistant uses a private retrieval workflow:

1. Load rows from the SQLite `items` table.
2. Convert each row into a text document.
3. Normalize tokens and plural forms.
4. Rank rows by relevance.
5. Generate a grounded answer from the retrieved rows.
6. Display retrieved rows as evidence.

This is retrieval-augmented generation in a lightweight local form. The assistant does not invent inventory records; it answers from retrieved database rows.

---

### 3. Plural and Synonym Handling

The assistant handles simple singular/plural variants.

Examples:

```text
resistor = resistors
capacitor = capacitors
sensor = sensors
category = categories
tool = tools
item = items
```

This improves real user questions because users often type naturally instead of matching exact database terms.

---

### 4. Optional Transformer Decoder Prototype

File:

```text
app/ai_transformer_decoder.py
```

This file implements a tiny GPT-style decoder-only Transformer for offline experimentation.

It includes:

- Token embeddings
- Positional embeddings
- Causal self-attention
- Multi-head attention
- Feed-forward MLP
- Residual connections
- Layer normalization
- Next-character prediction objective

This file is optional and is not required for the live Railway deployment.

---

### 5. Decoder and Deployment Lab

File:

```text
app/ai_decoder_lab.py
```

This file demonstrates concepts from large language model decoding and deployment:

- Greedy decoding
- Beam search
- Temperature sampling
- Top-k sampling
- Top-p sampling
- Repetition penalty
- KV-cache speedup explanation
- Continuous batching idea
- Quantization memory estimate
- Simple model-cascade routing

This gives visible class evidence without making the production Railway app heavy.

---

## Why the Production App Does Not Load PyTorch

The deployed Railway app is intentionally kept lightweight.

Loading PyTorch or a Transformer model at startup can:

- Increase memory usage
- Slow deployment
- Increase cold-start time
- Cause Railway deployment failure on small resources

Therefore:

- Production uses fast local retrieval and intent classification.
- The Transformer decoder is included as an optional offline class experiment.
- The decoder lab is dependency-free and safe to show in the deployed UI.

This is a compute-realistic design choice.

---

## Training and Compute

### Production AI

The production assistant does not require GPU training.

Compute requirements:

```text
CPU only
SQLite database
FastAPI server
No paid API key
No GPU
```

### Intent Classifier

The lightweight intent classifier is trained from local example phrases and runs instantly on CPU.

### Optional Transformer Decoder

The optional Transformer decoder can be trained locally on CPU for demonstration.

Recommended hardware:

```text
CPU laptop or desktop
Python 3.11+
Optional PyTorch install
Small inventory text corpus
```

Example training command:

```bash
python app/ai_transformer_decoder.py --db app/uploads/inventory.db --epochs 10 --out results/inventory_decoder.pt
```

Example generation command:

```bash
python app/ai_transformer_decoder.py --db app/uploads/inventory.db --load results/inventory_decoder.pt --prompt "question: what sensors do I have? answer:" --generate
```

Fallback choice:

If full Transformer training is too slow or unnecessary, use the deployed RAG assistant and decoder lab as the core reproducible system.

---

## Evaluation Plan

### Primary Metrics

Recommended metrics:

1. Intent accuracy  
   Measures whether the system classifies the question correctly.

2. Retrieval precision@k  
   Measures whether the correct inventory row appears in the top retrieved rows.

3. No-match accuracy  
   Measures whether the assistant correctly says it cannot find an item.

4. Low-stock accuracy  
   Measures whether warnings match the rule:

```text
quantity <= 2
```

5. Duplicate detection accuracy  
   Measures whether duplicate groups match category/type/value groups.

6. Inventory value correctness  
   Checks:

```text
sum(quantity × price)
```

---

## Diagnostic Analysis

Recommended diagnostic question groups:

| Test Type | Example Question | Expected Behavior |
|---|---|---|
| Show all | `show all items` | List all records |
| Category summary | `show categories` | Group items by category/type |
| Singular search | `do I have resistor?` | Match resistor records |
| Plural search | `do I have resistors?` | Match same resistor records |
| Low stock | `what should I restock?` | Show low quantity items |
| Duplicate check | `check duplicates` | Show duplicate groups if present |
| Value | `what is my inventory worth?` | Sum quantity × price |
| No match | `do I have oscilloscope?` | Refuse if no row exists |

---

## Ablation Ideas

Recommended ablations for the final report:

### Ablation 1: Keyword Baseline vs Normalized Retrieval

Compare exact keyword search against normalized retrieval with singular/plural handling.

Expected improvement:

```text
"resistor" and "resistors" should retrieve the same item group.
```

### Ablation 2: RAG Only vs Intent + RAG

Compare retrieval alone against intent classification plus retrieval.

Expected improvement:

```text
"show all items" and "show categories" should trigger structured answers instead of normal item search.
```

### Ablation 3: Production RAG vs Optional Decoder

Compare the production grounded assistant with the optional Transformer decoder.

Expected result:

```text
RAG is more reliable for factual inventory answers.
Decoder is useful for class demonstration but can hallucinate if trained on too little data.
```

---

## Responsible AI

### Privacy

Inventory data may contain private item names, locations, photos, and prices.

Mitigation:

- No paid external AI API is required.
- Local SQLite records are used as the source of truth.
- Do not commit database files, backups, photos, or secrets to GitHub.

### Hallucination

A general text generator may invent items.

Mitigation:

- The production assistant answers only from retrieved database rows.
- Evidence rows are displayed.
- If no strong match is found, the system says it cannot find a strong match.

### Bias

This project does not classify people, but bias can still appear in assumptions about item importance or categories.

Mitigation:

- The assistant uses the user's own category labels.
- The user can edit item categories and tags.

### Robustness

Possible failure cases:

- Misspelled item names
- Very short queries
- Missing tags
- Inconsistent category names
- Old records with missing price

Mitigation:

- Use plural normalization
- Show retrieved evidence
- Keep price default as `$0.00`
- Allow manual correction through edit forms

---

## Project Structure

Recommended repository structure:

```text
.
├── app/
│   ├── main.py
│   ├── ai_intent_model.py
│   ├── ai_transformer_decoder.py
│   ├── ai_decoder_lab.py
│   ├── templates/
│   │   ├── ai.html
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── item.html
│   │   ├── item_detail.html
│   │   ├── labels.html
│   │   ├── scan.html
│   │   ├── backup.html
│   │   └── login.html
│   ├── static/
│   │   └── style.css
│   └── uploads/
├── configs/
│   └── ai_decoder_small.json
├── scripts/
│   └── run_ai_decoder_lab.py
├── results/
│   └── README.md
├── report/
│   └── AI_PROJECT_OUTLINE.md
├── requirements.txt
├── requirements-ai.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jentimanatol/electronics_inventory.git
cd electronics_inventory
```

If your repository name is different, replace the URL and folder name.

---

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Production Requirements

```bash
pip install -r requirements.txt
```

Optional AI experiment requirements:

```bash
pip install -r requirements-ai.txt
```

Only install `requirements-ai.txt` locally. It is not required for Railway.

---

## Run Locally

Start the app:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Open AI assistant:

```text
http://127.0.0.1:8000/ai
```

---

## Run Decoder Lab

Dependency-free decoder/deployment lab:

```bash
python scripts/run_ai_decoder_lab.py --db app/uploads/inventory.db --question "do we have any sensors?"
```

---

## Run Optional Transformer Decoder

Install optional dependencies first:

```bash
pip install -r requirements-ai.txt
```

Train:

```bash
python app/ai_transformer_decoder.py --db app/uploads/inventory.db --epochs 10 --out results/inventory_decoder.pt
```

Generate:

```bash
python app/ai_transformer_decoder.py --db app/uploads/inventory.db --load results/inventory_decoder.pt --prompt "question: what sensors do I have? answer:" --generate
```

---

## Reproducibility

To reproduce the core results:

1. Install requirements.
2. Run the app locally.
3. Add several inventory records.
4. Include singular and plural item examples, such as:
   - resistor
   - resistors
   - capacitor
   - capacitors
   - sensor
   - sensors
5. Add prices and quantities.
6. Open `/ai`.
7. Test the example questions.

Suggested reproducibility questions:

```text
show all items
show categories
do I have resistor?
do I have resistors?
do we have sensors?
what should I restock?
what is my inventory worth?
check duplicates
```

Expected outputs:

- Structured list for all items
- Category/type summary
- Same matching behavior for singular and plural forms
- Low-stock warning for quantity <= 2
- Total inventory value using quantity × price
- Evidence rows displayed for retrieved items

Random seed:

```text
Recommended seed for experiments: 42
```

---

## Railway Deployment Notes

The app is deployed on Railway.

Recommended production command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Important deployment note:

Do not install heavy optional AI dependencies on Railway unless necessary. The production app is designed to work with `requirements.txt` only.

---

## GitHub Safety Checklist

Do not commit:

```text
.env
*.db
*.sqlite
app/uploads/
database backups
private item photos
Railway credentials
results/*.pt
```

Safe to commit:

```text
source code
templates
static CSS
configs
scripts
README.md
report outline
results/README.md
```

---

## Example AI Questions

```text
Show all items
Show categories
Do I have resistors?
Do I have resistor in store?
Do we have any sensors?
Find Arduino parts
Find stepper motor drivers
What should I restock?
What is my inventory worth?
Check duplicates
Which items are most expensive?
```

---

## Limitations

- The production assistant is not a full large language model.
- The optional Transformer decoder is small and mainly for learning/demo purposes.
- The assistant depends on the quality of item names, categories, tags, and notes.
- Very misspelled queries may still fail.
- Items with missing prices are counted as `$0.00`.
- The deployed system is optimized for privacy and stability, not maximum language fluency.

---

## Next Steps

Possible future improvements:

- Add fuzzy spelling correction
- Add item recommendation by project type
- Add CSV export/import
- Add charts for inventory value by category
- Add model-card documentation
- Add a larger fixed evaluation set
- Add voice input for inventory questions
- Add barcode support
- Add more robust duplicate detection
- Add user-defined category aliases

---

## Author

Anatolie Jentimir

Project type:

```text
AI-assisted deployed inventory management system
```

Course focus:

```text
Advanced Topics in Deep Learning, Large Language Models, and Large Vision Models
```
