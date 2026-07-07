# Resume Screening and Ranking System Using NLP and Machine Learning

## Overview

This project is a terminal-only AI/ML pipeline for screening PDF resumes against job descriptions. It will extract resume text, parse candidate attributes, compute semantic similarity with Sentence Transformers, engineer ranking features, and train a Random Forest Regressor on synthetic resume/job-description data.

This repository is currently at **Step 1 only**. The full project structure, dependency files, and module placeholders are present, but the implementation has intentionally not started yet.

## Architecture

The planned pipeline follows the assessment flow:

```text
PDF Resume
    -> Text Extraction
    -> Resume Parsing
    -> Feature Extraction
    -> Sentence Embeddings
    -> Cosine Similarity
    -> Feature Engineering
    -> Random Forest Regressor
    -> Candidate Score
    -> Ranking
    -> rankings.csv
```

Core responsibilities are separated by package:

- `parser/`: PDF text extraction and structured resume parsing.
- `nlp/`: Sentence Transformer embeddings and cosine similarity scoring.
- `ml/`: synthetic data generation, feature vector construction, model training, and candidate ranking.
- `utils/`: shared logging, file handling, and validation helpers.
- `data/`: local input resumes, job descriptions, and generated synthetic training data.
- `models/`: serialized trained model artifacts.
- `output/`: generated ranking reports such as `rankings.csv`.
- `tests/`: pytest test modules for parser, NLP, ML, and utility behavior.

## Folder Structure

```text
resume-screening-ranking-system/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── main.py
├── config.py
├── data/
│   ├── resumes/
│   ├── job_descriptions/
│   └── generated/
├── models/
├── output/
├── parser/
├── nlp/
├── ml/
├── utils/
├── tests/
└── notebooks/
```

## Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The implementation will use `sentence-transformers/all-MiniLM-L6-v2` for semantic embeddings. A spaCy language model installation command will be added during the implementation step once the parser code is introduced.

## Usage

Usage commands will be added after implementation. The intended final entry point is:

```bash
python main.py
```

The final application will write:

```text
output/rankings.csv
```

with the columns:

```text
Resume Name, Similarity Score, Predicted Candidate Score, Rank
```

## Model Choice

The planned ranking model is a `RandomForestRegressor` from scikit-learn. This model is suitable for the assessment because it handles non-linear interactions between structured resume features, semantic similarity scores, skills overlap, education indicators, and years of experience without requiring a large real-world dataset.

Sentence embeddings will be generated with `sentence-transformers/all-MiniLM-L6-v2`, a compact model that balances semantic quality, speed, and local execution.

## Tradeoffs

- Synthetic training data keeps the project self-contained and avoids external dataset dependencies, but it may not fully represent real hiring data distributions.
- A Random Forest model is interpretable enough for an assessment and robust on tabular features, but it will not learn deep cross-document semantics by itself.
- Cosine similarity provides a strong semantic baseline, while feature engineering adds domain-specific ranking signals.
- Local-only execution improves reproducibility and privacy, but first-time model downloads may be required when dependencies are installed and models are loaded.

## Future Improvements

- Add robust PDF parsing with PyMuPDF.
- Add spaCy-assisted name, education, certification, and experience parsing.
- Generate realistic synthetic resumes and job descriptions.
- Train and evaluate the Random Forest ranking model.
- Add pytest coverage for parser, NLP, feature engineering, and ranking behavior.
- Add model evaluation reports and feature importance summaries.

## Example Output

The final `rankings.csv` will follow this shape:

| Resume Name | Similarity Score | Predicted Candidate Score | Rank |
| --- | ---: | ---: | ---: |
| resume_001.pdf | 0.87 | 91.4 | 1 |
| resume_002.pdf | 0.79 | 84.2 | 2 |
| resume_003.pdf | 0.64 | 72.8 | 3 |

