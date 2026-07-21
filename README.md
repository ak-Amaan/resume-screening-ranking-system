# Resume Screening Agent Using NLP and Machine Learning

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![NLP](https://img.shields.io/badge/NLP-Sentence%20Transformers-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Project Overview

Recruiters and hiring teams often need to screen many PDF resumes against a specific job description. Job seekers also need a clear way to understand how a single resume aligns with a target role. Manual review is slow, keyword-only screening is brittle, and pure semantic matching can miss important structured signals such as education, certifications, projects, and years of experience.

This project solves that problem with a local AI/ML pipeline that parses PDF resumes, extracts candidate attributes, computes NLP-based semantic similarity, engineers interpretable ranking features, trains a machine learning model, outputs ranked candidate lists, and generates deterministic single-resume review reports.

The goal is to demonstrate a production-quality terminal application for resume screening, ranking, and review using NLP and machine learning. Recruiter mode produces `output/rankings.csv`, containing candidates sorted by predicted suitability for a selected job description. Review mode produces `output/review_report.txt`, containing a professional recruiter-style assessment for one resume and one job description.

## Features

- ✔ Resume Ranking
- ✔ Resume Review Mode
- ✔ Deterministic Recruiter Reasoning
- ✔ Candidate Recommendation
- ✔ Feature Engineering
- ✔ NLP Similarity
- ✔ Random Forest Ranking
- ✔ Resume Parsing
- ✔ Candidate Scoring
- ✔ Automated Testing
- ✔ CLI Interface
- ✔ Professional Documentation

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Folder Structure](#folder-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [First Time Setup](#first-time-setup)
- [Running the Project](#running-the-project)
- [Resume Review Report](#resume-review-report)
- [Project Workflow](#project-workflow)
- [Pipeline Explanation](#pipeline-explanation)
- [Machine Learning](#machine-learning)
- [Evaluation](#evaluation)
- [Sample Outputs](#sample-outputs)
- [Trade-offs](#trade-offs)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Architecture

```mermaid
flowchart TD
    A[PDF Resume] --> B[Text Extraction]
    B --> C[Resume Parsing]
    C --> D[Feature Engineering]
    E[Job Description] --> F[Job Description Parsing]
    F --> D
    C --> G[Sentence Embeddings]
    F --> G
    G --> H[Cosine Similarity]
    H --> D
    D --> I[Random Forest]
    I --> J[Candidate Ranking]
    J --> K[rankings.csv]
```

## Folder Structure

```text
resume-screening-ranking-system/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-lock.txt
├── pyproject.toml
├── main.py
├── predict_candidates.py
├── verify_features.py
├── Demo.ipynb
├── config.py
├── data/
│   ├── resumes/
│   ├── job_descriptions/
│   └── generated/
├── docs/
│   ├── architecture.md
│   ├── design_decisions.md
│   └── tradeoffs.md
├── models/
├── output/
├── parser/
├── nlp/
├── ml/
├── utils/
├── tests/
└── notebooks/
```

## Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python 3.13 |
| Libraries | NumPy, Pandas, Joblib, tqdm |
| ML | scikit-learn, RandomForestRegressor |
| NLP | spaCy, Sentence Transformers, all-MiniLM-L6-v2 |
| Testing | pytest |
| Utilities | PyMuPDF, argparse, logging |

## Installation

```bash
git clone https://github.com/ak-Amaan/resume-screening-ranking-system.git
cd resume-screening-ranking-system
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For exact reproducibility with the current development environment:

```bash
pip install -r requirements-lock.txt
```

## First Time Setup

The project uses the Sentence Transformer model `sentence-transformers/all-MiniLM-L6-v2` for semantic matching. On first execution, Sentence Transformers automatically downloads this model once to the local Hugging Face cache. The model is approximately 90 MB.

After the model is cached, the project runs completely offline from local files. No external APIs are used for parsing, scoring, training, or ranking. No resume data, job description data, or user data leaves the local machine.

## Running the Project

Train the Random Forest ranking model and save `models/model.pkl`:

```bash
python main.py train
```

Rank all PDF resumes in `data/resumes/` against the default job description and write `output/rankings.csv`:

```bash
python main.py predict
```

Run a compact end-to-end demonstration using the sample data:

```bash
python main.py demo
```

Verify feature vectors across all sample resumes and job descriptions:

```bash
python main.py verify
```

Review one PDF resume against one TXT job description and write `output/review_report.txt`:

```bash
python main.py review \
    --resume path/to/resume.pdf \
    --jd path/to/job_description.txt
```

Run the complete automated test suite:

```bash
pytest
```

Use a specific job description:

```bash
python main.py predict --job-description data/job_descriptions/machine_learning_engineer.txt
```

## Resume Review Report

Review mode evaluates a single PDF resume against a single TXT job description. It reuses the same PDF extraction, resume parsing, job description parsing, feature engineering, NLP similarity, and Random Forest scoring pipeline as the ranking workflow, but it does not rank candidates.

The generated `output/review_report.txt` reads like a professional recruiter or ATS report. It includes the candidate name, job title, overall score, deterministic overall assessment, strengths, weaknesses, matched skills, missing skills, detailed feature scores, practical recommendations, and an interview recommendation.

The report explanations are generated deterministically from computed feature values and parsed skill matches. No LLM is required.

## Project Workflow

### Recruiter Workflow

Recruiters run `python main.py predict` to compare multiple PDF resumes against one job description. The system extracts resume text, parses candidate fields, parses the TXT job description, computes feature vectors and NLP similarity scores, predicts candidate suitability with the Random Forest model, and writes sorted results to `output/rankings.csv`.

### Job Seeker Workflow

Job seekers run `python main.py review --resume path/to/resume.pdf --jd path/to/job_description.txt` to evaluate one resume against one target role. The system computes the same feature values used for ranking, then generates `output/review_report.txt` with an overall score, strengths, weaknesses, matched skills, missing skills, recommendations, and an interview recommendation.

## Pipeline Explanation

### Resume Parsing

PDF resumes are read with PyMuPDF. The parser extracts clean text across pages, then identifies candidate fields such as name, email, phone number, skills, education, certifications, projects, experience lines, and years of experience.

### Feature Extraction

Structured resume fields and parsed job description fields are converted into measurable candidate-job signals. These include skill overlap, education match, certification match, project relevance, programming language match, framework match, tools match, and years of experience difference.

### Semantic Similarity

The resume and job description are embedded using `sentence-transformers/all-MiniLM-L6-v2`. Cosine similarity is computed and clipped into the range `0.0` to `1.0`.

### Feature Engineering

The system combines semantic similarity with structured feature scores into a single pandas DataFrame. This feature vector is the input to the machine learning ranking model.

### Machine Learning

The Random Forest Regressor predicts a candidate suitability score from engineered features. It is trained on synthetic candidate-job samples generated locally.

### Candidate Ranking

The trained model predicts candidate scores, estimates confidence from Random Forest tree variance, sorts candidates in descending order, and writes the final ranking to `output/rankings.csv`.

### Recruiter Mode

Recruiter mode uses `python main.py predict` to rank multiple resumes against one job description. It parses every PDF resume in the selected resume directory, computes feature vectors for each candidate, predicts suitability scores with the Random Forest model, sorts candidates by score, and saves the final table to `output/rankings.csv`.

## Machine Learning

Random Forest was selected because it performs well on tabular engineered features, handles non-linear interactions, trains quickly on a laptop, provides feature importances, and can be saved cleanly with Joblib.

Synthetic data was used because no labelled resume-ranking dataset was provided with the assessment. Instead of downloading external datasets, the project generates realistic candidate-job feature vectors programmatically and labels them with a transparent weighted formula:

```text
Candidate Score =
  0.40 * Semantic Similarity
+ 0.20 * Skill Match
+ 0.15 * Experience Match
+ 0.10 * Education Match
+ 0.05 * Certification Match
+ 0.05 * Programming Language Match
+ 0.03 * Framework Match
+ 0.02 * Tools Match
```

Scores are normalized to `0-100`.

## Evaluation

Latest model evaluation:

```text
MAE: 2.9248
RMSE: 3.7910
R² Score: 0.9405
5-Fold Cross Validation RMSE: 4.3016 +/- 0.4375
```

| Metric | Meaning |
| --- | --- |
| MAE | Mean Absolute Error; average absolute difference between predicted and target scores. Lower is better. |
| RMSE | Root Mean Squared Error; penalizes larger prediction errors more strongly. Lower is better. |
| R² | Proportion of target-score variance explained by the model. Higher is better. |
| Cross Validation | 5-fold validation estimates how stable model performance is across different train/test splits. |

## Sample Outputs

Sample `output/review_report.txt`:

```text
Resume Review Report
=========================================================

Candidate Name
Aisha Khan

Job Title
Data Scientist

Overall Candidate Score
76.01

Overall Assessment
Strong Match

=========================================================

Section 1
Summary
The candidate is a strong match with an overall score of 76.01. Semantic similarity is moderate at 66.99%, and skill match is 90.91%. Matched skills include Python, SQL, Machine Learning, and 7 more. No required skill gaps were identified from the parsed data.

Section 2
Strengths
- Good semantic similarity with the job description
- Strong match against required and preferred skills
- High experience match for the role
- Relevant education for the role
- Good programming language match
- Good framework match

Section 3
Weaknesses
- Relevant certification match was not identified

Section 4
Matched Skills
- Python
- SQL
- Machine Learning
- Pandas
- NumPy
- scikit-learn
- Statistics
- NLP
- AWS
- Tableau

Section 5
Missing Skills
None

Section 6
Detailed Scores
Semantic Similarity: 66.99%
Skill Match: 90.91%
Experience Match: 100.00%
Education Match: 100.00%
Certification Match: 0.00%
Programming Languages: 100.00%
Frameworks: 100.00%
Tools: 66.67%
Project Relevance: 53.85%

Section 7
Recommendations
- Highlight projects that directly use the role's key skills.

Section 8
Interview Recommendation
Recommended
```

```text
Candidate Rankings
 Rank Candidate Name  Similarity Score  Predicted Score
    1     Aisha Khan            0.6699            76.01
    2 Maria Gonzalez            0.5689            61.04
    3    Rohan Mehta            0.5275            54.43
    4     Nina Patel            0.4799            50.67
    5  Liam O'Connor            0.5665            47.41
```

Sample `output/rankings.csv`:

```csv
Rank,Candidate Name,Similarity Score,Predicted Score
1,Aisha Khan,0.6699,76.01
2,Maria Gonzalez,0.5689,61.04
3,Rohan Mehta,0.5275,54.43
4,Nina Patel,0.4799,50.67
5,Liam O'Connor,0.5665,47.41
```

## Trade-offs

### Advantages

- Fully local terminal workflow after first model download.
- Transparent feature engineering and scoring formula.
- Deterministic explanations generated from computed feature values.
- No LLM is required for ranking or review reports.
- No Kaggle or external resume datasets required.
- Interpretable Random Forest feature importances.
- Clear separation between parsing, NLP, feature engineering, training, and ranking.
- PDF resumes are supported.

### Limitations

- Synthetic labels are useful for assessment reproducibility but do not replace real hiring feedback.
- Regex and dictionary-based parsing may miss unusual resume layouts or uncommon skills.
- Scanned PDFs require OCR, which is not currently implemented.
- TXT job descriptions are currently supported.
- Embeddings are computed per resume-job pair rather than fully batched.

### Future Improvements

- OCR support for image-only PDF resumes.
- Better handling of scanned resumes and non-standard formatting.
- PDF Job Descriptions.
- DOCX Job Descriptions.
- One Resume vs Many JDs.
- ATS Keyword Optimization.
- Optional LLM Explanation.
- Batch Processing Improvements.
- Skill Synonym Matching.
- Batch embedding optimisation and embedding caching.
- More ranking models and model comparison reports.
- Fine-tuned embeddings for recruiting-specific language.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

This project uses and acknowledges:

- Sentence Transformers
- spaCy
- scikit-learn
- PyMuPDF
