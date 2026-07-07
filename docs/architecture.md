# Architecture

This project is organized as a local terminal pipeline. Each package owns one stage of the workflow and communicates with neighboring stages through plain Python dataclasses or pandas DataFrames.

```text
data/resumes/*.pdf
    -> parser/pdf_extractor.py
    -> parser/resume_parser.py
    -> parser/schemas.py::Resume

data/job_descriptions/*.txt
    -> parser/job_description_parser.py
    -> JobDescription

Resume + JobDescription
    -> nlp/embeddings.py
    -> nlp/similarity.py
    -> ml/features.py
    -> pandas feature vector

Synthetic feature vectors
    -> ml/synthetic_data.py
    -> ml/train_model.py
    -> models/model.pkl

Real candidate feature vectors
    -> ml/ranker.py
    -> predict_candidates.py
    -> output/rankings.csv
```

## Runtime Entry Points

- `python main.py train`: generates or loads synthetic data, trains the model, saves `models/model.pkl`, and writes feature importance.
- `python main.py predict`: ranks resumes for one job description and saves `output/rankings.csv`.
- `python main.py demo`: runs the prediction workflow, training first only if the model is missing.
- `python main.py verify`: prints feature vectors across all sample resume/job-description pairs.

## Data Flow

The parser layer produces structured text-derived objects. The NLP layer produces semantic similarity. The feature layer converts structured and semantic signals into model-ready numerical columns. The ML layer trains and applies the ranking model.

