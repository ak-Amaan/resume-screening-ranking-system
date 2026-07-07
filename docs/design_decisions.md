# Design Decisions

## Terminal-Only Application

The assessment requires no frontend, Flask, FastAPI, Streamlit, database, or external APIs. The project therefore exposes command-line workflows only.

## PDF-Only Resume Support

The parser supports PDF resumes only. PyMuPDF was selected because it is fast, widely used, and handles multi-page PDFs cleanly.

## Dataclasses for Parsed Objects

Resume-related structures use dataclasses to keep parsed data explicit, typed, and easy to serialize.

## MiniLM for Semantic Similarity

`sentence-transformers/all-MiniLM-L6-v2` balances speed and semantic quality. A singleton loader avoids repeated model initialization.

## Feature Engineering Before ML

The model does not consume raw text. Instead, it consumes interpretable features such as skill match, experience match, education match, and semantic similarity. This makes scoring easier to inspect and test.

## Synthetic Data

The project does not use Kaggle or downloaded datasets. Synthetic candidate-job samples are generated programmatically, then labeled by a documented weighted formula.

## Random Forest Regressor

Random Forest is a practical model for this assessment because it handles non-linear tabular relationships, trains quickly, supports feature importance, and requires minimal preprocessing.

