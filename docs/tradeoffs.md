# Tradeoffs

## Synthetic Labels vs Real Hiring Data

Synthetic labels make the project reproducible and self-contained, but they reflect the scoring formula rather than real-world hiring outcomes.

## Regex Parsing vs Full Document Understanding

Regex and section heuristics are fast, explainable, and easy to test. They can struggle with highly visual resumes, unusual section names, or scanned PDFs.

## Compact Embedding Model vs Larger Model

MiniLM is fast and local-friendly. Larger embedding models may improve semantic matching, but they would be slower and heavier for a terminal assessment project.

## Random Forest vs Boosted Trees

Random Forest is reliable, simple, and interpretable enough for the project. Gradient boosting could improve accuracy but would add more tuning complexity.

## Local Execution vs Cloud Services

Local execution keeps candidate data private and avoids external APIs. The tradeoff is that users must install dependencies locally and may need internet access for first-time model downloads.

