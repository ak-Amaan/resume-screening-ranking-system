"""Predict and rank resume candidates for one job description."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ml.features import create_feature_vector
from ml.ranker import rank_candidates
from parser.job_description_parser import parse_job_description
from parser.pdf_extractor import PDFExtractionError, extract_text_from_pdf
from parser.resume_parser import parse_resume

DEFAULT_RESUME_DIR = Path("data/resumes")
DEFAULT_JOB_DESCRIPTION = Path("data/job_descriptions/data_scientist.txt")
DEFAULT_OUTPUT_PATH = Path("output/rankings.csv")

logger = logging.getLogger(__name__)


def predict_candidates(
    resume_dir: str | Path = DEFAULT_RESUME_DIR,
    job_description_path: str | Path = DEFAULT_JOB_DESCRIPTION,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Predict candidate scores for all PDF resumes in a directory.

    Args:
        resume_dir: Directory containing PDF resumes.
        job_description_path: Path to one job description text file.
        output_path: Destination CSV path for rankings.

    Returns:
        Ranked candidate DataFrame.
    """
    resume_paths = sorted(Path(resume_dir).glob("*.pdf"))
    if not resume_paths:
        raise FileNotFoundError(f"No PDF resumes found in: {resume_dir}")

    jd_path = Path(job_description_path)
    if not jd_path.exists():
        raise FileNotFoundError(f"Job description not found: {jd_path}")

    job_description = parse_job_description(jd_path.read_text(encoding="utf-8"))
    candidate_rows: list[pd.DataFrame] = []

    for resume_path in resume_paths:
        try:
            resume = parse_resume(extract_text_from_pdf(resume_path))
        except PDFExtractionError as exc:
            logger.error("Skipping %s: %s", resume_path.name, exc)
            continue

        feature_vector = create_feature_vector(resume, job_description)
        feature_vector.insert(0, "Candidate Name", resume.name or resume_path.stem)
        feature_vector.insert(1, "Resume File", resume_path.name)
        feature_vector.insert(2, "Similarity Score", feature_vector["semantic_similarity"])
        candidate_rows.append(feature_vector)

    if not candidate_rows:
        raise ValueError("No valid candidate feature vectors were generated.")

    candidate_features = pd.concat(candidate_rows, ignore_index=True)
    ranked_candidates = rank_candidates(candidate_features)

    output_columns = ["Rank", "Candidate Name", "Similarity Score", "Predicted Score"]
    rankings = ranked_candidates[output_columns].copy()
    rankings["Similarity Score"] = rankings["Similarity Score"].round(4)
    rankings["Predicted Score"] = rankings["Predicted Score"].round(2)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output, index=False)
    logger.info("Saved candidate rankings to %s", output)

    print("\nCandidate Rankings")
    print(rankings.to_string(index=False))
    return rankings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank resume candidates for one job.")
    parser.add_argument("--resume-dir", default=str(DEFAULT_RESUME_DIR))
    parser.add_argument("--job-description", default=str(DEFAULT_JOB_DESCRIPTION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    """Run candidate prediction from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    predict_candidates(
        resume_dir=args.resume_dir,
        job_description_path=args.job_description,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

