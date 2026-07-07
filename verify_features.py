"""Verification script for Step 3 NLP feature engineering."""

from __future__ import annotations

import logging
from pathlib import Path

from ml.features import create_feature_vector
from parser.job_description_parser import parse_job_description
from parser.pdf_extractor import PDFExtractionError, extract_text_from_pdf
from parser.resume_parser import parse_resume

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Load resumes and job descriptions, then print feature vectors."""
    resume_paths = sorted(Path("data/resumes").glob("*.pdf"))
    job_description_paths = sorted(Path("data/job_descriptions").glob("*.txt"))

    if not resume_paths:
        logger.warning("No PDF resumes found in data/resumes.")
        return
    if not job_description_paths:
        logger.warning("No job descriptions found in data/job_descriptions.")
        return

    for job_path in job_description_paths:
        job_description = parse_job_description(job_path.read_text(encoding="utf-8"))
        print(f"\n## Job Description: {job_path.name} ({job_description.title})")

        for resume_path in resume_paths:
            try:
                resume = parse_resume(extract_text_from_pdf(resume_path))
            except PDFExtractionError as exc:
                logger.error("Skipping %s: %s", resume_path.name, exc)
                continue

            feature_vector = create_feature_vector(resume, job_description)
            row = feature_vector.iloc[0]
            print(f"\nResume Name: {resume_path.name}")
            print(f"Similarity Score: {row['semantic_similarity']:.4f}")
            print(f"Skill Match: {row['skill_match_percentage']:.4f}")
            print("Feature Vector:")
            print(feature_vector.to_string(index=False))


if __name__ == "__main__":
    main()

