"""Command-line entry point for the resume screening and ranking pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml.train_model import DEFAULT_MODEL_PATH, train_candidate_ranking_model
from predict_candidates import (
    DEFAULT_JOB_DESCRIPTION,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RESUME_DIR,
    predict_candidates,
)
from review_candidate import review_candidate
from utils.logging_config import configure_logging
from verify_features import main as verify_features_main

logger = logging.getLogger(__name__)


def train_command(_: argparse.Namespace) -> None:
    """Train the Random Forest candidate ranking model."""
    train_candidate_ranking_model()


def predict_command(args: argparse.Namespace) -> None:
    """Predict and rank candidates for one job description."""
    predict_candidates(
        resume_dir=args.resume_dir,
        job_description_path=args.job_description,
        output_path=args.output,
    )


def demo_command(args: argparse.Namespace) -> None:
    """Run a compact end-to-end demo using existing sample data."""
    if not Path(DEFAULT_MODEL_PATH).exists():
        logger.info("No trained model found. Training a model first.")
        train_candidate_ranking_model()

    predict_candidates(
        resume_dir=args.resume_dir,
        job_description_path=args.job_description,
        output_path=args.output,
    )
    logger.info("Demo complete. Rankings written to %s", args.output)


def verify_command(_: argparse.Namespace) -> None:
    """Run feature-vector verification across resumes and job descriptions."""
    verify_features_main()


def review_command(args: argparse.Namespace) -> None:
    """Generate a professional review report for one resume and one job."""
    try:
        review_candidate(resume_path=args.resume, job_description_path=args.jd)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""
    parser = argparse.ArgumentParser(
        description="Resume screening and candidate ranking system.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save the model.")
    train_parser.set_defaults(func=train_command)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate candidate rankings for one job description.",
    )
    predict_parser.add_argument("--resume-dir", default=str(DEFAULT_RESUME_DIR))
    predict_parser.add_argument("--job-description", default=str(DEFAULT_JOB_DESCRIPTION))
    predict_parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    predict_parser.set_defaults(func=predict_command)

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a compact training-aware prediction demo.",
    )
    demo_parser.add_argument("--resume-dir", default=str(DEFAULT_RESUME_DIR))
    demo_parser.add_argument("--job-description", default=str(DEFAULT_JOB_DESCRIPTION))
    demo_parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    demo_parser.set_defaults(func=demo_command)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Print feature vectors for all sample resume/job pairs.",
    )
    verify_parser.set_defaults(func=verify_command)

    review_parser = subparsers.add_parser(
        "review",
        help="Review one resume against one job description.",
    )
    review_parser.add_argument("--resume", required=True)
    review_parser.add_argument("--jd", required=True)
    review_parser.set_defaults(func=review_command)
    return parser


def main() -> None:
    """Run the requested command-line workflow."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
