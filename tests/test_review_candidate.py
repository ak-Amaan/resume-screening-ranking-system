"""Tests for single-candidate review reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import review_candidate as review_module
from ml.features import FEATURE_COLUMNS
from parser.job_description_parser import JobDescription
from parser.schemas import Resume


def test_review_candidate_writes_single_resume_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate a report without ranking multiple candidates."""
    resume_path = tmp_path / "resume.pdf"
    jd_path = tmp_path / "job.txt"
    output_path = tmp_path / "review_report.txt"
    resume_path.write_bytes(b"%PDF-1.7")
    jd_path.write_text("Data Scientist", encoding="utf-8")

    resume = Resume(
        name="Aisha Khan",
        email="aisha@example.com",
        phone="555-555-5555",
        skills=["Python", "SQL"],
        years_of_experience=4.0,
    )
    job_description = JobDescription(
        title="Data Scientist",
        required_skills=["Python", "SQL", "AWS"],
        preferred_skills=[],
    )
    feature_vector = pd.DataFrame(
        [
            {
                "semantic_similarity": 0.75,
                "skill_match_percentage": 2 / 3,
                "experience_match_score": 1.0,
                "education_match_score": 0.0,
                "certification_match_score": 0.0,
                "project_relevance_score": 0.5,
                "programming_language_match": 1.0,
                "framework_match": 0.0,
                "tools_match": 0.0,
                "years_of_experience_difference": 1.0,
                "number_of_matching_skills": 2.0,
                "total_candidate_skills": 2.0,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    monkeypatch.setattr(review_module, "extract_text_from_pdf", lambda path: "resume")
    monkeypatch.setattr(review_module, "parse_resume", lambda text: resume)
    monkeypatch.setattr(
        review_module,
        "parse_job_description",
        lambda text: job_description,
    )
    monkeypatch.setattr(
        review_module,
        "create_feature_vector",
        lambda parsed_resume, parsed_job: feature_vector,
    )
    monkeypatch.setattr(review_module, "load_model", lambda: object())
    monkeypatch.setattr(
        review_module,
        "predict_candidate_scores",
        lambda model, features: pd.DataFrame({"Predicted Score": [82.5]}),
    )

    report = review_module.review_candidate(resume_path, jd_path, output_path)

    assert "Resume Review Report" in report
    assert "Candidate Name\nAisha Khan" in report
    assert "Job Title\nData Scientist" in report
    assert "Overall Candidate Score\n82.50" in report
    assert "Overall Assessment\nStrong Match" in report
    assert "Section 1\nSummary" in report
    assert "Section 2\nStrengths" in report
    assert "Section 3\nWeaknesses" in report
    assert "- Python" in report
    assert "- AWS" in report
    assert "Detailed Scores" in report
    assert "Interview Recommendation\nHighly Recommended" in report
    assert output_path.read_text(encoding="utf-8") == report


def test_review_candidate_validates_input_paths(tmp_path: Path) -> None:
    """Raise friendly missing-file errors before parsing starts."""
    jd_path = tmp_path / "job.txt"
    jd_path.write_text("Data Scientist", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Resume file not found"):
        review_module.review_candidate(tmp_path / "missing.pdf", jd_path)
