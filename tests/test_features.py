"""Tests for candidate-job feature engineering."""

from __future__ import annotations

import pandas as pd

from ml import features
from ml.features import FEATURE_COLUMNS, create_feature_vector
from parser.schemas import Certification, Education, Project, Resume


def test_create_feature_vector_returns_expected_dataframe(monkeypatch) -> None:
    """Build the requested numerical feature vector for one resume and job."""
    monkeypatch.setattr(
        features,
        "calculate_resume_job_similarity",
        lambda resume, job_description: 0.82,
    )
    resume = Resume(
        name="Aisha Khan",
        email="aisha@example.com",
        phone="555-555-5555",
        skills=["Python", "SQL", "Machine Learning", "AWS", "Docker"],
        education=[Education(degree="M.Sc Computer Science")],
        certifications=[Certification(name="AWS Certified Machine Learning")],
        projects=[
            Project(
                name="ML Pipeline",
                description="Built Python and AWS machine learning pipeline.",
                technologies=["Python", "AWS"],
            )
        ],
        years_of_experience=5.0,
    )
    job_text = """
    Data Scientist

    Required Skills
    Python, SQL, Machine Learning

    Preferred Skills
    AWS, Tableau

    Education
    Master's degree in Computer Science or Statistics

    Certifications
    AWS Certified Machine Learning preferred

    Programming Languages
    Python, SQL

    Frameworks
    scikit-learn

    Tools
    AWS, Docker

    Qualifications
    - 4+ years of data science experience
    """

    feature_vector = create_feature_vector(resume, job_text)

    assert isinstance(feature_vector, pd.DataFrame)
    assert list(feature_vector.columns) == FEATURE_COLUMNS
    assert feature_vector.shape == (1, len(FEATURE_COLUMNS))
    assert feature_vector.loc[0, "semantic_similarity"] == 0.82
    assert feature_vector.loc[0, "skill_match_percentage"] == 0.8
    assert feature_vector.loc[0, "number_of_matching_skills"] == 3.0
    assert feature_vector.loc[0, "total_candidate_skills"] == 5.0
    assert feature_vector.loc[0, "experience_match_score"] == 1.0

