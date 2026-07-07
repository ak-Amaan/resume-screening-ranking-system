"""Tests for job description parsing."""

from __future__ import annotations

from parser.job_description_parser import parse_job_description


def test_parse_job_description_extracts_required_fields() -> None:
    """Extract structured fields from a realistic job description."""
    text = """
    Machine Learning Engineer

    Required Skills
    Python, Machine Learning, PyTorch, Docker, Kubernetes

    Preferred Skills
    AWS, Airflow

    Education
    Master's degree in Computer Science or Statistics

    Certifications
    AWS Certified Machine Learning preferred

    Responsibilities
    - Deploy machine learning services

    Qualifications
    - 5+ years of machine learning engineering experience
    """

    job_description = parse_job_description(text)

    assert job_description.title == "Machine Learning Engineer"
    assert "Python" in job_description.required_skills
    assert "AWS" in job_description.preferred_skills
    assert job_description.years_of_experience == 5.0
    assert job_description.education
    assert job_description.certifications
    assert job_description.responsibilities
    assert job_description.qualifications

