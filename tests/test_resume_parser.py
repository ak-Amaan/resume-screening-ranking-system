"""Tests for resume parsing behavior."""

from __future__ import annotations

from parser.resume_parser import parse_resume


def test_parse_resume_extracts_core_fields() -> None:
    """Parse contact, skills, education, project, certification, and experience."""
    resume_text = """
    Aisha Khan
    aisha.khan@example.com
    +1 415 555 0198

    Professional Summary
    Data Scientist with 5 years of experience building NLP and analytics systems.

    Skills
    Python, SQL, Machine Learning, NLP, Pandas, scikit-learn, AWS

    Education
    M.Sc Computer Science, Stanford University, 2020

    Certifications
    AWS Certified Machine Learning - Specialty, Amazon Web Services, 2022

    Projects
    Resume Ranking System
    Built an NLP pipeline with Python, scikit-learn, and Pandas.

    Experience
    Data Scientist, BrightData Labs, 2021 - Present
    Machine Learning Intern, Insight Analytics, 2019 - 2020
    """

    resume = parse_resume(resume_text)

    assert resume.name == "Aisha Khan"
    assert resume.email == "aisha.khan@example.com"
    assert resume.phone == "+1 415 555 0198"
    assert "Python" in resume.skills
    assert "Machine Learning" in resume.skills
    assert resume.education[0].degree.startswith("M.Sc")
    assert resume.certifications[0].name.startswith("AWS Certified")
    assert resume.projects[0].name == "Resume Ranking System"
    assert resume.years_of_experience == 5.0
    assert len(resume.experience) >= 2

