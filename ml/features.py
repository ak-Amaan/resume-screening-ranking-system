"""Feature extraction and feature vector construction module."""

from __future__ import annotations

import logging
import re

import pandas as pd

from nlp.similarity import calculate_resume_job_similarity
from parser.job_description_parser import JobDescription, parse_job_description
from parser.schemas import Resume

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "semantic_similarity",
    "skill_match_percentage",
    "experience_match_score",
    "education_match_score",
    "certification_match_score",
    "project_relevance_score",
    "programming_language_match",
    "framework_match",
    "tools_match",
    "years_of_experience_difference",
    "number_of_matching_skills",
    "total_candidate_skills",
]


def create_feature_vector(
    resume: Resume,
    job_description: str | JobDescription,
) -> pd.DataFrame:
    """Create a candidate-job feature vector as a pandas DataFrame.

    Args:
        resume: Parsed resume dataclass.
        job_description: Raw or parsed job description.

    Returns:
        Single-row DataFrame containing numerical ranking features.
    """
    parsed_job = (
        parse_job_description(job_description)
        if isinstance(job_description, str)
        else job_description
    )

    logger.info(
        "Creating feature vector for resume '%s' and job '%s'.",
        resume.name,
        parsed_job.title,
    )

    matching_skills = _matching_items(resume.skills, parsed_job.required_skills)
    all_job_skills = _dedupe(parsed_job.required_skills + parsed_job.preferred_skills)

    features = {
        "semantic_similarity": calculate_resume_job_similarity(resume, parsed_job),
        "skill_match_percentage": _match_percentage(resume.skills, all_job_skills),
        "experience_match_score": _experience_match_score(
            resume.years_of_experience,
            parsed_job.years_of_experience,
        ),
        "education_match_score": _education_match_score(resume, parsed_job),
        "certification_match_score": _certification_match_score(resume, parsed_job),
        "project_relevance_score": _project_relevance_score(resume, parsed_job),
        "programming_language_match": _match_percentage(
            resume.skills,
            parsed_job.programming_languages,
        ),
        "framework_match": _match_percentage(resume.skills, parsed_job.frameworks),
        "tools_match": _match_percentage(resume.skills, parsed_job.tools),
        "years_of_experience_difference": abs(
            resume.years_of_experience - parsed_job.years_of_experience
        ),
        "number_of_matching_skills": float(len(matching_skills)),
        "total_candidate_skills": float(len(resume.skills)),
    }

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


def _experience_match_score(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 0.0
    return min(candidate_years / required_years, 1.0)


def _education_match_score(resume: Resume, job_description: JobDescription) -> float:
    if not job_description.education:
        return 0.0

    resume_text = " ".join(
        f"{education.degree} {education.institution}" for education in resume.education
    ).lower()
    job_text = " ".join(job_description.education).lower()

    degree_terms = ("ph.d", "doctorate", "master", "m.sc", "m.tech", "bachelor", "b.sc", "b.tech", "degree")
    matches = [
        term
        for term in degree_terms
        if term in resume_text and (term in job_text or _equivalent_degree(term, job_text))
    ]
    return 1.0 if matches else 0.0


def _certification_match_score(resume: Resume, job_description: JobDescription) -> float:
    if not job_description.certifications:
        return 0.0

    resume_certs = [certification.name for certification in resume.certifications]
    return _match_percentage(resume_certs, job_description.certifications)


def _project_relevance_score(resume: Resume, job_description: JobDescription) -> float:
    project_text = " ".join(
        f"{project.name} {project.description} {' '.join(project.technologies)}"
        for project in resume.projects
    )
    job_terms = _dedupe(
        job_description.required_skills
        + job_description.preferred_skills
        + job_description.frameworks
        + job_description.tools
    )
    return _keyword_coverage(project_text, job_terms)


def _match_percentage(candidate_values: list[str], required_values: list[str]) -> float:
    if not required_values:
        return 0.0
    return len(_matching_items(candidate_values, required_values)) / len(_dedupe(required_values))


def _matching_items(candidate_values: list[str], required_values: list[str]) -> list[str]:
    candidate_normalized = {_normalize_token(value) for value in candidate_values}
    matches = [
        required
        for required in _dedupe(required_values)
        if _normalize_token(required) in candidate_normalized
    ]
    return matches


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    unique_keywords = _dedupe(keywords)
    if not unique_keywords:
        return 0.0

    text_lower = text.lower()
    matches = [
        keyword
        for keyword in unique_keywords
        if re.search(
            rf"(?<![A-Za-z0-9+#.-]){re.escape(keyword.lower())}(?![A-Za-z0-9+#.-])",
            text_lower,
        )
    ]
    return len(matches) / len(unique_keywords)


def _equivalent_degree(term: str, job_text: str) -> bool:
    if term in {"b.sc", "b.tech", "bachelor"}:
        return "bachelor" in job_text or "degree" in job_text
    if term in {"m.sc", "m.tech", "master"}:
        return "master" in job_text
    if term in {"ph.d", "doctorate"}:
        return "ph.d" in job_text or "doctorate" in job_text
    return False


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", value.lower()).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        key = _normalize_token(value)
        if key and key not in seen:
            seen.add(key)
            unique_values.append(value)
    return unique_values
