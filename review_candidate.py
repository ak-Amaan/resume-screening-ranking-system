"""Single-candidate review report generation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml.features import _dedupe, _matching_items, create_feature_vector
from ml.ranker import load_model, predict_candidate_scores
from parser.job_description_parser import parse_job_description
from parser.pdf_extractor import extract_text_from_pdf
from parser.resume_parser import parse_resume

DEFAULT_REVIEW_OUTPUT_PATH = Path("output/review_report.txt")

logger = logging.getLogger(__name__)


def review_candidate(
    resume_path: str | Path,
    job_description_path: str | Path,
    output_path: str | Path = DEFAULT_REVIEW_OUTPUT_PATH,
) -> str:
    """Evaluate one resume against one job description and write a report."""
    resume_file = Path(resume_path)
    jd_file = Path(job_description_path)

    if not resume_file.exists():
        raise FileNotFoundError(f"Resume file not found: {resume_file}")
    if not jd_file.exists():
        raise FileNotFoundError(f"Job description file not found: {jd_file}")

    resume = parse_resume(extract_text_from_pdf(resume_file))
    job_description = parse_job_description(jd_file.read_text(encoding="utf-8"))
    feature_vector = create_feature_vector(resume, job_description)

    model = load_model()
    predictions = predict_candidate_scores(model, feature_vector)
    overall_score = float(predictions.loc[0, "Predicted Score"])

    report = _format_review_report(
        candidate_name=resume.name or resume_file.stem,
        job_title=job_description.title or jd_file.stem,
        overall_score=overall_score,
        feature_vector=feature_vector,
        matched_skills=_matching_items(
            resume.skills,
            _dedupe(
                job_description.required_skills + job_description.preferred_skills
            ),
        ),
        missing_skills=_missing_items(
            resume.skills,
            _dedupe(job_description.required_skills),
        ),
        has_education_requirements=bool(job_description.education),
        has_certification_requirements=bool(job_description.certifications),
        has_programming_language_requirements=bool(
            job_description.programming_languages
        ),
        has_framework_requirements=bool(job_description.frameworks),
        has_tool_requirements=bool(job_description.tools),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    logger.info("Saved review report to %s", output)

    print(report)
    return report


def _format_review_report(
    candidate_name: str,
    job_title: str,
    overall_score: float,
    feature_vector: pd.DataFrame,
    matched_skills: list[str],
    missing_skills: list[str],
    has_education_requirements: bool,
    has_certification_requirements: bool,
    has_programming_language_requirements: bool,
    has_framework_requirements: bool,
    has_tool_requirements: bool,
) -> str:
    row = feature_vector.iloc[0]
    assessment = _overall_assessment(overall_score)
    lines = [
        "Resume Review Report",
        "=========================================================",
        "",
        "Candidate Name",
        candidate_name,
        "",
        "Job Title",
        job_title,
        "",
        "Overall Candidate Score",
        f"{overall_score:.2f}",
        "",
        "Overall Assessment",
        assessment,
        "",
        "=========================================================",
        "",
        "Section 1",
        "Summary",
        _summary(row, overall_score, assessment, matched_skills, missing_skills),
        "",
        "Section 2",
        "Strengths",
        _format_list(_strengths(row)),
        "",
        "Section 3",
        "Weaknesses",
        _format_list(
            _weaknesses(
                row,
                missing_skills,
                has_education_requirements,
                has_certification_requirements,
                has_programming_language_requirements,
                has_framework_requirements,
                has_tool_requirements,
            )
        ),
        "",
        "Section 4",
        "Matched Skills",
        _format_list(matched_skills),
        "",
        "Section 5",
        "Missing Skills",
        _format_list(missing_skills),
        "",
        "Section 6",
        "Detailed Scores",
        f"Semantic Similarity: {_format_ratio(row['semantic_similarity'])}",
        f"Skill Match: {_format_ratio(row['skill_match_percentage'])}",
        f"Experience Match: {_format_ratio(row['experience_match_score'])}",
        f"Education Match: {_format_ratio(row['education_match_score'])}",
        f"Certification Match: {_format_ratio(row['certification_match_score'])}",
        f"Programming Languages: {_format_ratio(row['programming_language_match'])}",
        f"Frameworks: {_format_ratio(row['framework_match'])}",
        f"Tools: {_format_ratio(row['tools_match'])}",
        f"Project Relevance: {_format_ratio(row['project_relevance_score'])}",
        "",
        "Section 7",
        "Recommendations",
        _format_list(
            _recommendations(
                row,
                missing_skills,
                has_programming_language_requirements,
                has_framework_requirements,
                has_tool_requirements,
            )
        ),
        "",
        "Section 8",
        "Interview Recommendation",
        _interview_recommendation(overall_score),
    ]
    return "\n".join(lines) + "\n"


def _missing_items(candidate_values: list[str], required_values: list[str]) -> list[str]:
    matched = set(_matching_items(candidate_values, required_values))
    return [item for item in required_values if item not in matched]


def _format_ratio(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _format_list(values: list[str]) -> str:
    if not values:
        return "None"
    return "\n".join(f"- {value}" for value in values)


def _overall_assessment(score: float) -> str:
    if score >= 85.0:
        return "Excellent Match"
    if score >= 70.0:
        return "Strong Match"
    if score >= 55.0:
        return "Moderate Match"
    if score >= 40.0:
        return "Weak Match"
    return "Poor Match"


def _interview_recommendation(score: float) -> str:
    if score >= 80.0:
        return "Highly Recommended"
    if score >= 65.0:
        return "Recommended"
    if score >= 50.0:
        return "Consider"
    return "Not Recommended"


def _summary(
    row: pd.Series,
    overall_score: float,
    assessment: str,
    matched_skills: list[str],
    missing_skills: list[str],
) -> str:
    sentences = [
        (
            f"The candidate is a {assessment.lower()} with an overall score of "
            f"{overall_score:.2f}."
        ),
        (
            f"Semantic similarity is {_score_label(row['semantic_similarity'])} at "
            f"{_format_ratio(row['semantic_similarity'])}, and skill match is "
            f"{_format_ratio(row['skill_match_percentage'])}."
        ),
    ]

    if matched_skills:
        sentences.append(
            "Matched skills include " + _inline_items(matched_skills) + "."
        )
    if missing_skills:
        sentences.append(
            "The main gaps are " + _inline_items(missing_skills) + "."
        )
    else:
        sentences.append("No required skill gaps were identified from the parsed data.")

    return " ".join(sentences[:4])


def _strengths(row: pd.Series) -> list[str]:
    strengths: list[str] = []
    if row["semantic_similarity"] >= 0.75:
        strengths.append("Strong semantic similarity with the job description")
    elif row["semantic_similarity"] >= 0.65:
        strengths.append("Good semantic similarity with the job description")
    if row["skill_match_percentage"] >= 0.70:
        strengths.append("Strong match against required and preferred skills")
    if row["experience_match_score"] >= 0.80:
        strengths.append("High experience match for the role")
    if row["education_match_score"] >= 0.80:
        strengths.append("Relevant education for the role")
    if row["certification_match_score"] >= 0.80:
        strengths.append("Relevant certifications are present")
    if row["project_relevance_score"] >= 0.60:
        strengths.append("Relevant projects align with job requirements")
    if row["programming_language_match"] >= 0.70:
        strengths.append("Good programming language match")
    if row["framework_match"] >= 0.70:
        strengths.append("Good framework match")
    if row["tools_match"] >= 0.70:
        strengths.append("Good tools and platform match")
    return strengths


def _weaknesses(
    row: pd.Series,
    missing_skills: list[str],
    has_education_requirements: bool,
    has_certification_requirements: bool,
    has_programming_language_requirements: bool,
    has_framework_requirements: bool,
    has_tool_requirements: bool,
) -> list[str]:
    weaknesses: list[str] = []
    if missing_skills:
        weaknesses.append("Missing required skills: " + _inline_items(missing_skills))
    if row["skill_match_percentage"] < 0.50:
        weaknesses.append("Overall skill match is low")
    if row["experience_match_score"] < 0.70:
        weaknesses.append("Experience match is below the role requirement")
    if has_education_requirements and row["education_match_score"] < 0.50:
        weaknesses.append("Education alignment is limited or not clearly stated")
    if has_certification_requirements and row["certification_match_score"] == 0.0:
        weaknesses.append("Relevant certification match was not identified")
    if (
        has_programming_language_requirements
        and row["programming_language_match"] < 0.50
    ):
        weaknesses.append("Programming language match is limited")
    if has_framework_requirements and row["framework_match"] == 0.0:
        weaknesses.append("Framework experience is missing or not clearly stated")
    if has_tool_requirements and row["tools_match"] == 0.0:
        weaknesses.append("Tools or cloud technology match is missing")
    if row["project_relevance_score"] < 0.40:
        weaknesses.append("Project relevance to the role is limited")
    return weaknesses


def _recommendations(
    row: pd.Series,
    missing_skills: list[str],
    has_programming_language_requirements: bool,
    has_framework_requirements: bool,
    has_tool_requirements: bool,
) -> list[str]:
    recommendations: list[str] = []
    for skill in missing_skills:
        recommendations.append(f"Add {skill} if you have hands-on experience.")
    if row["project_relevance_score"] < 0.60:
        recommendations.append("Highlight projects that directly use the role's key skills.")
    if has_tool_requirements and row["tools_match"] < 0.50:
        recommendations.append("Mention relevant tools, cloud platforms, and deployment work.")
    if has_framework_requirements and row["framework_match"] < 0.50:
        recommendations.append("Add framework experience that matches the job description.")
    if (
        has_programming_language_requirements
        and row["programming_language_match"] < 0.50
    ):
        recommendations.append("Emphasize programming languages requested by the role.")
    if row["experience_match_score"] < 0.80:
        recommendations.append("Clarify total years of relevant experience.")
    if row["semantic_similarity"] < 0.65:
        recommendations.append("Align resume wording more closely with the job requirements.")
    if row["skill_match_percentage"] < 0.70:
        recommendations.append("Move the most relevant skills into a clear skills section.")
    return recommendations


def _score_label(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.50:
        return "moderate"
    return "low"


def _inline_items(values: list[str]) -> str:
    if len(values) <= 3:
        return ", ".join(values)
    return ", ".join(values[:3]) + f", and {len(values) - 3} more"
