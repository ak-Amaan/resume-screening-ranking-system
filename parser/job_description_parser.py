"""Job description parsing module for feature engineering inputs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "required_skills": ("required skills", "must have skills", "requirements"),
    "preferred_skills": ("preferred skills", "nice to have", "preferred qualifications"),
    "education": ("education", "education requirements"),
    "certifications": ("certifications", "certification requirements"),
    "responsibilities": ("responsibilities", "key responsibilities", "what you will do"),
    "qualifications": ("qualifications", "minimum qualifications"),
    "tools": ("tools", "platforms and tools"),
    "frameworks": ("frameworks", "libraries and frameworks"),
    "programming_languages": ("programming languages", "languages"),
}

SKILL_KEYWORDS: tuple[str, ...] = (
    "Python",
    "SQL",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Natural Language Processing",
    "Pandas",
    "NumPy",
    "scikit-learn",
    "TensorFlow",
    "PyTorch",
    "spaCy",
    "PyMuPDF",
    "AWS",
    "Azure",
    "Docker",
    "Kubernetes",
    "Git",
    "Linux",
    "Tableau",
    "Power BI",
    "Excel",
    "Java",
    "JavaScript",
    "TypeScript",
    "React",
    "Django",
    "Flask",
    "FastAPI",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Airflow",
    "Spark",
    "Hadoop",
    "MLOps",
    "Statistics",
    "Data Visualization",
    "Terraform",
    "Jenkins",
    "Cyber Security",
    "SIEM",
    "Network Security",
)

PROGRAMMING_LANGUAGES = ("Python", "SQL", "Java", "JavaScript", "TypeScript", "Go", "Bash")
FRAMEWORKS = (
    "React",
    "Django",
    "Flask",
    "FastAPI",
    "TensorFlow",
    "PyTorch",
    "scikit-learn",
    "spaCy",
    "Spark",
)
TOOLS = (
    "AWS",
    "Azure",
    "Docker",
    "Kubernetes",
    "Git",
    "Linux",
    "Tableau",
    "Power BI",
    "Excel",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Airflow",
    "Hadoop",
    "Terraform",
    "Jenkins",
    "SIEM",
)

EDUCATION_PATTERN = re.compile(
    r"\b(?:B\.?Tech|M\.?Tech|B\.?E\.?|M\.?E\.?|B\.?Sc|M\.?Sc|Bachelor(?:'s)?|"
    r"Master(?:'s)?|MBA|Ph\.?D\.?|Doctorate|degree)\b[^.\n]*",
    re.IGNORECASE,
)
YEARS_PATTERN = re.compile(
    r"(?P<years>\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class JobDescription:
    """Structured job-description fields used by feature engineering."""

    title: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    years_of_experience: float = 0.0
    certifications: list[str] = field(default_factory=list)
    programming_languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    qualifications: list[str] = field(default_factory=list)
    raw_text: str = ""


def parse_job_description(text: str) -> JobDescription:
    """Parse a job description into structured fields.

    Args:
        text: Raw job description text.

    Returns:
        Parsed job-description dataclass.

    Raises:
        ValueError: If the job description is empty.
    """
    normalized_text = _normalize_text(text)
    if not normalized_text:
        raise ValueError("Job description text cannot be empty.")

    logger.info("Parsing job description with %d characters.", len(normalized_text))
    sections = _extract_sections(normalized_text)

    required_text = sections.get("required_skills", normalized_text)
    preferred_text = sections.get("preferred_skills", "")
    all_text = normalized_text

    return JobDescription(
        title=_extract_title(normalized_text),
        required_skills=_extract_keywords(required_text, SKILL_KEYWORDS),
        preferred_skills=_extract_keywords(preferred_text, SKILL_KEYWORDS),
        education=_extract_education(sections.get("education", normalized_text)),
        years_of_experience=_extract_years_of_experience(normalized_text),
        certifications=_extract_certifications(
            sections.get("certifications", normalized_text)
        ),
        programming_languages=_extract_keywords(all_text, PROGRAMMING_LANGUAGES),
        frameworks=_extract_keywords(all_text, FRAMEWORKS),
        tools=_extract_keywords(all_text, TOOLS),
        responsibilities=_extract_bullets(sections.get("responsibilities", "")),
        qualifications=_extract_bullets(sections.get("qualifications", "")),
        raw_text=normalized_text,
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\x00", " ")).strip()


def _extract_title(text: str) -> str:
    for line in _meaningful_lines(text):
        if _match_section_header(line) is None:
            return line.strip(" -:")
    return ""


def _extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for line in text.splitlines():
        section_name = _match_section_header(line)
        if section_name:
            current_section = section_name
            sections.setdefault(current_section, [])
            continue
        if current_section:
            sections[current_section].append(line)

    return {section: "\n".join(lines).strip() for section, lines in sections.items()}


def _match_section_header(line: str) -> str | None:
    normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
    for section, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return section
    return None


def _extract_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    text_lower = text.lower()
    matches = [
        keyword
        for keyword in keywords
        if re.search(
            rf"(?<![A-Za-z0-9+#.-]){re.escape(keyword.lower())}(?![A-Za-z0-9+#.-])",
            text_lower,
        )
    ]
    return _dedupe(matches)


def _extract_education(text: str) -> list[str]:
    matches = [match.group(0).strip(" -,.") for match in EDUCATION_PATTERN.finditer(text)]
    return _dedupe(matches)


def _extract_years_of_experience(text: str) -> float:
    years = [float(match.group("years")) for match in YEARS_PATTERN.finditer(text)]
    return max(years, default=0.0)


def _extract_certifications(text: str) -> list[str]:
    certifications: list[str] = []
    for line in _meaningful_lines(text):
        if re.search(
            r"\b(certified|certificate|certification|aws|azure|security\+|cissp|"
            r"kubernetes|scrum|tensorflow)\b",
            line,
            re.IGNORECASE,
        ):
            certifications.append(line.strip(" -,."))
    return _dedupe(certifications)


def _extract_bullets(text: str) -> list[str]:
    return [line for line in _meaningful_lines(text) if _match_section_header(line) is None]


def _meaningful_lines(text: str) -> list[str]:
    return [line.strip(" -\t") for line in text.splitlines() if line.strip(" -\t")]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            unique_values.append(value)
    return unique_values
