"""Resume parsing module for candidate identity, skills, education, and experience."""

from __future__ import annotations

import logging
import re
from functools import lru_cache

try:
    import spacy
    from spacy.language import Language
except ImportError:  # pragma: no cover - exercised only without dependency.
    spacy = None
    Language = object  # type: ignore[misc,assignment]

from parser.schemas import Certification, Education, Project, Resume

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_PATTERN = re.compile(r"(?:\(?\+?\d[\d\s().-]{7,}\d)")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
EXPERIENCE_PATTERN = re.compile(
    r"(?P<years>\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience",
    re.IGNORECASE,
)

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "skills": ("skills", "technical skills", "core skills", "technologies"),
    "education": ("education", "academic background"),
    "certifications": ("certifications", "certification", "licenses"),
    "projects": ("projects", "selected projects", "project experience"),
    "experience": ("experience", "work experience", "professional experience"),
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
)

DEGREE_PATTERN = re.compile(
    r"\b(?P<degree>(?:B\.?Tech|M\.?Tech|B\.?E\.?|M\.?E\.?|B\.?Sc|M\.?Sc|"
    r"Bachelor(?:'s)?|Master(?:'s)?|MBA|Ph\.?D\.?|Doctorate)[^\n,;]*)",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_spacy_model() -> Language | None:
    """Load a spaCy pipeline, falling back to a blank English tokenizer."""
    if spacy is None:
        logger.warning("spaCy is not installed; name extraction will use heuristics.")
        return None

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.warning(
            "spaCy model 'en_core_web_sm' is not installed; using blank English model."
        )
        return spacy.blank("en")


def parse_resume(text: str) -> Resume:
    """Parse raw resume text into a structured Resume dataclass.

    Args:
        text: Clean text extracted from a PDF resume.

    Returns:
        Parsed resume fields including contact details, skills, education,
        certifications, projects, and years of experience.

    Raises:
        ValueError: If the input text is empty.
    """
    clean_resume_text = _normalize_text(text)
    if not clean_resume_text:
        raise ValueError("Resume text cannot be empty.")

    logger.info("Parsing resume text with %d characters.", len(clean_resume_text))
    sections = _extract_sections(clean_resume_text)

    resume = Resume(
        name=extract_name(clean_resume_text),
        email=extract_email(clean_resume_text),
        phone=extract_phone(clean_resume_text),
        skills=extract_skills(clean_resume_text),
        education=extract_education(sections.get("education", clean_resume_text)),
        experience=extract_experience(sections.get("experience", clean_resume_text)),
        certifications=extract_certifications(
            sections.get("certifications", clean_resume_text)
        ),
        projects=extract_projects(sections.get("projects", "")),
        years_of_experience=extract_years_of_experience(clean_resume_text),
        raw_text=clean_resume_text,
    )
    logger.info("Parsed resume for candidate: %s", resume.name or "Unknown")
    return resume


def extract_name(text: str) -> str:
    """Extract a candidate name using spaCy entities and line heuristics."""
    nlp = _load_spacy_model()
    first_lines = [
        line.strip()
        for line in text.splitlines()[:8]
        if line.strip() and not EMAIL_PATTERN.search(line) and not PHONE_PATTERN.search(line)
    ]
    header_text = "\n".join(first_lines)

    if nlp is not None and nlp.pipe_names:
        doc = nlp(header_text)
        for entity in doc.ents:
            if entity.label_ == "PERSON" and _looks_like_name(entity.text):
                return _title_name(entity.text)

    for line in first_lines:
        normalized = re.sub(r"[^A-Za-z .'-]", "", line).strip()
        if _looks_like_name(normalized):
            return _title_name(normalized)

    return ""


def extract_email(text: str) -> str:
    """Extract the first email address from resume text."""
    match = EMAIL_PATTERN.search(text)
    return match.group(0).lower() if match else ""


def extract_phone(text: str) -> str:
    """Extract the first phone number from resume text."""
    for match in PHONE_PATTERN.finditer(text):
        candidate = re.sub(r"\s+", " ", match.group(0)).strip(" |,;")
        digit_count = len(re.sub(r"\D", "", candidate))
        if 10 <= digit_count <= 15:
            return candidate
    return ""


def extract_skills(text: str) -> list[str]:
    """Extract known technical and analytical skills from resume text."""
    text_lower = text.lower()
    skills = [
        skill
        for skill in SKILL_KEYWORDS
        if re.search(rf"(?<![A-Za-z0-9+#.-]){re.escape(skill.lower())}(?![A-Za-z0-9+#.-])", text_lower)
    ]
    return _dedupe_preserve_order(skills)


def extract_education(text: str) -> list[Education]:
    """Extract education credentials from text."""
    education: list[Education] = []
    for line in _meaningful_lines(text):
        degree_match = DEGREE_PATTERN.search(line)
        if not degree_match:
            continue
        degree = degree_match.group("degree").strip(" -,:;")
        year_match = YEAR_PATTERN.search(line)
        institution = _extract_institution(line, degree)
        education.append(
            Education(
                degree=degree,
                institution=institution,
                year=year_match.group(0) if year_match else "",
            )
        )

    return _dedupe_dataclasses(education, key=lambda item: item.degree.lower())


def extract_experience(text: str) -> list[str]:
    """Extract work experience bullet lines from text."""
    experience_lines: list[str] = []
    for line in _meaningful_lines(text):
        if _is_section_header(line):
            continue
        if re.search(r"\b(engineer|analyst|developer|scientist|intern|manager|lead)\b", line, re.IGNORECASE):
            experience_lines.append(line)
        elif re.search(r"\b(?:19|20)\d{2}\s*[-–]\s*(?:present|(?:19|20)\d{2})\b", line, re.IGNORECASE):
            experience_lines.append(line)

    return _dedupe_preserve_order(experience_lines)


def extract_certifications(text: str) -> list[Certification]:
    """Extract certifications from text."""
    certifications: list[Certification] = []
    for line in _meaningful_lines(text):
        if _is_section_header(line):
            continue
        if not re.search(
            r"\b(certified|certificate|certification|aws|azure|google|scrum|tensorflow)\b",
            line,
            re.IGNORECASE,
        ):
            continue
        year_match = YEAR_PATTERN.search(line)
        name, issuer = _split_name_and_issuer(line)
        certifications.append(
            Certification(
                name=name,
                issuer=issuer,
                year=year_match.group(0) if year_match else "",
            )
        )

    return _dedupe_dataclasses(certifications, key=lambda item: item.name.lower())


def extract_projects(text: str) -> list[Project]:
    """Extract project entries from a projects section."""
    projects: list[Project] = []
    current_name = ""
    current_description: list[str] = []

    for line in _meaningful_lines(text):
        if _is_section_header(line):
            continue
        if _looks_like_project_heading(line):
            if current_name:
                projects.append(_build_project(current_name, current_description))
            current_name = line.strip(" -:")
            current_description = []
        elif current_name:
            current_description.append(line)

    if current_name:
        projects.append(_build_project(current_name, current_description))

    if not projects:
        for line in _meaningful_lines(text):
            if re.search(r"\b(project|dashboard|system|pipeline|classifier|analysis)\b", line, re.IGNORECASE):
                projects.append(_build_project(line, []))

    return _dedupe_dataclasses(projects, key=lambda item: item.name.lower())


def extract_years_of_experience(text: str) -> float:
    """Extract total years of experience from explicit resume statements."""
    years = [float(match.group("years")) for match in EXPERIENCE_PATTERN.finditer(text)]
    if years:
        return max(years)

    ranges: list[float] = []
    for start, end in re.findall(
        r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2}|present)\b",
        text,
        flags=re.IGNORECASE,
    ):
        start_year = int(start)
        end_year = 2026 if end.lower() == "present" else int(end)
        if end_year >= start_year:
            ranges.append(float(end_year - start_year))
    return max(ranges, default=0.0)


def _normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\x00", " ")).strip()


def _extract_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for line in lines:
        section_name = _match_section_header(line)
        if section_name:
            current_section = section_name
            sections.setdefault(current_section, [])
            continue
        if current_section:
            sections[current_section].append(line)

    return {section: "\n".join(content).strip() for section, content in sections.items()}


def _match_section_header(line: str) -> str | None:
    normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
    for section, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return section
    return None


def _is_section_header(line: str) -> bool:
    return _match_section_header(line) is not None


def _meaningful_lines(text: str) -> list[str]:
    return [line.strip(" -\t") for line in text.splitlines() if line.strip(" -\t")]


def _looks_like_name(value: str) -> bool:
    words = value.split()
    if not 2 <= len(words) <= 4:
        return False
    blocked_words = {"resume", "curriculum", "email", "phone", "skills", "education"}
    if any(word.lower() in blocked_words for word in words):
        return False
    return all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", word) for word in words)


def _title_name(value: str) -> str:
    return " ".join(word[:1].upper() + word[1:] for word in value.split())


def _extract_institution(line: str, degree: str) -> str:
    without_degree = line.replace(degree, "")
    without_year = YEAR_PATTERN.sub("", without_degree)
    parts = [part.strip(" -,:;") for part in re.split(r"\s{2,}|\||,", without_year)]
    return next((part for part in parts if part), "")


def _split_name_and_issuer(line: str) -> tuple[str, str]:
    cleaned = YEAR_PATTERN.sub("", line).strip(" -,:;")
    parts = [part.strip(" -,:;") for part in re.split(r"\s+-\s+|\s+\|\s+|,", cleaned, maxsplit=1)]
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _looks_like_project_heading(line: str) -> bool:
    if len(line) > 90:
        return False
    if line.endswith("."):
        return False
    if re.search(
        r"\b(project|dashboard|system|pipeline|classifier|analysis|platform|"
        r"service|search|toolkit)\b",
        line,
        re.IGNORECASE,
    ):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'.-]*", line)
    if 2 <= len(words) <= 6:
        significant_words = [word for word in words if len(word) > 2]
        return bool(significant_words) and all(
            word[0].isupper() for word in significant_words
        )
    return line.endswith(":")


def _build_project(name: str, description_lines: list[str]) -> Project:
    description = " ".join(description_lines).strip()
    combined_text = f"{name} {description}"
    return Project(
        name=name.strip(" -:"),
        description=description,
        technologies=extract_skills(combined_text),
    )


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            unique_values.append(value)
    return unique_values


def _dedupe_dataclasses[T](items: list[T], key) -> list[T]:
    seen: set[str] = set()
    unique_items: list[T] = []
    for item in items:
        item_key = key(item)
        if item_key not in seen:
            seen.add(item_key)
            unique_items.append(item)
    return unique_items
