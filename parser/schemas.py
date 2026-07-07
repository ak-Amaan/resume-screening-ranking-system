"""Dataclass schemas for parsed resume information."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Education:
    """Education credential extracted from a resume."""

    degree: str
    institution: str = ""
    year: str = ""


@dataclass(slots=True)
class Project:
    """Project entry extracted from a resume."""

    name: str
    description: str = ""
    technologies: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Certification:
    """Certification entry extracted from a resume."""

    name: str
    issuer: str = ""
    year: str = ""


@dataclass(slots=True)
class Resume:
    """Structured resume information parsed from raw PDF text."""

    name: str
    email: str
    phone: str
    skills: list[str] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    certifications: list[Certification] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)
    years_of_experience: float = 0.0
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)
