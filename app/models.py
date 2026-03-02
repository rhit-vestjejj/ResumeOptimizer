from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class DateRange(StrictModel):
    start: Optional[str] = None
    end: Optional[str] = None


class Identity(StrictModel):
    name: str
    email: str
    phone: str
    location: str
    links: List[str] = Field(default_factory=list)


class EducationEntry(StrictModel):
    school: str
    degree: str
    major: str
    minors: List[str] = Field(default_factory=list)
    gpa: str = ''
    dates: DateRange = Field(default_factory=DateRange)
    coursework: List[str] = Field(default_factory=list)


class ExperienceEntry(StrictModel):
    company: str
    title: str
    location: str
    dates: DateRange = Field(default_factory=DateRange)
    bullets: List[str] = Field(default_factory=list)


class ProjectEntry(StrictModel):
    name: str
    link: Optional[str] = None
    dates: Optional[DateRange] = None
    tech: List[str] = Field(default_factory=list)
    bullets: List[str] = Field(default_factory=list)
    section: str = 'projects'

    @field_validator('section')
    @classmethod
    def validate_section(cls, value: str) -> str:
        normalized = (value or '').strip().lower()
        if normalized in {'', 'projects'}:
            return 'projects'
        if normalized in {'minor', 'minor_projects', 'minor-projects', 'minorprojects'}:
            return 'minor_projects'
        raise ValueError('section must be "projects" or "minor_projects"')


class Skills(StrictModel):
    categories: Dict[str, List[str]] = Field(default_factory=dict)


class CanonicalResume(StrictModel):
    schema_version: str = '1.1.0'
    identity: Identity
    summary: Optional[str] = None
    education: List[EducationEntry] = Field(default_factory=list)
    experience: List[ExperienceEntry] = Field(default_factory=list)
    projects: List[ProjectEntry] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    certifications: List[str] = Field(default_factory=list)
    awards: Optional[List[str]] = None


class VaultItemType(str, Enum):
    project = 'project'
    job = 'job'
    club = 'club'
    coursework = 'coursework'
    award = 'award'
    skillset = 'skillset'
    other = 'other'


class VaultBullet(StrictModel):
    text: str
    situation: Optional[str] = None
    task: Optional[str] = None
    action: Optional[str] = None
    outcome: Optional[str] = None
    impact: Optional[str] = None


class VaultItem(StrictModel):
    type: VaultItemType
    title: str
    dates: Optional[DateRange] = None
    tags: List[str] = Field(default_factory=list)
    tech: List[str] = Field(default_factory=list)
    bullets: List[VaultBullet] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    source_artifacts: List[str] = Field(default_factory=list)

    @field_validator('bullets', mode='before')
    @classmethod
    def coerce_bullets(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list):
            normalized: List[Dict[str, Any]] = []
            for item in value:
                if isinstance(item, str):
                    normalized.append({'text': item})
                else:
                    normalized.append(item)
            return normalized
        raise TypeError('bullets must be a list')


class JobRecord(StrictModel):
    job_id: str
    url: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    scraped_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TailorMode(str, Enum):
    HARD_TRUTH = 'HARD_TRUTH'
    FUCK_IT = 'FUCK_IT'


class JDAnalysis(StrictModel):
    target_role_keywords: List[str] = Field(default_factory=list)
    required_skills: List[str] = Field(default_factory=list)
    nice_to_haves: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)


class CandidateItem(StrictModel):
    source_type: str
    source_id: str
    title: str
    dates: Optional[DateRange] = None
    tags: List[str] = Field(default_factory=list)
    tech: List[str] = Field(default_factory=list)
    bullets: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None


class SelectedItem(StrictModel):
    source_type: str
    source_id: str
    title: str
    score: float
    why_included: str = ''


class VaultRelevanceItem(StrictModel):
    item_id: str
    title: str
    item_type: str
    relevance_score: float
    selected: bool = False
    why_selected: str = ''
    why_not_selected: str = ''
    matched_required_terms: List[str] = Field(default_factory=list)
    missing_required_terms: List[str] = Field(default_factory=list)


class RequiredSkillEvidence(StrictModel):
    required_term: str
    has_evidence: bool = False
    source_title: str = ''
    source_type: str = ''
    evidence_bullet: str = ''


class TailorReport(StrictModel):
    chosen_items: List[SelectedItem] = Field(default_factory=list)
    vault_relevance: List[VaultRelevanceItem] = Field(default_factory=list)
    missing_required_evidence: List[str] = Field(default_factory=list)
    required_skill_evidence_map: List[RequiredSkillEvidence] = Field(default_factory=list)
    keywords_covered: List[str] = Field(default_factory=list)
    keywords_missed: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    mode: TailorMode


class LLMResumeExtraction(StrictModel):
    resume: CanonicalResume


class LLMRewriteResponse(StrictModel):
    rewritten_bullets: List[str]
