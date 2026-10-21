from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .defaults import DEFAULT_EXPLANATION_MODEL


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Level(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=4000)
    points: float = Field(ge=0, le=10000)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("IDs must contain only letters, numbers, hyphens or underscores.")
        return value


class Criterion(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    levels: list[Level] = Field(min_length=2, max_length=12)

    _valid_id = field_validator("id")(Level.valid_id.__func__)

    @model_validator(mode="after")
    def valid_levels(self):
        if len({level.id for level in self.levels}) != len(self.levels):
            raise ValueError("Each level needs a unique ID within its criterion.")
        if max(level.points for level in self.levels) <= 0:
            raise ValueError("Each criterion needs at least one positive-point level.")
        return self


class Rubric(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    instructions: str = Field(default="", max_length=20000)
    criteria: list[Criterion] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def unique_criteria(self):
        if len({item.id for item in self.criteria}) != len(self.criteria):
            raise ValueError("Each criterion needs a unique ID.")
        return self


class Settings(StrictModel):
    provider: Literal["laya", "systemone"] = "laya"
    display_name: str = Field(default="", max_length=80)
    endpoint: str = Field(default="http://127.0.0.1:1234/v1/systemone", max_length=500)
    model: str = Field(default="models/laya-multilingual", min_length=1, max_length=1000)
    confidence_threshold: float = Field(default=0.65, ge=0, le=1)
    scoring_mode: Literal["suggest", "strict_review"] = "suggest"
    max_input_chars: int = Field(default=24000, ge=100, le=200000)

    @model_validator(mode="after")
    def local_endpoint(self):
        if self.provider == "systemone":
            from .providers import validate_local_endpoint
            self.endpoint = validate_local_endpoint(self.endpoint)
        return self


def model_identity(settings):
    """UI labels and scoring preferences do not create a different model."""
    return (settings.provider, settings.model,
            settings.endpoint if settings.provider == "systemone" else None)


class ProbeRequest(StrictModel):
    settings: Settings


class GradeRequest(StrictModel):
    rubric: Rubric
    settings: Settings
    text: str = Field(min_length=1, max_length=200000)
    submission_name: str = Field(default="Untitled submission", max_length=250)
    assignment_details: str = Field(default="", max_length=20000)
    grading_guidance: str = Field(default="", max_length=12000)


class CompareRequest(StrictModel):
    rubric: Rubric
    models: list[Settings] = Field(min_length=1, max_length=8)
    text: str = Field(min_length=1, max_length=200000)
    submission_name: str = Field(default="Untitled submission", max_length=250)
    assignment_details: str = Field(default="", max_length=20000)
    grading_guidance: str = Field(default="", max_length=12000)

    @model_validator(mode="after")
    def distinct_models(self):
        seen = set()
        for model in self.models:
            identity = model_identity(model)
            if identity in seen:
                raise ValueError("Choose different local models for the comparison.")
            seen.add(identity)
        return self


class ExplanationSettings(StrictModel):
    enabled: bool = True
    model: str = Field(default=DEFAULT_EXPLANATION_MODEL, min_length=1, max_length=200)

    @field_validator("model")
    @classmethod
    def local_model_name(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", value) or "://" in value:
            raise ValueError("Use the name of an installed local Ollama model.")
        return value


class VisionSettings(StrictModel):
    model: str = Field(default="openbmb/minicpm-v4.7:1b", min_length=1, max_length=200)
    auto_detect: bool = Field(default=True, strict=True)
    detailed_crops: bool = Field(default=True, strict=True)

    @field_validator("model")
    @classmethod
    def local_vision_model(cls, value):
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*){0,2}(?::[A-Za-z0-9][A-Za-z0-9._-]*)?", value)
                or value.lower().endswith((":cloud", "-cloud"))):
            raise ValueError("Use the name of an installed local vision model, without a URL or cloud tag.")
        return value


class VisionRequest(StrictModel):
    model: str = Field(min_length=1, max_length=200)
    image_data_url: str = Field(min_length=1, max_length=11_000_000)
    detailed_crops: bool = Field(default=True, strict=True)
    transcribe: bool = Field(default=True, strict=True)

    _valid_model = field_validator("model")(VisionSettings.local_vision_model.__func__)


class ExplanationDecision(StrictModel):
    provider: Literal["laya", "systemone"]
    model: str = Field(min_length=1, max_length=1000)
    endpoint: str | None = Field(default=None, max_length=500)
    ok: bool = Field(strict=True)
    level_id: str | None = Field(default=None, max_length=64)
    points: float | None = Field(default=None, ge=0, le=10000, strict=True)

    @model_validator(mode="after")
    def local_endpoint(self):
        if self.provider == "systemone" and self.endpoint is not None:
            from .providers import validate_local_endpoint
            self.endpoint = validate_local_endpoint(self.endpoint)
        return self


class ExplanationRequest(StrictModel):
    rubric: Rubric
    text: str = Field(min_length=1, max_length=200000)
    assignment_details: str = Field(default="", max_length=20000)
    grading_guidance: str = Field(default="", max_length=12000)
    criterion_id: str = Field(min_length=1, max_length=64)
    decisions: list[ExplanationDecision] = Field(min_length=2, max_length=8)
    comparison_receipt: str = Field(min_length=1, max_length=128)
    settings: ExplanationSettings = Field(default_factory=ExplanationSettings)

    _valid_id = field_validator("criterion_id")(Level.valid_id.__func__)

    @model_validator(mode="after")
    def distinct_decisions(self):
        if len({model_identity(item) for item in self.decisions}) != len(self.decisions):
            raise ValueError("Choose different local model configurations for the explanation.")
        return self
