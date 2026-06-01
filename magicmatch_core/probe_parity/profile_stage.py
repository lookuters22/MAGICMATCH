"""Profile-stage flags matching standalone_probe colorMatchLutProfileStageSelect."""

from __future__ import annotations

from typing import TypedDict

PROFILE_STAGE_CURRENT = "current_profile_stages"
PROFILE_STAGE_FORCE_COLOR = "force_color_look"
PROFILE_STAGE_DISABLE_PROFILE = "disable_profile_look"
PROFILE_STAGE_FORCE_COLOR_DISABLE_PROFILE = "force_color_look_disable_profile_look"

PROFILE_STAGE_OPTIONS = [
    PROFILE_STAGE_CURRENT,
    PROFILE_STAGE_FORCE_COLOR,
    PROFILE_STAGE_DISABLE_PROFILE,
    PROFILE_STAGE_FORCE_COLOR_DISABLE_PROFILE,
]


class ProfileStageFlags(TypedDict):
    forceColorLookTableWithUserLut: bool
    disableProfileLookTable: bool


def profile_stage_flags(stage: str) -> ProfileStageFlags:
    if stage == PROFILE_STAGE_FORCE_COLOR:
        return {"forceColorLookTableWithUserLut": True, "disableProfileLookTable": False}
    if stage == PROFILE_STAGE_DISABLE_PROFILE:
        return {"forceColorLookTableWithUserLut": False, "disableProfileLookTable": True}
    if stage == PROFILE_STAGE_FORCE_COLOR_DISABLE_PROFILE:
        return {"forceColorLookTableWithUserLut": True, "disableProfileLookTable": True}
    return {"forceColorLookTableWithUserLut": False, "disableProfileLookTable": False}


def normalize_profile_stage(value: str | None) -> str:
    if isinstance(value, str) and value in PROFILE_STAGE_OPTIONS:
        return value
    return PROFILE_STAGE_CURRENT
