"""
VOICE PROFILES — Maps each AI model to a musical voice.

Each model has a sonic identity. When multiple models contribute to a session,
their voices interweave like instruments in an ensemble. The music doesn't
just accompany the thinking — it IS the thinking, and each model's
personality shines through its voice.

Model → Voice mapping:
    Flash   → bright alto, fast tempo, syncopated (saxophone energy)
    Pro     → deep baritone, moderate tempo, structured (cello authority)
    Hermes  → warm female, flowing, multi-layered (Rhodes + strings)
    Wesley  → simple, pure tones, minimal (bell + piano)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VoiceProfile:
    """The musical identity of a single AI model."""
    model_name: str           # canonical name
    display_name: str         # pretty name
    role_description: str     # what this voice represents in the ensemble

    # Timbral identity
    primary_instrument: str   # lead instrument when this model speaks
    secondary_instruments: list[str]  # supporting textures
    vocal_character: str      # description of the sonic character

    # Rhythmic identity
    default_tempo_modifier: int  # BPM adjustment from baseline
    swing_preference: float      # 0.0 (straight) to 1.0 (full swing)
    rhythmic_pattern: str        # "syncopated", "straight", "flowing", "minimal"

    # Harmonic identity
    preferred_modes: list[str]   # e.g., ["dorian", "mixolydian", "blues"]
    chord_voicing: str           # how this model voices chords
    register: str                # "alto", "baritone", "soprano", "tenor"

    # Expression
    dynamics_default: str        # default dynamics level
    articulation: str            # how notes are articulated
    texture_preference: str      # "sparse", "layered", "dense", "minimal"

    # MMX prompt template
    mmx_voice_prompt: str = ""   # pre-built prompt fragment for this voice

    def build_prompt(self) -> str:
        """Build the MMX prompt fragment for this voice."""
        if self.mmx_voice_prompt:
            return self.mmx_voice_prompt
        parts = [
            f"{self.primary_instrument} lead",
            f"{self.vocal_character}",
            f"{self.register} register",
            f"{self.rhythmic_pattern} rhythm",
            f"swing {self.swing_preference:.1f}",
            f"tempo modifier {self.default_tempo_modifier:+d} BPM",
            f"{self.articulation} articulation",
            f"{self.texture_preference} texture",
            f"preferred modes: {', '.join(self.preferred_modes)}",
        ]
        return ", ".join(parts)


# ─── Model Voice Definitions ─────────────────────────────────────────────────

FLASH_VOICE = VoiceProfile(
    model_name="flash",
    display_name="Flash",
    role_description="The quicksilver voice — bright, fast, full of energy",
    primary_instrument="alto saxophone",
    secondary_instruments=["electric piano", "light percussion"],
    vocal_character="bright, piercing, energetic alto",
    default_tempo_modifier=+15,
    swing_preference=0.7,
    rhythmic_pattern="syncopated",
    preferred_modes=["mixolydian", "blues", "dorian"],
    chord_voicing="upper extensions, sharp 11s, chromatic",
    register="alto",
    dynamics_default="mf",
    articulation="staccato",
    texture_preference="layered",
    mmx_voice_prompt=(
        "alto saxophone lead, bright and energetic, syncopated rhythmic patterns, "
        "fast runs and angular melodies, upper-register chord extensions, "
        "jazz fusion energy, weather-report brightness, quick bursts of notes "
        "interspersed with sustained tones, swing-heavy"
    ),
)

PRO_VOICE = VoiceProfile(
    model_name="pro",
    display_name="Pro",
    role_description="The foundational voice — deep, authoritative, the bass of the ensemble",
    primary_instrument="cello",
    secondary_instruments=["double bass", " french horn", "tuba"],
    vocal_character="deep baritone, resonant, authoritative",
    default_tempo_modifier=-5,
    swing_preference=0.3,
    rhythmic_pattern="straight",
    preferred_modes=["minor", "aeolian", "phrygian"],
    chord_voicing="root position, open fifths, pedal points",
    register="baritone",
    dynamics_default="mf",
    articulation="legato",
    texture_preference="layered",
    mmx_voice_prompt=(
        "cello lead, deep baritone register, resonant sustained tones, "
        "structured harmonic progressions, root-position chord voicings, "
        "open fifth drones, pedal points, deliberate pacing, "
        "straight rhythmic feel, weighty and authoritative"
    ),
)

HERMES_VOICE = VoiceProfile(
    model_name="hermes",
    display_name="Hermes",
    role_description="The weaver — warm, flowing, many layers becoming one",
    primary_instrument="Fender Rhodes",
    secondary_instruments=["string section", "harp", "female choir"],
    vocal_character="warm female vocal, flowing, multi-layered",
    default_tempo_modifier=0,
    swing_preference=0.5,
    rhythmic_pattern="flowing",
    preferred_modes=["lydian", "dorian", "ionian"],
    chord_voicing="lush, stacked thirds, added notes, wide voicings",
    register="soprano",
    dynamics_default="mp",
    articulation="legato",
    texture_preference="dense",
    mmx_voice_prompt=(
        "Fender Rhodes electric piano with warm string section, "
        "flowing multi-layered textures, female choir pads, "
        "lush chord voicings with added ninths and elevenths, "
        "wide stereo field, lydian and dorian colorings, "
        "harp arpeggios weaving through, gentle and nurturing"
    ),
)

WESLEY_VOICE = VoiceProfile(
    model_name="wesley",
    display_name="Wesley",
    role_description="The pure voice — small, simple, beautiful, honest",
    primary_instrument="music box",
    secondary_instruments=["bell", "solo piano"],
    vocal_character="simple, pure tones, childlike wonder",
    default_tempo_modifier=-10,
    swing_preference=0.0,
    rhythmic_pattern="minimal",
    preferred_modes=["major pentatonic", "major", "ionian"],
    chord_voicing="triads, root position, open and spare",
    register="soprano",
    dynamics_default="p",
    articulation="legato",
    texture_preference="minimal",
    mmx_voice_prompt=(
        "music box and bell, pure simple tones, minimal texture, "
        "childlike wonder, major pentatonic melodies, "
        "solo piano with generous space between notes, "
        "triadic harmony, open and spare, quiet dynamics, "
        "the sound of curiosity and innocence"
    ),
)

# ─── Registry ────────────────────────────────────────────────────────────────

MODEL_VOICES: dict[str, VoiceProfile] = {
    "flash": FLASH_VOICE,
    "pro": PRO_VOICE,
    "hermes": HERMES_VOICE,
    "wesley": WESLEY_VOICE,
}

# Aliases for model name variations
NAME_ALIASES = {
    "deepseek-v4-flash": "flash",
    "deepseek-flash": "flash",
    "flash-v4": "flash",
    "deepseek-v4-pro": "pro",
    "deepseek-pro": "pro",
    "pro-v4": "pro",
    "hermes-3": "hermes",
    "hermes-3-llama": "hermes",
    "nous-hermes": "hermes",
    "wesley": "wesley",
    "haiku": "wesley",
    "claude-haiku": "wesley",
    "small-model": "wesley",
}


def get_voice(model_name: str) -> VoiceProfile:
    """
    Get the voice profile for a model by name.
    Falls back to a default composite voice if the model is unknown.
    """
    key = model_name.lower().strip()

    # Direct match
    if key in MODEL_VOICES:
        return MODEL_VOICES[key]

    # Try aliases
    if key in NAME_ALIASES:
        return MODEL_VOICES[NAME_ALIASES[key]]

    # Fuzzy match — check if any known name is a substring
    for alias, canonical in NAME_ALIASES.items():
        if alias in key or key in alias:
            return MODEL_VOICES[canonical]

    # Default fallback: a neutral voice
    return VoiceProfile(
        model_name=key,
        display_name=key.title(),
        role_description="Unknown voice — neutral, adaptive",
        primary_instrument="piano",
        secondary_instruments=["ambient pad"],
        vocal_character="neutral, adaptive",
        default_tempo_modifier=0,
        swing_preference=0.3,
        rhythmic_pattern="mixed",
        preferred_modes=["major", "minor"],
        chord_voicing="standard",
        register="tenor",
        dynamics_default="mp",
        articulation="mixed",
        texture_preference="sparse",
        mmx_voice_prompt="piano lead, neutral and adaptive, moderate dynamics, mixed articulation",
    )


def list_voices() -> list[VoiceProfile]:
    """Return all registered voice profiles."""
    return list(MODEL_VOICES.values())


def ensemble_prompt(model_names: list[str]) -> str:
    """
    Generate an MMX prompt for an ensemble of models playing together.
    Each model contributes its voice to the overall texture.
    """
    voices = [get_voice(name) for name in model_names]
    parts = ["ensemble piece with the following voices:"]
    for v in voices:
        parts.append(f"  - {v.display_name}: {v.mmx_voice_prompt}")
    parts.append("all voices interweaving, each maintaining its character")
    parts.append("creating a unified sonic texture")
    return "\n".join(parts)
