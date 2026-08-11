"""
HARMONIC DICTIONARY — The mapping between emotional/cognitive states and musical parameters.

This is the heart of the Sonic Shape Engine. It translates confidence levels,
emotional states, and cognitive modes into concrete musical parameters that
drive MMX (MiniMax) music generation.

The core insight: music IS the system thinking. When confidence is low,
the music sounds uncertain. When confidence enters the creative band,
the music explores. When confidence is high, the music resolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConfidenceBand(Enum):
    """The five bands of confidence that map to distinct musical territories."""
    UNCERTAIN = "uncertain"        # 0.0 - 0.3: searching, lost, blue notes
    CREATIVE = "creative"          # 0.4 - 0.6: the creative band — jazz, exploration
    EMERGING = "emerging"          # 0.7 - 0.85: finding resolution, major key
    CONFIDENT = "confident"        # 0.86 - 1.0: bright, celebratory, arrival
    TRANSITIONAL = "transitional"  # between bands — brief, shimmering


class EmotionalState(Enum):
    """Emotional states derived from session analysis."""
    LOST = "lost"
    SEARCHING = "searching"
    EXPLORING = "exploring"
    WONDERING = "wondering"
    DISCOVERING = "discovering"
    CONFIDENT = "confident"
    TRIUMPHANT = "triumphant"
    CONTEMPLATIVE = "contemplative"
    PLAYFUL = "playful"
    MELANCHOLY = "melancholy"


@dataclass
class MusicalParameters:
    """Complete set of parameters that define a musical output."""
    # Tonality
    key: str                    # e.g., "C minor", "F major", "Bb"
    mode: str                   # "minor", "major", "dorian", "mixolydian", "blues"
    root_note: str              # "C", "F#", "Bb"

    # Rhythm
    tempo_bpm: int              # 50-160
    time_signature: str         # "4/4", "3/4", "5/4", "6/8"
    swing: float                # 0.0 (straight) to 1.0 (full swing)

    # Harmony
    chord_quality: str          # "minor7", "dominant7", "maj7", "sus4", "power"
    resolution: str             # "unresolved", "tension", "resolving", "resolved"
    dissonance: float           # 0.0 (consonant) to 1.0 (very dissonant)

    # Timbre
    primary_instrument: str     # lead voice
    secondary_instruments: list[str]  # supporting voices
    texture: str                # "sparse", "layered", "dense", "minimal"

    # Expression
    dynamics: str               # "pp", "p", "mp", "mf", "f", "ff"
    articulation: str           # "legato", "staccato", "mixed", "blue"
    mood_words: list[str]       # descriptors for MMX prompt

    # Generation
    mmx_prompt: str = ""        # Full prompt string for MMX
    duration_seconds: int = 45  # Target length
    band: str = ""              # Confidence band name

    def to_mmx_prompt(self) -> str:
        """Generate a complete MMX music generation prompt."""
        if self.mmx_prompt:
            return self.mmx_prompt
        parts = [
            f"{self.key} in {self.mode} mode",
            f"tempo {self.tempo_bpm} BPM in {self.time_signature}",
            f"led by {self.primary_instrument}",
        ]
        if self.secondary_instruments:
            parts.append(f"with {' '.join(self.secondary_instruments)}")
        parts.append(f"{self.resolution} harmony")
        parts.append(f"{self.texture} texture")
        parts.append(f"{self.dynamics} dynamics, {self.articulation}")
        if self.swing > 0.3:
            parts.append(f"swing feel ({self.swing:.1f})")
        parts.append(f"dissonance level {self.dissonance:.1f}")
        parts.append(", ".join(self.mood_words))
        return ", ".join(parts)


# ─── Confidence → Band Mapping ───────────────────────────────────────────────

BAND_BOUNDARIES = [
    # (min, max, band)
    (0.00, 0.30, ConfidenceBand.UNCERTAIN),
    (0.30, 0.40, ConfidenceBand.TRANSITIONAL),
    (0.40, 0.60, ConfidenceBand.CREATIVE),
    (0.60, 0.70, ConfidenceBand.TRANSITIONAL),
    (0.70, 0.85, ConfidenceBand.EMERGING),
    (0.85, 0.86, ConfidenceBand.TRANSITIONAL),
    (0.86, 1.01, ConfidenceBand.CONFIDENT),
]


def get_band(confidence: float) -> ConfidenceBand:
    """Map a confidence value (0.0-1.0) to a musical band."""
    confidence = max(0.0, min(1.0, confidence))
    for low, high, band in BAND_BOUNDARIES:
        if low <= confidence < high:
            return band
    return ConfidenceBand.CONFIDENT  # edge case at exactly 1.0


# ─── Band → Musical Parameters ───────────────────────────────────────────────

UNCERTAIN_PROFILES = [
    MusicalParameters(
        key="D minor", mode="minor", root_note="D",
        tempo_bpm=50, time_signature="4/4", swing=0.2,
        chord_quality="minor7", resolution="unresolved", dissonance=0.7,
        primary_instrument="trombone", secondary_instruments=["soft piano", "ambient drone"],
        texture="sparse", dynamics="pp", articulation="legato",
        mood_words=["melancholy", "searching", "lost in fog", "blue notes", "unresolved yearning"],
        duration_seconds=45, band="uncertain",
    ),
    MusicalParameters(
        key="G minor", mode="minor", root_note="G",
        tempo_bpm=55, time_signature="3/4", swing=0.1,
        chord_quality="minor7", resolution="unresolved", dissonance=0.6,
        primary_instrument="cello", secondary_instruments=["trombone", "whispered piano"],
        texture="sparse", dynamics="pp", articulation="legato",
        mood_words=["wandering", "uncertain", "dim corridors", "questioning", "low blue light"],
        duration_seconds=50, band="uncertain",
    ),
    MusicalParameters(
        key="F# minor", mode="minor", root_note="F#",
        tempo_bpm=60, time_signature="6/8", swing=0.15,
        chord_quality="minor9", resolution="unresolved", dissonance=0.8,
        primary_instrument="trombone", secondary_instruments=["ambient pad", "distant trumpet"],
        texture="sparse", dynamics="p", articulation="blue",
        mood_words=["aching", "deep uncertainty", "searching for meaning", "blue note longing"],
        duration_seconds=55, band="uncertain",
    ),
]

CREATIVE_PROFILES = [
    MusicalParameters(
        key="Bb", mode="blues", root_note="Bb",
        tempo_bpm=75, time_signature="4/4", swing=0.7,
        chord_quality="dominant7", resolution="tension", dissonance=0.4,
        primary_instrument="saxophone", secondary_instruments=["jazz piano", "upright bass", "brushed drums"],
        texture="layered", dynamics="mf", articulation="mixed",
        mood_words=["jazz", "blue notes", "exploratory", "creative flow", "modal jazz", "kind of blue"],
        duration_seconds=50, band="creative",
    ),
    MusicalParameters(
        key="D dorian", mode="dorian", root_note="D",
        tempo_bpm=80, time_signature="4/4", swing=0.6,
        chord_quality="sus4", resolution="tension", dissonance=0.35,
        primary_instrument="saxophone", secondary_instruments=["Fender Rhodes", "bass", "drums"],
        texture="layered", dynamics="mf", articulation="mixed",
        mood_words=["modal jazz", "so what", "cool exploration", "suspended", "floating chords"],
        duration_seconds=55, band="creative",
    ),
    MusicalParameters(
        key="F mixolydian", mode="mixolydian", root_note="F",
        tempo_bpm=85, time_signature="5/4", swing=0.65,
        chord_quality="sus4", resolution="tension", dissonance=0.45,
        primary_instrument="saxophone", secondary_instruments=["jazz piano", "bass", "drums"],
        texture="layered", dynamics="mf", articulation="staccato",
        mood_words=["time steps", "odd meter jazz", "exploratory", "take five", "curiosity"],
        duration_seconds=45, band="creative",
    ),
    MusicalParameters(
        key="C", mode="blues", root_note="C",
        tempo_bpm=90, time_signature="12/8", swing=0.75,
        chord_quality="dominant7", resolution="tension", dissonance=0.5,
        primary_instrument="saxophone", secondary_instruments=["hammond organ", "bass", "drums"],
        texture="dense", dynamics="f", articulation="mixed",
        mood_words=["soul jazz", "deep groove", "creative fire", "blue note records", "exploration"],
        duration_seconds=60, band="creative",
    ),
]

EMERGING_PROFILES = [
    MusicalParameters(
        key="C major", mode="major", root_note="C",
        tempo_bpm=90, time_signature="4/4", swing=0.3,
        chord_quality="maj7", resolution="resolving", dissonance=0.2,
        primary_instrument="piano", secondary_instruments=["strings", "soft brass"],
        texture="layered", dynamics="mf", articulation="legato",
        mood_words=["resolving", "finding clarity", "dawning understanding", "warm major"],
        duration_seconds=40, band="emerging",
    ),
    MusicalParameters(
        key="G major", mode="major", root_note="G",
        tempo_bpm=100, time_signature="4/4", swing=0.2,
        chord_quality="maj7", resolution="resolving", dissonance=0.15,
        primary_instrument="piano", secondary_instruments=["guitar", "light percussion"],
        texture="layered", dynamics="mf", articulation="mixed",
        mood_words=["confident exploration", "major key journey", "structured but curious"],
        duration_seconds=45, band="emerging",
    ),
    MusicalParameters(
        key="F major", mode="major", root_note="F",
        tempo_bpm=110, time_signature="4/4", swing=0.25,
        chord_quality="maj9", resolution="resolving", dissonance=0.2,
        primary_instrument="trumpet", secondary_instruments=["piano", "bass", "drums"],
        texture="layered", dynamics="mf", articulation="mixed",
        mood_words=["arriving", "taking shape", "confident major", "structure emerging"],
        duration_seconds=40, band="emerging",
    ),
    MusicalParameters(
        key="D major", mode="major", root_note="D",
        tempo_bpm=120, time_signature="4/4", swing=0.15,
        chord_quality="maj7", resolution="resolved", dissonance=0.1,
        primary_instrument="piano", secondary_instruments=["strings", "percussion"],
        texture="layered", dynamics="f", articulation="mixed",
        mood_words=["resolved", "clear understanding", "bright major", "confidence rising"],
        duration_seconds=35, band="emerging",
    ),
]

CONFIDENT_PROFILES = [
    MusicalParameters(
        key="D major", mode="major", root_note="D",
        tempo_bpm=120, time_signature="4/4", swing=0.1,
        chord_quality="maj7", resolution="resolved", dissonance=0.05,
        primary_instrument="trumpet", secondary_instruments=["full orchestra", "timpani"],
        texture="dense", dynamics="f", articulation="mixed",
        mood_words=["triumphant", "celebratory", "bright", "confident arrival", "fanfare"],
        duration_seconds=40, band="confident",
    ),
    MusicalParameters(
        key="A major", mode="major", root_note="A",
        tempo_bpm=140, time_signature="4/4", swing=0.05,
        chord_quality="power", resolution="resolved", dissonance=0.05,
        primary_instrument="trumpet", secondary_instruments=["brass section", "strings", "drums"],
        texture="dense", dynamics="ff", articulation="staccato",
        mood_words=["celebration", "victorious", "bright major", "exhilarating", "arrival"],
        duration_seconds=35, band="confident",
    ),
    MusicalParameters(
        key="C major", mode="major", root_note="C",
        tempo_bpm=160, time_signature="4/4", swing=0.0,
        chord_quality="maj7", resolution="resolved", dissonance=0.0,
        primary_instrument="piano", secondary_instruments=["full orchestra", "cymbals"],
        texture="dense", dynamics="ff", articulation="staccato",
        mood_words=["exuberant", "pure joy", "celebratory", "bright", "ecstatic resolution"],
        duration_seconds=30, band="confident",
    ),
    MusicalParameters(
        key="Eb major", mode="major", root_note="Eb",
        tempo_bpm=130, time_signature="6/8", swing=0.2,
        chord_quality="maj7", resolution="resolved", dissonance=0.08,
        primary_instrument="trumpet", secondary_instruments=["gospel organ", "choir", "brass"],
        texture="dense", dynamics="f", articulation="mixed",
        mood_words=["gospel celebration", "warm triumph", "arrived home", "glorious major"],
        duration_seconds=40, band="confident",
    ),
]

TRANSITIONAL_PROFILES = [
    MusicalParameters(
        key="A", mode="aeolian", root_note="A",
        tempo_bpm=65, time_signature="4/4", swing=0.3,
        chord_quality="sus2", resolution="tension", dissonance=0.3,
        primary_instrument="piano", secondary_instruments=["ambient pad", "bell"],
        texture="sparse", dynamics="mp", articulation="legato",
        mood_words=["shimmering", "between states", "transforming", "crystalline"],
        duration_seconds=20, band="transitional",
    ),
    MusicalParameters(
        key="E", mode="lydian", root_note="E",
        tempo_bpm=70, time_signature="4/4", swing=0.25,
        chord_quality="maj7#11", resolution="tension", dissonance=0.25,
        primary_instrument="piano", secondary_instruments=["strings", "bell"],
        texture="sparse", dynamics="p", articulation="legato",
        mood_words=["floating", "weightless", "between worlds", "shimmering transition"],
        duration_seconds=15, band="transitional",
    ),
]

BAND_PROFILES = {
    ConfidenceBand.UNCERTAIN: UNCERTAIN_PROFILES,
    ConfidenceBand.CREATIVE: CREATIVE_PROFILES,
    ConfidenceBand.EMERGING: EMERGING_PROFILES,
    ConfidenceBand.CONFIDENT: CONFIDENT_PROFILES,
    ConfidenceBand.TRANSITIONAL: TRANSITIONAL_PROFILES,
}

# ─── Emotional State → Musical Modifiers ─────────────────────────────────────

EMOTIONAL_MODIFIERS = {
    EmotionalState.LOST: {"tempo_modifier": -5, "dynamics_shift": -1, "mood_add": ["adrift"]},
    EmotionalState.SEARCHING: {"tempo_modifier": 0, "dynamics_shift": 0, "mood_add": ["hunting"]},
    EmotionalState.EXPLORING: {"tempo_modifier": +5, "dynamics_shift": +1, "mood_add": ["curious"]},
    EmotionalState.WONDERING: {"tempo_modifier": -3, "dynamics_shift": -1, "mood_add": ["awe"]},
    EmotionalState.DISCOVERING: {"tempo_modifier": +3, "dynamics_shift": +1, "mood_add": ["revelation"]},
    EmotionalState.CONFIDENT: {"tempo_modifier": +5, "dynamics_shift": +1, "mood_add": ["assured"]},
    EmotionalState.TRIUMPHANT: {"tempo_modifier": +10, "dynamics_shift": +2, "mood_add": ["victorious"]},
    EmotionalState.CONTEMPLATIVE: {"tempo_modifier": -8, "dynamics_shift": -1, "mood_add": ["reflective"]},
    EmotionalState.PLAYFUL: {"tempo_modifier": +8, "dynamics_shift": +1, "mood_add": ["playful"]},
    EmotionalState.MELANCHOLY: {"tempo_modifier": -5, "dynamics_shift": -2, "mood_add": ["sorrowful"]},
}

DYNAMICS_ORDER = ["pp", "p", "mp", "mf", "f", "ff"]


def apply_emotional_modifier(
    params: MusicalParameters, emotion: EmotionalState
) -> MusicalParameters:
    """Apply an emotional state modifier to a set of musical parameters."""
    mod = EMOTIONAL_MODIFIERS.get(emotion, {})
    tempo = max(40, min(180, params.tempo_bpm + mod.get("tempo_modifier", 0)))

    dyn_idx = DYNAMICS_ORDER.index(params.dynamics) if params.dynamics in DYNAMICS_ORDER else 3
    new_dyn_idx = max(0, min(len(DYNAMICS_ORDER) - 1, dyn_idx + mod.get("dynamics_shift", 0)))

    mood = list(params.mood_words) + mod.get("mood_add", [])

    return MusicalParameters(
        key=params.key, mode=params.mode, root_note=params.root_note,
        tempo_bpm=tempo, time_signature=params.time_signature, swing=params.swing,
        chord_quality=params.chord_quality, resolution=params.resolution,
        dissonance=params.dissonance,
        primary_instrument=params.primary_instrument,
        secondary_instruments=params.secondary_instruments,
        texture=params.texture,
        dynamics=DYNAMICS_ORDER[new_dyn_idx],
        articulation=params.articulation,
        mood_words=mood, duration_seconds=params.duration_seconds, band=params.band,
    )


# ─── Main API ────────────────────────────────────────────────────────────────

import random


def confidence_to_music(
    confidence: float,
    emotion: Optional[EmotionalState] = None,
    seed: Optional[int] = None,
) -> MusicalParameters:
    """
    The primary API: map a confidence level (and optional emotional state)
    to a complete set of musical parameters ready for MMX generation.

    Args:
        confidence: 0.0 to 1.0
        emotion: optional emotional state to color the output
        seed: for reproducible profile selection
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    band = get_band(confidence)
    profiles = BAND_PROFILES.get(band, CREATIVE_PROFILES)
    params = rng.choice(profiles)

    if emotion:
        params = apply_emotional_modifier(params, emotion)

    # Regenerate the MMX prompt
    params.mmx_prompt = params.to_mmx_prompt()
    return params


def get_all_profiles() -> dict[str, list[MusicalParameters]]:
    """Return all profiles organized by band name."""
    return {band.value: profiles for band, profiles in BAND_PROFILES.items()}


def summarize_mapping() -> str:
    """Human-readable summary of the confidence → music mapping."""
    lines = [
        "SONIC SHAPE ENGINE — Harmonic Dictionary",
        "=" * 50,
        "",
        "CONFIDENCE BANDS:",
        f"  0.00 - 0.30  UNCERTAIN    →  minor key, 50-60 BPM, trombone, unresolved",
        f"  0.30 - 0.40  TRANSITIONAL →  shimmering, suspended, between states",
        f"  0.40 - 0.60  CREATIVE     →  blue notes, 65-90 BPM, jazz, suspended chords",
        f"  0.60 - 0.70  TRANSITIONAL →  shimmering, suspended, between states",
        f"  0.70 - 0.85  EMERGING     →  major key, 90-120 BPM, resolving",
        f"  0.85 - 0.86  TRANSITIONAL →  shimmering, suspended, between states",
        f"  0.86 - 1.00  CONFIDENT    →  bright major, 120-160 BPM, celebratory",
        "",
        f"Total profiles: {sum(len(v) for v in BAND_PROFILES.values())}",
        f"Emotional states: {len(EmotionalState)}",
        "",
        "The music IS the system thinking.",
    ]
    return "\n".join(lines)
