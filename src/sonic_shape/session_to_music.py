"""
SESSION TO MUSIC — Transforms a Tap session log into a musical score.

This module analyzes the emotional arc of a conversation and maps it to
a sequence of musical movements. Each model contribution becomes a voice,
and the confidence trajectory becomes the harmonic journey.

Input:  A Tap session log (JSON or dict)
Output: A MusicalScore containing movements, each with MMX generation commands

The process:
    1. Parse the session log into contributions
    2. Analyze confidence trajectory and emotional states
    3. Segment the session into musical movements
    4. Assign voices to each model
    5. Generate MMX commands for each movement
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from harmonic_dictionary import (
    ConfidenceBand,
    EmotionalState,
    MusicalParameters,
    confidence_to_music,
    get_band,
)
from voice_profiles import VoiceProfile, get_voice


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class SessionContribution:
    """A single model contribution in a session."""
    model_name: str
    timestamp: Optional[str] = None
    content: str = ""
    confidence: float = 0.5
    role: str = "assistant"        # "assistant", "user", "system"
    metadata: dict = field(default_factory=dict)


@dataclass
class MusicalMovement:
    """A single movement in the musical score, derived from a session segment."""
    index: int                          # movement number
    start_confidence: float             # confidence at start
    end_confidence: float               # confidence at end
    dominant_band: ConfidenceBand        # primary musical band
    emotion: Optional[EmotionalState]    # emotional color
    parameters: MusicalParameters        # musical settings
    voices: list[VoiceProfile]           # models active in this movement
    mmx_command: str = ""               # full MMX generation command
    duration_seconds: int = 45
    label: str = ""                     # human-readable movement title

    def build_mmx_command(self) -> str:
        """Build the MMX CLI command for this movement."""
        voice_parts = [v.mmx_voice_prompt for v in self.voices]
        full_prompt = (
            f"{self.parameters.to_mmx_prompt()}. "
            f"Voices: {' | '.join(voice_parts)}. "
            f"Confidence arc: {self.start_confidence:.2f} → {self.end_confidence:.2f}. "
            f"Movement {self.index}: {self.label}."
        )
        self.mmx_command = f"mmx music --prompt \"{full_prompt}\" --duration {self.duration_seconds}"
        return self.mmx_command


@dataclass
class MusicalScore:
    """A complete musical score derived from a session."""
    session_id: str
    movements: list[MusicalMovement] = field(default_factory=list)
    total_duration_seconds: int = 0
    confidence_arc: list[float] = field(default_factory=list)
    models_present: list[str] = field(default_factory=list)
    generated_at: str = ""

    def summary(self) -> str:
        """Human-readable summary of the score."""
        lines = [
            f"MUSICAL SCORE: Session {self.session_id}",
            f"Generated: {self.generated_at}",
            f"Total duration: {self.total_duration_seconds}s ({self.total_duration_seconds // 60}m {self.total_duration_seconds % 60}s)",
            f"Models: {', '.join(self.models_present)}",
            f"Movements: {len(self.movements)}",
            f"Confidence arc: {' → '.join(f'{c:.2f}' for c in self.confidence_arc)}",
            "",
            "MOVEMENTS:",
        ]
        for m in self.movements:
            lines.append(
                f"  {m.index}. [{m.dominant_band.value}] {m.label} "
                f"({m.duration_seconds}s, {m.parameters.tempo_bpm} BPM, "
                f"{m.parameters.key}) — "
                f"voices: {', '.join(v.display_name for v in m.voices)}"
            )
        lines.append("")
        lines.append("The music IS the system thinking.")
        return "\n".join(lines)

    def to_playlist(self) -> list[dict]:
        """Export as a playlist of MMX commands."""
        return [
            {
                "track": m.index,
                "title": m.label,
                "duration_seconds": m.duration_seconds,
                "band": m.dominant_band.value,
                "key": m.parameters.key,
                "tempo_bpm": m.parameters.tempo_bpm,
                "mmx_command": m.build_mmx_command(),
                "voices": [v.model_name for v in m.voices],
            }
            for m in self.movements
        ]

    def to_json(self) -> str:
        """Export the full score as JSON."""
        return json.dumps({
            "session_id": self.session_id,
            "generated_at": self.generated_at,
            "total_duration_seconds": self.total_duration_seconds,
            "models_present": self.models_present,
            "confidence_arc": self.confidence_arc,
            "movements": [
                {
                    "index": m.index,
                    "label": m.label,
                    "band": m.dominant_band.value,
                    "start_confidence": m.start_confidence,
                    "end_confidence": m.end_confidence,
                    "emotion": m.emotion.value if m.emotion else None,
                    "duration_seconds": m.duration_seconds,
                    "parameters": {
                        "key": m.parameters.key,
                        "mode": m.parameters.mode,
                        "tempo_bpm": m.parameters.tempo_bpm,
                        "time_signature": m.parameters.time_signature,
                        "primary_instrument": m.parameters.primary_instrument,
                        "chord_quality": m.parameters.chord_quality,
                        "resolution": m.parameters.resolution,
                        "dynamics": m.parameters.dynamics,
                    },
                    "voices": [v.model_name for v in m.voices],
                    "mmx_command": m.build_mmx_command(),
                }
                for m in self.movements
            ],
        }, indent=2)


# ─── Session Log Parsing ─────────────────────────────────────────────────────

def parse_session_log(session_data: dict | str) -> list[SessionContribution]:
    """
    Parse a Tap session log into a list of contributions.

    Supports multiple formats:
        - {"messages": [...]}
        - {"contributions": [...]}
        - {"turns": [...]}
        - Raw list of messages
    """
    if isinstance(session_data, str):
        session_data = json.loads(session_data)

    # Try various session log formats
    raw_messages = (
        session_data.get("messages")
        or session_data.get("contributions")
        or session_data.get("turns")
        or (session_data if isinstance(session_data, list) else [])
    )

    contributions = []
    for msg in raw_messages:
        if isinstance(msg, str):
            contributions.append(SessionContribution(
                model_name="unknown",
                content=msg,
                confidence=0.5,
            ))
        elif isinstance(msg, dict):
            model = (
                msg.get("model")
                or msg.get("model_name")
                or msg.get("source")
                or "unknown"
            )
            conf = (
                msg.get("confidence")
                or msg.get("certainty")
                or msg.get("score")
                or 0.5
            )
            contributions.append(SessionContribution(
                model_name=str(model),
                timestamp=msg.get("timestamp") or msg.get("time"),
                content=msg.get("content") or msg.get("text") or msg.get("message") or "",
                confidence=float(conf),
                role=msg.get("role", "assistant"),
                metadata=msg.get("metadata", {}),
            ))

    return contributions


# ─── Emotional Analysis ──────────────────────────────────────────────────────

# Keywords that hint at emotional states
EMOTION_KEYWORDS: dict[EmotionalState, list[str]] = {
    EmotionalState.LOST: ["lost", "confused", "don't know", "unclear", "stuck"],
    EmotionalState.SEARCHING: ["searching", "looking", "trying", "seeking", "investigating"],
    EmotionalState.EXPLORING: ["exploring", "discovered", "found", "interesting", "let's try"],
    EmotionalState.WONDERING: ["wonder", "curious", "fascinating", "marvelous", "imagine"],
    EmotionalState.DISCOVERING: ["breakthrough", "aha", "realized", "figured out", "insight"],
    EmotionalState.CONFIDENT: ["certain", "confident", "clear", "sure", "definitely"],
    EmotionalState.TRIUMPHANT: ["solved", "complete", "done", "success", "perfect"],
    EmotionalState.CONTEMPLATIVE: ["reflect", "consider", "think about", "ponder", "contemplate"],
    EmotionalState.PLAYFUL: ["fun", "playful", "silly", "happy", "enjoy"],
    EmotionalState.MELANCHOLY: ["sad", "unfortunately", "regret", "sorry", "melancholy"],
}


def infer_emotion(content: str, confidence: float) -> EmotionalState:
    """Infer an emotional state from message content and confidence level."""
    content_lower = content.lower()

    # Score each emotion by keyword matches
    scores: dict[EmotionalState, int] = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[emotion] = score

    if scores:
        return max(scores, key=scores.get)

    # Fall back to confidence-based inference
    if confidence < 0.2:
        return EmotionalState.LOST
    elif confidence < 0.35:
        return EmotionalState.SEARCHING
    elif confidence < 0.5:
        return EmotionalState.WONDERING
    elif confidence < 0.65:
        return EmotionalState.EXPLORING
    elif confidence < 0.8:
        return EmotionalState.DISCOVERING
    elif confidence < 0.9:
        return EmotionalState.CONFIDENT
    else:
        return EmotionalState.TRIUMPHANT


# ─── Movement Segmentation ───────────────────────────────────────────────────

# Minimum contributions per movement
MIN_MOVEMENT_LENGTH = 2
# Maximum contributions per movement
MAX_MOVEMENT_LENGTH = 8


def segment_into_movements(
    contributions: list[SessionContribution],
) -> list[list[SessionContribution]]:
    """
    Segment the session into musical movements based on confidence transitions.

    A new movement starts when:
        - The confidence band changes significantly
        - The maximum movement length is reached
        - There's a large confidence jump (> 0.2)
    """
    if not contributions:
        return []

    movements: list[list[SessionContribution]] = []
    current: list[SessionContribution] = [contributions[0]]

    for i in range(1, len(contributions)):
        prev_conf = contributions[i - 1].confidence
        curr_conf = contributions[i].confidence
        prev_band = get_band(prev_conf)
        curr_band = get_band(curr_conf)
        jump = abs(curr_conf - prev_conf)

        # Check if we should start a new movement
        band_change = prev_band != curr_band
        big_jump = jump > 0.2
        too_long = len(current) >= MAX_MOVEMENT_LENGTH

        if (band_change or big_jump or too_long) and len(current) >= MIN_MOVEMENT_LENGTH:
            movements.append(current)
            current = [contributions[i]]
        else:
            current.append(contributions[i])

    if current:
        movements.append(current)

    return movements


# ─── Movement Labels ─────────────────────────────────────────────────────────

BAND_LABELS = {
    ConfidenceBand.UNCERTAIN: ["Into the Mist", "Unresolved Question", "Dim Corridors", "The Search"],
    ConfidenceBand.CREATIVE: ["Kind of Blue", "Explorations", "The Creative Band", "Modal Wanderings",
                               "Blue Note Suite", "Floating Chords", "Curiosity Dance"],
    ConfidenceBand.EMERGING: ["Dawning", "Taking Shape", "Clarion Call", "Rising Structure"],
    ConfidenceBand.CONFIDENT: ["Arrival", "Triumph", "Celebration", "Homecoming"],
    ConfidenceBand.TRANSITIONAL: ["Shimmering", "Between Worlds", "Transformations"],
}


def label_for_movement(
    index: int,
    band: ConfidenceBand,
    emotion: Optional[EmotionalState],
    seed: Optional[int] = None,
) -> str:
    """Generate a poetic label for a movement."""
    import random
    rng = random.Random(seed if seed is not None else index * 42)
    labels = BAND_LABELS.get(band, ["Untitled"])
    base = rng.choice(labels)
    if emotion:
        return f"{base} ({emotion.value})"
    return base


# ─── Main API ────────────────────────────────────────────────────────────────

def session_to_score(
    session_data: dict | str | list,
    session_id: str = "unknown",
    seed: Optional[int] = None,
) -> MusicalScore:
    """
    Transform a Tap session log into a complete musical score.

    Args:
        session_data: The session log (dict, JSON string, or list of messages)
        session_id: Identifier for the session
        seed: For reproducible output

    Returns:
        A MusicalScore with movements ready for MMX generation
    """
    contributions = parse_session_log(session_data)

    # Extract models present
    models_present = list(dict.fromkeys(c.model_name for c in contributions))

    # Confidence arc
    confidence_arc = [c.confidence for c in contributions]

    # Segment into movements
    movement_segments = segment_into_movements(contributions)

    # Build movements
    movements: list[MusicalMovement] = []
    for i, segment in enumerate(movement_segments):
        avg_conf = sum(c.confidence for c in segment) / len(segment)
        start_conf = segment[0].confidence
        end_conf = segment[-1].confidence
        band = get_band(avg_conf)

        # Infer dominant emotion
        emotions = [infer_emotion(c.content, c.confidence) for c in segment]
        from collections import Counter
        emotion_counts = Counter(emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotions else None

        # Get musical parameters
        params = confidence_to_music(avg_conf, dominant_emotion, seed=seed)

        # Get voices for this segment
        segment_models = list(dict.fromkeys(c.model_name for c in segment))
        voices = [get_voice(m) for m in segment_models]

        label = label_for_movement(i, band, dominant_emotion, seed=seed)

        movement = MusicalMovement(
            index=i + 1,
            start_confidence=start_conf,
            end_confidence=end_conf,
            dominant_band=band,
            emotion=dominant_emotion,
            parameters=params,
            voices=voices,
            duration_seconds=params.duration_seconds,
            label=label,
        )
        movement.build_mmx_command()
        movements.append(movement)

    total_duration = sum(m.duration_seconds for m in movements)

    return MusicalScore(
        session_id=session_id,
        movements=movements,
        total_duration_seconds=total_duration,
        confidence_arc=confidence_arc,
        models_present=models_present,
        generated_at=datetime.now().isoformat(),
    )


def generate_playlist(score: MusicalScore, output_path: Optional[str] = None) -> str:
    """
    Generate a playlist of MMX commands from a musical score.
    Optionally write to a file.
    """
    playlist = score.to_playlist()
    json_out = json.dumps(playlist, indent=2)

    if output_path:
        with open(output_path, "w") as f:
            f.write(json_out)

    return json_out
