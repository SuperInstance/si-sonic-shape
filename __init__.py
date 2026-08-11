# Sonic Shape package entry point
import sys
import os

_src = os.path.join(os.path.dirname(__file__), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from harmonic_dictionary import (
    ConfidenceBand,
    EmotionalState,
    MusicalParameters,
    confidence_to_music,
    get_band,
    get_all_profiles,
    summarize_mapping,
)
from voice_profiles import VoiceProfile, get_voice, list_voices, ensemble_prompt
from session_to_music import (
    SessionContribution,
    MusicalMovement,
    MusicalScore,
    parse_session_log,
    session_to_score,
    generate_playlist,
)
from live_generator import LiveGenerator, QueuedPiece, ConfidenceSnapshot, create_default_generator

__version__ = "0.1.0"
