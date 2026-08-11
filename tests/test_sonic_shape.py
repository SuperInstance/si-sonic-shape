"""
Tests for the Sonic Shape Engine — 14 tests covering all modules.

Run: python -m pytest tests/ -v
Or:  python tests/run_tests.py
"""

import sys
import os
import json
import math
import asyncio

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harmonic_dictionary import (
    ConfidenceBand,
    EmotionalState,
    MusicalParameters,
    confidence_to_music,
    get_band,
    get_all_profiles,
    apply_emotional_modifier,
    summarize_mapping,
    DYNAMICS_ORDER,
)
from voice_profiles import (
    VoiceProfile,
    get_voice,
    list_voices,
    ensemble_prompt,
    MODEL_VOICES,
)
from session_to_music import (
    SessionContribution,
    MusicalMovement,
    MusicalScore,
    parse_session_log,
    infer_emotion,
    segment_into_movements,
    session_to_score,
    generate_playlist,
)
from live_generator import (
    LiveGenerator,
    QueuedPiece,
    GeneratorState,
    ConfidenceSnapshot,
    create_default_generator,
)


# ─── Test Runner ─────────────────────────────────────────────────────────────

class TestRunner:
    """Simple test runner to avoid pytest dependency."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run(self, name, func):
        try:
            func()
            print(f"  ✓ {name}")
            self.passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            self.failed += 1
            self.errors.append((name, str(e)))
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            self.failed += 1
            self.errors.append((name, f"{type(e).__name__}: {e}"))

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for name, err in self.errors:
                print(f"  - {name}: {err}")
        return self.failed == 0


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_band_mapping_boundaries():
    """Test 1: Confidence values map to correct bands at boundaries."""
    assert get_band(0.0) == ConfidenceBand.UNCERTAIN
    assert get_band(0.15) == ConfidenceBand.UNCERTAIN
    assert get_band(0.29) == ConfidenceBand.UNCERTAIN
    assert get_band(0.30) == ConfidenceBand.TRANSITIONAL
    assert get_band(0.39) == ConfidenceBand.TRANSITIONAL
    assert get_band(0.40) == ConfidenceBand.CREATIVE
    assert get_band(0.50) == ConfidenceBand.CREATIVE
    assert get_band(0.59) == ConfidenceBand.CREATIVE
    assert get_band(0.60) == ConfidenceBand.TRANSITIONAL
    assert get_band(0.69) == ConfidenceBand.TRANSITIONAL
    assert get_band(0.70) == ConfidenceBand.EMERGING
    assert get_band(0.84) == ConfidenceBand.EMERGING
    assert get_band(0.86) == ConfidenceBand.CONFIDENT
    assert get_band(1.0) == ConfidenceBand.CONFIDENT


def test_band_clamping():
    """Test 2: Confidence values are clamped to 0-1 range."""
    assert get_band(-0.5) == ConfidenceBand.UNCERTAIN
    assert get_band(1.5) == ConfidenceBand.CONFIDENT
    assert get_band(0.5) == ConfidenceBand.CREATIVE


def test_confidence_to_music_uncertain():
    """Test 3: Low confidence produces uncertain music (minor key, slow, trombone)."""
    params = confidence_to_music(0.1, seed=42)
    assert params.band == "uncertain", f"Expected 'uncertain', got '{params.band}'"
    assert "minor" in params.mode or "minor" in params.key.lower(), \
        f"Expected minor key for uncertain, got {params.key} / {params.mode}"
    assert params.tempo_bpm <= 65, f"Expected slow tempo (≤65), got {params.tempo_bpm}"
    assert params.resolution == "unresolved", f"Expected unresolved, got {params.resolution}"


def test_confidence_to_music_creative():
    """Test 4: Mid confidence (0.4-0.6) produces jazz/blue note music."""
    params = confidence_to_music(0.5, seed=42)
    assert params.band == "creative", f"Expected 'creative', got '{params.band}'"
    assert params.tempo_bpm >= 65, f"Expected ≥65 BPM, got {params.tempo_bpm}"
    assert params.tempo_bpm <= 95, f"Expected ≤95 BPM, got {params.tempo_bpm}"
    assert params.swing >= 0.5, f"Expected swing feel, got {params.swing}"
    # Should be blues/dorian/mixolydian
    assert params.mode in ("blues", "dorian", "mixolydian"), \
        f"Expected exploratory mode, got {params.mode}"


def test_confidence_to_music_confident():
    """Test 5: High confidence produces bright major celebratory music."""
    params = confidence_to_music(0.95, seed=42)
    assert params.band == "confident", f"Expected 'confident', got '{params.band}'"
    assert params.tempo_bpm >= 120, f"Expected ≥120 BPM, got {params.tempo_bpm}"
    assert params.resolution == "resolved", f"Expected resolved, got {params.resolution}"
    assert "major" in params.mode, f"Expected major mode, got {params.mode}"


def test_emotional_modifier():
    """Test 6: Emotional states modify musical parameters correctly."""
    base = confidence_to_music(0.5, seed=42)
    modified = apply_emotional_modifier(base, EmotionalState.PLAYFUL)
    assert modified.tempo_bpm > base.tempo_bpm, \
        f"Playful should increase tempo: {base.tempo_bpm} → {modified.tempo_bpm}"

    sad = apply_emotional_modifier(base, EmotionalState.MELANCHOLY)
    assert sad.tempo_bpm < base.tempo_bpm, \
        f"Melancholy should decrease tempo: {base.tempo_bpm} → {sad.tempo_bpm}"


def test_all_profiles_have_required_fields():
    """Test 7: All musical profiles have complete required fields."""
    profiles = get_all_profiles()
    for band_name, band_profiles in profiles.items():
        for p in band_profiles:
            assert p.key, f"Profile in {band_name} missing key"
            assert p.tempo_bpm >= 40 and p.tempo_bpm <= 180, \
                f"Profile tempo {p.tempo_bpm} out of range in {band_name}"
            assert p.primary_instrument, f"Profile in {band_name} missing instrument"
            assert len(p.mood_words) > 0, f"Profile in {band_name} missing mood words"
            assert p.band == band_name, \
                f"Profile band mismatch: expected {band_name}, got {p.band}"


def test_voice_profiles_exist():
    """Test 8: All four core model voices exist with correct instruments."""
    flash = get_voice("flash")
    assert "sax" in flash.primary_instrument.lower(), \
        f"Flash should use saxophone, got {flash.primary_instrument}"

    pro = get_voice("pro")
    assert "cello" in pro.primary_instrument.lower(), \
        f"Pro should use cello, got {pro.primary_instrument}"

    hermes = get_voice("hermes")
    assert "rhodes" in hermes.primary_instrument.lower() or "piano" in hermes.primary_instrument.lower(), \
        f"Hermes should use Rhodes/piano, got {hermes.primary_instrument}"

    wesley = get_voice("wesley")
    assert "bell" in wesley.primary_instrument.lower() or "box" in wesley.primary_instrument.lower(), \
        f"Wesley should use bell/music box, got {wesley.primary_instrument}"


def test_voice_profile_aliases():
    """Test 9: Model name aliases resolve correctly."""
    assert get_voice("deepseek-v4-flash").display_name == "Flash"
    assert get_voice("deepseek-v4-pro").display_name == "Pro"
    assert get_voice("hermes-3").display_name == "Hermes"
    assert get_voice("haiku").display_name == "Wesley"

    # Unknown model gets a fallback
    unknown = get_voice("some-random-model")
    assert unknown.primary_instrument == "piano"


def test_session_parsing():
    """Test 10: Session logs parse correctly in multiple formats."""
    # Messages format
    session1 = {
        "messages": [
            {"model": "flash", "content": "hello", "confidence": 0.5},
            {"model": "pro", "content": "world", "confidence": 0.7},
        ]
    }
    contribs = parse_session_log(session1)
    assert len(contribs) == 2
    assert contribs[0].model_name == "flash"
    assert contribs[0].confidence == 0.5

    # Turns format
    session2 = {
        "turns": [
            {"model": "hermes", "text": "hi", "confidence": 0.3},
        ]
    }
    contribs2 = parse_session_log(session2)
    assert len(contribs2) == 1
    assert contribs2[0].model_name == "hermes"

    # JSON string
    json_str = json.dumps({"messages": [{"model": "flash", "content": "test", "confidence": 0.8}]})
    contribs3 = parse_session_log(json_str)
    assert len(contribs3) == 1


def test_emotion_inference():
    """Test 11: Emotional states are inferred from message content."""
    assert infer_emotion("I'm lost and confused", 0.1) == EmotionalState.LOST
    assert infer_emotion("That's fascinating, I wonder", 0.4) == EmotionalState.WONDERING
    assert infer_emotion("I figured it out! Breakthrough!", 0.6) == EmotionalState.DISCOVERING
    assert infer_emotion("Solved it perfectly", 0.95) == EmotionalState.TRIUMPHANT
    # Fallback to confidence-based
    assert infer_emotion("no keywords here", 0.15) == EmotionalState.LOST


def test_session_to_score():
    """Test 12: A full session converts to a musical score with movements."""
    session = {
        "messages": [
            {"model": "pro", "content": "I'm lost and confused", "confidence": 0.1},
            {"model": "pro", "content": "Let me search more", "confidence": 0.2},
            {"model": "flash", "content": "Interesting discovery here", "confidence": 0.5},
            {"model": "flash", "content": "Exploring the pattern", "confidence": 0.55},
            {"model": "hermes", "content": "Now I'm confident", "confidence": 0.8},
            {"model": "hermes", "content": "Solved it perfectly", "confidence": 0.95},
        ]
    }
    score = session_to_score(session, session_id="test-session", seed=42)

    assert len(score.movements) >= 2, f"Expected ≥2 movements, got {len(score.movements)}"
    assert score.total_duration_seconds > 0
    assert len(score.confidence_arc) == 6
    assert "pro" in score.models_present
    assert "flash" in score.models_present

    # Each movement should have an MMX command
    for m in score.movements:
        assert m.mmx_command.startswith("mmx music"), \
            f"Movement {m.index} missing MMX command"

    # Score summary should mention it's thinking
    assert "thinking" in score.summary().lower()


def test_score_to_playlist():
    """Test 13: A musical score exports as a playlist with MMX commands."""
    session = {
        "messages": [
            {"model": "flash", "content": "Exploring", "confidence": 0.45},
            {"model": "pro", "content": "Certain now", "confidence": 0.9},
        ]
    }
    score = session_to_score(session, session_id="playlist-test", seed=7)
    playlist = score.to_playlist()

    assert len(playlist) >= 1
    for entry in playlist:
        assert "mmx_command" in entry
        assert "title" in entry
        assert "duration_seconds" in entry
        assert "band" in entry

    # Test JSON export
    json_out = score.to_json()
    parsed = json.loads(json_out)
    assert "movements" in parsed


def test_live_generator_feed():
    """Test 14: Live generator responds to confidence feeds correctly."""
    gen = create_default_generator(seed=99)
    pieces = []

    gen.on_generate(lambda p: pieces.append(p))

    # Start generator and feed confidence values
    async def _run_feed():
        await gen.start()
        # Feed uncertain confidence
        gen.feed_confidence(0.1, model_name="pro")
        assert len(pieces) >= 1, f"Expected ≥1 piece after first feed, got {len(pieces)}"
        assert pieces[0].params.band == "uncertain"

        # Feed confident confidence (band change)
        gen.feed_confidence(0.95, model_name="hermes")
        assert len(pieces) >= 2, f"Expected ≥2 pieces after band change, got {len(pieces)}"

        # Queue should have pieces
        assert len(gen.queue) >= 0
        assert gen.total_generated >= 2

        await gen.stop()
        assert gen.state == GeneratorState.STOPPED

    asyncio.run(_run_feed())


def test_live_generator_simulation():
    """Test 15: Live generator simulation runs through bands."""
    gen = create_default_generator(seed=42)
    pieces = []
    gen.on_generate(lambda p: pieces.append(p))

    async def _run():
        await gen.start()
        await gen.run_simulation(duration_seconds=6, models=["flash", "pro"])
    asyncio.run(_run())

    assert len(pieces) >= 2, f"Expected ≥2 pieces from simulation, got {len(pieces)}"
    # Should span multiple bands
    bands = set(p.params.band for p in pieces)
    assert len(bands) >= 2, f"Expected ≥2 bands, got {bands}"

    asyncio.run(gen.stop())


def test_mmx_prompt_generation():
    """Test 16: Musical parameters generate valid MMX prompts."""
    params = confidence_to_music(0.5, seed=1)
    prompt = params.to_mmx_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 20
    assert str(params.tempo_bpm) in prompt
    assert params.key in prompt

    # Voice profile prompt
    voice = get_voice("flash")
    vprompt = voice.build_prompt()
    assert "saxophone" in vprompt.lower() or "alto" in vprompt.lower()

    # Ensemble prompt
    eprompt = ensemble_prompt(["flash", "pro", "hermes"])
    assert "ensemble" in eprompt.lower()


def test_deterministic_output():
    """Test 17: Same seed produces same musical parameters."""
    p1 = confidence_to_music(0.5, seed=123)
    p2 = confidence_to_music(0.5, seed=123)
    assert p1.key == p2.key
    assert p1.tempo_bpm == p2.tempo_bpm
    assert p1.mode == p2.mode


# ─── Run All Tests ───────────────────────────────────────────────────────────

def run_all_tests():
    runner = TestRunner()

    print("\n🎵 SONIC SHAPE ENGINE — Test Suite\n")

    runner.run("Band mapping boundaries", test_band_mapping_boundaries)
    runner.run("Band clamping", test_band_clamping)
    runner.run("Low confidence → uncertain music", test_confidence_to_music_uncertain)
    runner.run("Mid confidence → creative jazz", test_confidence_to_music_creative)
    runner.run("High confidence → bright major", test_confidence_to_music_confident)
    runner.run("Emotional modifiers", test_emotional_modifier)
    runner.run("All profiles have required fields", test_all_profiles_have_required_fields)
    runner.run("Voice profiles exist", test_voice_profiles_exist)
    runner.run("Voice profile aliases", test_voice_profile_aliases)
    runner.run("Session parsing", test_session_parsing)
    runner.run("Emotion inference", test_emotion_inference)
    runner.run("Session to score", test_session_to_score)
    runner.run("Score to playlist", test_score_to_playlist)
    runner.run("Live generator feed", test_live_generator_feed)
    runner.run("Live generator simulation", test_live_generator_simulation)
    runner.run("MMX prompt generation", test_mmx_prompt_generation)
    runner.run("Deterministic output", test_deterministic_output)

    return runner.summary()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
