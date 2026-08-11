"""
LIVE GENERATOR — Real-time music generation driven by session confidence.

This module monitors a Tap session's confidence in real-time and generates
music that mirrors the system's thinking state. When confidence enters the
creative band (0.4-0.6), it triggers jazz generation. When confidence is
high, it generates resolution music. When uncertain, it generates blue notes.

Modes of operation:
    1. WebSocket: Connect to a live session feed
    2. Polling: Poll a session API endpoint
    3. Simulation: Drive from an internal simulator (for testing/demo)

The generator maintains a queue of upcoming pieces, smoothing transitions
between confidence bands so the music flows naturally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from harmonic_dictionary import (
    ConfidenceBand,
    EmotionalState,
    MusicalParameters,
    confidence_to_music,
    get_band,
)
from voice_profiles import get_voice, VoiceProfile

logger = logging.getLogger("sonic_shape.live_generator")


# ─── Queue Item ───────────────────────────────────────────────────────────────

@dataclass
class QueuedPiece:
    """A music piece queued for generation/playback."""
    params: MusicalParameters
    voice: Optional[VoiceProfile]
    mmx_command: str
    generated_at: float = field(default_factory=time.time)
    played: bool = False
    file_path: Optional[str] = None  # path to generated audio file

    def __repr__(self) -> str:
        return (
            f"QueuedPiece({self.params.band}, {self.params.tempo_bpm}BPM, "
            f"{self.params.key}, {self.params.duration_seconds}s)"
        )


# ─── Generator State ─────────────────────────────────────────────────────────

class GeneratorState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"


@dataclass
class ConfidenceSnapshot:
    """A point-in-time capture of session confidence."""
    confidence: float
    timestamp: float = field(default_factory=time.time)
    model_name: str = "unknown"
    emotional_state: Optional[EmotionalState] = None
    band: ConfidenceBand = field(default=ConfidenceBand.CREATIVE)

    def __post_init__(self):
        self.band = get_band(self.confidence)


# ─── The Live Generator ──────────────────────────────────────────────────────

class LiveGenerator:
    """
    Monitors session confidence and generates music in real-time.

    Usage:
        gen = LiveGenerator()
        await gen.start()
        gen.feed_confidence(0.45, model_name="flash")  # creative band → jazz
        gen.feed_confidence(0.9, model_name="pro")     # confident → resolution
        await gen.stop()
    """

    def __init__(
        self,
        queue_size: int = 10,
        poll_interval_seconds: float = 2.0,
        min_pieces_per_band: int = 2,
        max_pieces_per_band: int = 4,
        generation_callback: Optional[Callable[[QueuedPiece], None]] = None,
        auto_generate: bool = True,
        seed: Optional[int] = None,
    ):
        self.queue: deque[QueuedPiece] = deque(maxlen=queue_size)
        self.state: GeneratorState = GeneratorState.IDLE
        self.current_band: Optional[ConfidenceBand] = None
        self.confidence_history: list[ConfidenceSnapshot] = []
        self.poll_interval = poll_interval_seconds
        self.min_pieces_per_band = min_pieces_per_band
        self.max_pieces_per_band = max_pieces_per_band
        self.generation_callback = generation_callback
        self.auto_generate = auto_generate
        self.seed = seed
        self._rng = random.Random(seed)
        self._pieces_in_current_band = 0
        self._task: Optional[asyncio.Task] = None

        # WebSocket / polling config
        self.ws_url: Optional[str] = None
        self.poll_url: Optional[str] = None
        self.api_key: Optional[str] = None

        # Stats
        self.total_generated: int = 0
        self.total_played: int = 0

    # ─── Configuration ─────────────────────────────────────────

    def configure_websocket(self, url: str, api_key: Optional[str] = None):
        """Configure WebSocket-based session monitoring."""
        self.ws_url = url
        self.api_key = api_key

    def configure_polling(self, url: str, interval: float = 2.0, api_key: Optional[str] = None):
        """Configure HTTP polling-based session monitoring."""
        self.poll_url = url
        self.poll_interval = interval
        self.api_key = api_key

    def on_generate(self, callback: Callable[[QueuedPiece], None]):
        """Register a callback called when a new piece is generated."""
        self.generation_callback = callback

    # ─── Lifecycle ─────────────────────────────────────────────

    async def start(self):
        """Start the live generator."""
        if self.state == GeneratorState.RUNNING:
            logger.warning("Generator already running")
            return

        self.state = GeneratorState.RUNNING
        logger.info("Live generator started")

        if self.ws_url:
            self._task = asyncio.create_task(self._run_websocket())
        elif self.poll_url:
            self._task = asyncio.create_task(self._run_polling())
        else:
            # Manual mode — no background task
            self._task = None

    async def stop(self):
        """Stop the generator."""
        self.state = GeneratorState.STOPPED
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Live generator stopped")

    # ─── Manual Feeding ─────────────────────────────────────────

    def feed_confidence(
        self,
        confidence: float,
        model_name: str = "unknown",
        emotion: Optional[EmotionalState] = None,
    ):
        """
        Manually feed a confidence reading into the generator.
        This is the primary API for manual/testing mode.

        If confidence enters a new band (or stays in the creative band),
        and auto_generate is on, this triggers music generation.
        """
        if self.state != GeneratorState.RUNNING:
            logger.warning("Generator not running, ignoring feed")
            return

        snapshot = ConfidenceSnapshot(
            confidence=confidence,
            model_name=model_name,
            emotional_state=emotion,
        )
        self.confidence_history.append(snapshot)

        band = snapshot.band
        logger.info(f"Confidence: {confidence:.2f} → band: {band.value}")

        if self.auto_generate:
            self._check_and_generate(band, model_name, emotion)

    # ─── Generation Logic ──────────────────────────────────────

    def _check_and_generate(
        self,
        band: ConfidenceBand,
        model_name: str,
        emotion: Optional[EmotionalState],
    ):
        """Check if we should generate new music based on band changes."""
        should_generate = False

        if self.current_band is None:
            # First reading — generate immediately
            should_generate = True
            self._pieces_in_current_band = 0
        elif band != self.current_band:
            # Band changed — generate for the new band
            should_generate = True
            self._pieces_in_current_band = 0
            logger.info(f"Band transition: {self.current_band.value} → {band.value}")
        elif self._pieces_in_current_band < self.max_pieces_per_band:
            # Same band but we still need pieces
            if len(self.queue) < self.queue.maxlen // 2:
                should_generate = True

        if should_generate:
            self.current_band = band
            self._generate_piece(band, model_name, emotion)
            self._pieces_in_current_band += 1

    def _generate_piece(
        self,
        band: ConfidenceBand,
        model_name: str,
        emotion: Optional[EmotionalState],
    ) -> QueuedPiece:
        """Generate a music piece for the current state."""
        # Get a confidence value in the middle of the band
        band_confidence = self._band_center_confidence(band)

        params = confidence_to_music(
            band_confidence,
            emotion,
            seed=self._rng.randint(0, 2**32),
        )

        voice = get_voice(model_name) if model_name != "unknown" else None

        mmx_command = self._build_mmx_command(params, voice)
        piece = QueuedPiece(
            params=params,
            voice=voice,
            mmx_command=mmx_command,
        )
        self.queue.append(piece)
        self.total_generated += 1

        logger.info(
            f"Generated: {piece} | Queue: {len(self.queue)} | "
            f"Total: {self.total_generated}"
        )

        if self.generation_callback:
            try:
                self.generation_callback(piece)
            except Exception as e:
                logger.error(f"Generation callback error: {e}")

        return piece

    def _band_center_confidence(self, band: ConfidenceBand) -> float:
        """Get a representative confidence value for a band."""
        centers = {
            ConfidenceBand.UNCERTAIN: 0.15,
            ConfidenceBand.TRANSITIONAL: 0.35,
            ConfidenceBand.CREATIVE: 0.50,
            ConfidenceBand.EMERGING: 0.77,
            ConfidenceBand.CONFIDENT: 0.93,
        }
        # Add some jitter
        center = centers.get(band, 0.5)
        jitter = self._rng.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, center + jitter))

    def _build_mmx_command(
        self, params: MusicalParameters, voice: Optional[VoiceProfile]
    ) -> str:
        """Build an MMX CLI command for the piece."""
        prompt = params.to_mmx_prompt()
        if voice:
            prompt += f". {voice.mmx_voice_prompt}"
        return f'mmx music --prompt "{prompt}" --duration {params.duration_seconds}'

    # ─── Queue Management ──────────────────────────────────────

    def get_next_piece(self) -> Optional[QueuedPiece]:
        """Pop the next piece from the queue for playback."""
        while self.queue:
            piece = self.queue.popleft()
            piece.played = True
            self.total_played += 1
            return piece
        return None

    def peek_queue(self) -> list[QueuedPiece]:
        """Preview the current queue without modifying it."""
        return list(self.queue)

    def queue_status(self) -> dict:
        """Get queue status for monitoring."""
        return {
            "state": self.state.value,
            "current_band": self.current_band.value if self.current_band else None,
            "queue_length": len(self.queue),
            "total_generated": self.total_generated,
            "total_played": self.total_played,
            "pieces_in_current_band": self._pieces_in_current_band,
        }

    # ─── WebSocket Mode ────────────────────────────────────────

    async def _run_websocket(self):
        """Run the WebSocket-based session monitor."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed; pip install websockets")
            return

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        while self.state == GeneratorState.RUNNING:
            try:
                async with websockets.connect(self.ws_url, extra_headers=headers) as ws:
                    logger.info(f"WebSocket connected: {self.ws_url}")
                    async for message in ws:
                        if self.state != GeneratorState.RUNNING:
                            break
                        data = json.loads(message)
                        confidence = data.get("confidence", 0.5)
                        model = data.get("model", "unknown")
                        self.feed_confidence(confidence, model)
            except Exception as e:
                logger.error(f"WebSocket error: {e}, reconnecting in {self.poll_interval}s")
                await asyncio.sleep(self.poll_interval)

    # ─── Polling Mode ──────────────────────────────────────────

    async def _run_polling(self):
        """Run the HTTP polling-based session monitor."""
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp not installed; pip install aiohttp")
            return

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        while self.state == GeneratorState.RUNNING:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.poll_url, headers=headers) as resp:
                        data = await resp.json()
                        confidence = data.get("confidence", 0.5)
                        model = data.get("model", "unknown")
                        self.feed_confidence(confidence, model)
            except Exception as e:
                logger.error(f"Polling error: {e}")

            await asyncio.sleep(self.poll_interval)

    # ─── Simulation Mode ───────────────────────────────────────

    async def run_simulation(
        self,
        duration_seconds: int = 120,
        models: Optional[list[str]] = None,
        start_confidence: float = 0.1,
        end_confidence: float = 0.95,
    ):
        """
        Run a simulated session that walks through confidence bands.
        Useful for demos and testing.
        """
        if models is None:
            models = ["flash", "pro", "hermes"]

        steps = max(10, duration_seconds // 5)
        for i in range(steps):
            if self.state != GeneratorState.RUNNING:
                break

            # Interpolate confidence with some noise
            t = i / max(1, steps - 1)
            base = start_confidence + (end_confidence - start_confidence) * t
            noise = self._rng.uniform(-0.1, 0.1)
            confidence = max(0.0, min(1.0, base + noise))

            model = self._rng.choice(models)
            self.feed_confidence(confidence, model_name=model)

            await asyncio.sleep(duration_seconds / steps)

    # ─── MMX Execution ─────────────────────────────────────────

    async def execute_mmx(self, piece: QueuedPiece) -> Optional[str]:
        """
        Execute the MMX command to actually generate audio.
        Returns the path to the generated file, or None on failure.
        """
        try:
            proc = await asyncio.create_subprocess_shell(
                piece.mmx_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                # Try to parse output for file path
                output = stdout.decode().strip()
                piece.file_path = output.split("\n")[-1] if output else None
                logger.info(f"MMX generated: {piece.file_path}")
                return piece.file_path
            else:
                logger.error(f"MMX error: {stderr.decode()}")
                return None
        except FileNotFoundError:
            logger.warning("mmx CLI not found; skipping actual generation")
            return None
        except Exception as e:
            logger.error(f"MMX execution failed: {e}")
            return None


# ─── Convenience Functions ───────────────────────────────────────────────────

def create_default_generator(seed: Optional[int] = None) -> LiveGenerator:
    """Create a LiveGenerator with sensible defaults."""
    gen = LiveGenerator(seed=seed)
    return gen


async def demo_run():
    """Run a quick demo of the live generator."""
    gen = create_default_generator(seed=42)

    pieces_generated = []

    def on_gen(piece: QueuedPiece):
        pieces_generated.append(piece)
        print(f"  🎵 Generated: {piece}")

    gen.on_generate(on_gen)
    await gen.start()

    print("Running 60-second simulation...")
    await gen.run_simulation(duration_seconds=60)

    await gen.stop()

    print(f"\nGenerated {len(pieces_generated)} pieces:")
    for p in pieces_generated:
        print(f"  {p}")

    print(f"\nQueue status: {gen.queue_status()}")
    return gen


if __name__ == "__main__":
    asyncio.run(demo_run())
