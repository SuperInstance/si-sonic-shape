# 🎶 Sonic Shape

*Confidence-to-music mapping engine*

![🎶 Sonic Shape](docs/images/sonic-shape.jpg)

## What It Is

Every system has confidence signals. Sonic Shape turns them into music.

Low confidence becomes blue notes. High confidence becomes resolution. The creative band (0.4-0.6) becomes jazz. Your system's emotional state, sonified in real-time.

## Install

```bash
pip install superinstance-sonic-shape
```

## Features

- Harmonic dictionary: confidence → key, tempo, mode, instruments
- 17 voice profiles for different agent personalities
- Real-time generation via MMX or any TTS/music API
- Session-to-score: turn a conversation into a musical arc
- Live monitoring: confidence changes trigger musical shifts

## Quick Start

```python
from superinstance import sonic_shape

# See docs/api/sonic-shape-api.md for full documentation
```

## Use It For

**Medical AI that sonifies diagnostic uncertainty for clinicians**

Or anything else. This module is independently useful and Apache-2.0 licensed. Grow it for your industry. Send improvements back.

---

*Part of [LucidDreamer.AI](https://github.com/SuperInstance/luciddreamer-prototype) — built by [SuperInstance](https://github.com/SuperInstance).*
