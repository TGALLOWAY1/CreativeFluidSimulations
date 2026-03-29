# CLAUDE.md

## Project Overview

Audio-reactive fluid simulation that visualizes music as animated smoke/fluid using the **Stable Fluids** algorithm. A single-file Python application (`fluid_sim.py`) running GPU-accelerated physics via Taichi with real-time audio analysis.

## Tech Stack

- **Taichi** — GPU-accelerated physics kernels (auto-detects CUDA, Metal, Vulkan)
- **Pygame** — Audio playback and GUI windowing
- **NumPy** — Audio data processing
- **SciPy** — FFT frequency analysis and WAV file I/O
- **Python 3** — No version pinned; uses modern syntax

## Repository Structure

```
├── fluid_sim.py         # Entire application (single file)
├── README.md            # Minimal readme with screenshot
├── CLAUDE.md            # This file
├── REF IMAGES/          # Visual style references (Ferrofluid, Fire, Nebula)
└── REF SONGS/           # Test audio (TestSong.wav, 44100 Hz stereo)
```

## Architecture of fluid_sim.py

The file is organized into four sections:

1. **Configuration (top)** — Grid resolution (512×512), physics constants (`dt`, Jacobi iterations, decay rates), tunable parameters (`density_multiplier`, `music_responsiveness`, `bass_speed_multiplier`)
2. **Taichi Fields & GPU Kernels** — Velocity, density, pressure fields; kernels for advection, impulse injection, fanned emission, shockwave, edge turbulence, pressure solving, and rendering
3. **AudioAnalyzer class** — Loads WAV files, runs streaming FFT (4096-sample window), extracts bass (40–200 Hz), treble (2–10 kHz), stereo spread, pan, and transient onset detection with adaptive normalization and 2-frame bass smoothing
4. **Main loop** — GUI setup, keyboard input handling, audio energy → physics forces pipeline, render cycle

### Physics Pipeline (per frame)

```
Audio energy → Apply fanned emission → Shockwave (if transient) → Edge turbulence (if treble)
→ Advect velocity → Advect density → Compute divergence → Pressure solve (40 Jacobi iterations)
→ Subtract gradient → Render
```

### Key Patterns

- `@ti.kernel` / `@ti.func` decorators mark GPU code — these cannot use Python stdlib
- `_new_*` prefixed fields are double-buffer temporaries
- Semi-Lagrangian advection with bilinear interpolation for stability
- Gaussian falloff for force/density injection
- Color mapping uses density thresholds (0.3, 0.7) for dark blue → cyan palette
- `apply_shockwave` — expanding ring of outward velocity triggered by kick/snare transients (onset detection)
- `apply_edge_turbulence` — high-frequency pseudo-random velocity perturbations at density gradient boundaries, driven by treble energy

## Running the Project

```bash
# Install dependencies (no requirements.txt exists yet)
pip install taichi pygame numpy scipy

# Run the simulation
python fluid_sim.py
```

Requires a WAV file at `REF SONGS/TestSong.wav` for audio input. The simulation opens a 512×512 GUI window.

### Interactive Controls

| Key | Action |
|-----|--------|
| Q/A | Increase/decrease density multiplier |
| W/S | Increase/decrease music responsiveness |
| E/D | Increase/decrease bass speed multiplier |
| R/F | Increase/decrease decay rate |
| T | Reset all parameters to defaults |
| Space | Restart audio/animation |
| P | Pause/unpause |
| H | Toggle on-screen help overlay |

## Development Conventions

### Code Style

- **snake_case** for functions and variables, **CamelCase** for classes
- Configuration constants at the top of the file, not scattered
- Sectional headers with `=====` separators and numbered comments
- Physics concepts explained in inline comments/docstrings

### Branching

- `Phase-1-Stable-Fluids-Attempt-1` — Main development branch
- Feature branches as needed

### What's Not Set Up

- No `requirements.txt` or `pyproject.toml` — dependencies are implicit
- No tests, linting config, or CI/CD
- No `.gitignore` — be careful not to commit large binaries or virtual environments
- No logging framework — uses `print()` statements

## Guidelines for AI Assistants

- This is a **single-file project** — all code lives in `fluid_sim.py`. Do not split it into modules unless explicitly asked.
- Taichi kernel code (`@ti.kernel`, `@ti.func`) has restrictions: no Python objects, no dynamic allocation, limited control flow. Test changes carefully.
- Audio parameters are tuned for the specific WAV test file. Changes to frequency bands or normalization can drastically alter the visual output.
- The physics simulation is sensitive to parameter values — small changes to `dt`, decay rates, or Jacobi iterations can cause instability or visual artifacts.
- Reference images in `REF IMAGES/` show the target aesthetic (cosmic, liquid metal, fire). Keep visual changes aligned with this direction.
- The `REF SONGS/` directory contains a large WAV file (~21 MB) — do not re-add or duplicate it in commits.
