# CLAUDE.md

## Project Overview

**Frequency Towers** — A music-reactive fluid visualization where 5 separate smoke towers each react to a different instrument stem (kick, bass, snare, hats, instruments). Built on the Stable Fluids algorithm with GPU-accelerated physics via Taichi. Each tower has unique colors, physics behaviors, and force types. A single-file Python application (`fluid_sim.py`).

## Tech Stack

- **Taichi** — GPU-accelerated physics kernels (auto-detects CUDA, Metal, Vulkan)
- **Pygame** — Audio playback, display windowing, and UI rendering
- **NumPy** — Audio data processing and image conversion
- **SciPy** — WAV file I/O
- **Python 3** — No version pinned; uses modern syntax

## Repository Structure

```
├── fluid_sim.py         # Entire application (single file)
├── STEMS/               # Audio stems directory (one WAV per tower)
│   ├── kick.wav         # Kick drum stem
│   ├── bass.wav         # Bass stem
│   ├── snare.wav        # Snare drum stem
│   ├── hats.wav         # Hi-hats stem
│   └── instruments.wav  # Instruments stem
├── README.md            # Readme with screenshot
├── CLAUDE.md            # This file
├── REF IMAGES/          # Visual style references (Ferrofluid, Fire, Nebula)
└── REF SONGS/           # Original test audio
```

## Architecture of fluid_sim.py

The file is organized into six sections:

1. **Configuration (top)** — Grid resolution (1280×720), tower definitions (name, color, x_pos, physics params), global tunable parameters
2. **Data Fields** — Taichi fields: shared velocity (2D vector), per-tower density (5-component vector), pressure/divergence, pixel buffer, tower parameter fields for GPU access
3. **Physics Kernels** — GPU kernels for advection, per-tower emission (5 force types), pressure solving, and per-tower color rendering
4. **Audio System** — `StemAnalyzer` (single stem RMS + transient detection), `MultiStemAnalyzer` (synchronized multi-channel playback)
5. **Camera System** — `Camera` class with kick-triggered shake effect
6. **Main Loop** — Pygame display, event handling, audio → physics → render pipeline

### Tower Configuration

| Tower | Color | Force Type | Behavior |
|-------|-------|------------|----------|
| Kick | Orange | Impulse | Strong upward burst on transients |
| Bass | Red | Slow push | Heavy continuous upward flow |
| Snare | Blue | Radial | Outward burst on transients |
| Hats | Cyan | Noise | Chaotic random turbulence |
| Instruments | Purple | Attractor | Smooth flow with centering |

### Physics Pipeline (per frame)

```
Audio stems → RMS + transient per tower → Apply tower emissions (5 force types)
→ Advect velocity → Advect tower density (per-tower dissipation)
→ Compute divergence → Pressure solve (40 Jacobi iterations)
→ Subtract gradient → Render (per-tower color, additive blending) → Pygame display
```

### Key Patterns

- `@ti.kernel` / `@ti.func` decorators mark GPU code — these cannot use Python stdlib
- `_new_*` prefixed fields are double-buffer temporaries
- `tower_density` is a 5-component vector field — one component per tower, advected together
- `ti.static(range(NUM_TOWERS))` unrolls tower loops at compile time for per-tower force type branching
- Tower parameters stored in Taichi fields for GPU kernel access
- Additive color blending in render kernel allows natural tower overlap
- Camera shake applied via sample coordinate offset in render kernel

## Running the Project

```bash
# Install dependencies
pip install taichi pygame numpy scipy

# Place audio stems in STEMS/ directory
# Expected: kick.wav, bass.wav, snare.wav, hats.wav, instruments.wav

# Run the simulation
python fluid_sim.py
```

The simulation opens a 1280×720 pygame window. Towers emit ambient smoke even without stem files.

### Interactive Controls

| Key | Action |
|-----|--------|
| Q/A | Decrease/increase density multiplier |
| W/S | Decrease/increase music responsiveness |
| R/F | Decrease/increase velocity decay |
| T | Reset all parameters to defaults |
| Space | Restart audio/animation |
| P | Pause/unpause |
| H | Toggle on-screen HUD overlay |
| ESC | Quit |

## Development Conventions

### Code Style

- **snake_case** for functions and variables, **CamelCase** for classes
- Configuration constants and tower definitions at the top of the file
- Sectional headers with `=====` separators and numbered comments
- Physics concepts explained in inline comments/docstrings

### Branching

- `main` — Main branch
- Feature branches as needed

### What's Not Set Up

- No `requirements.txt` or `pyproject.toml` — dependencies are implicit
- No tests, linting config, or CI/CD
- No `.gitignore` — be careful not to commit large binaries or virtual environments
- No logging framework — uses `print()` statements

## Guidelines for AI Assistants

- This is a **single-file project** — all code lives in `fluid_sim.py`. Do not split it into modules unless explicitly asked.
- Taichi kernel code (`@ti.kernel`, `@ti.func`) has restrictions: no Python objects, no dynamic allocation, no `continue`/`break` in `ti.static` loops. Use nested `if` blocks instead. Test changes carefully.
- The 5-component `tower_density` vector field is advected as a single unit. Per-tower dissipation is applied component-wise in `advect_tower_density_k`.
- Tower parameters are stored in both Python dicts (`TOWER_CONFIG`) and Taichi fields (suffixed `_f`). Always update both if changing tower config.
- The physics simulation is sensitive to parameter values — small changes to `dt`, decay rates, or Jacobi iterations can cause instability or visual artifacts.
- Reference images in `REF IMAGES/` show the target aesthetic. Keep visual changes aligned with this direction.
- The `STEMS/` and `REF SONGS/` directories contain large WAV files — do not commit them.
