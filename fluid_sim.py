import taichi as ti
import numpy as np
import time
import os
from scipy.io import wavfile
import pygame

# Initialize Taichi (Auto-detects GPU: CUDA, Metal, or Vulkan)
ti.init(arch=ti.gpu)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
RES_X = 1280               # Simulation width (wide aspect for tower layout)
RES_Y = 720                # Simulation height
dt = 0.02                  # Time step
p_jacobi_iters = 40        # Pressure solver iterations
NUM_TOWERS = 5             # Number of frequency towers

# Tower definitions
TOWER_CONFIG = [
    {   # KICK — Strong upward impulse on transients
        "name": "kick",
        "color": (1.0, 0.5, 0.0),      # Orange
        "x_pos": 0.12,
        "density_scale": 1.0,
        "velocity_scale": 1.2,
        "dissipation": 0.984,
        "radius": 0.022,
        "spread": 20.0,
    },
    {   # BASS — Slow heavy continuous push
        "name": "bass",
        "color": (0.85, 0.1, 0.15),    # Red
        "x_pos": 0.31,
        "density_scale": 1.3,
        "velocity_scale": 0.45,
        "dissipation": 0.976,
        "radius": 0.032,
        "spread": 12.0,
    },
    {   # SNARE — Radial burst on transients
        "name": "snare",
        "color": (0.2, 0.35, 1.0),     # Blue
        "x_pos": 0.50,
        "density_scale": 0.85,
        "velocity_scale": 1.0,
        "dissipation": 0.984,
        "radius": 0.020,
        "spread": 40.0,
    },
    {   # HATS — Chaotic high-frequency turbulence
        "name": "hats",
        "color": (0.0, 0.9, 0.95),     # Cyan
        "x_pos": 0.69,
        "density_scale": 0.45,
        "velocity_scale": 0.9,
        "dissipation": 0.991,
        "radius": 0.016,
        "spread": 30.0,
    },
    {   # INSTRUMENTS — Smooth flowing motion with centering
        "name": "instruments",
        "color": (0.6, 0.15, 0.9),     # Purple
        "x_pos": 0.88,
        "density_scale": 0.7,
        "velocity_scale": 0.65,
        "dissipation": 0.984,
        "radius": 0.024,
        "spread": 18.0,
    },
]

# Audio stems directory
STEMS_DIR = "STEMS"

# Global tunable parameters (defaults)
DEFAULT_DENSITY_MULT = 0.5
DEFAULT_MUSIC_RESP = 1.0

# =============================================================================
# 2. DATA FIELDS
# =============================================================================
# Shared velocity field (all towers interact through this)
velocity = ti.Vector.field(2, dtype=float, shape=(RES_X, RES_Y))
_new_velocity = ti.Vector.field(2, dtype=float, shape=(RES_X, RES_Y))

# Per-tower density: 5 components per cell, one per tower
tower_density = ti.Vector.field(NUM_TOWERS, dtype=float, shape=(RES_X, RES_Y))
_new_tower_density = ti.Vector.field(NUM_TOWERS, dtype=float, shape=(RES_X, RES_Y))

# Pressure solver fields
divergence = ti.field(dtype=float, shape=(RES_X, RES_Y))
pressure = ti.field(dtype=float, shape=(RES_X, RES_Y))
_new_pressure = ti.field(dtype=float, shape=(RES_X, RES_Y))

# Visualization buffer (RGB pixels)
pixels = ti.Vector.field(3, dtype=float, shape=(RES_X, RES_Y))

# Tower parameters accessible from GPU kernels
tower_colors_f = ti.Vector.field(3, dtype=float, shape=NUM_TOWERS)
tower_x_f = ti.field(dtype=float, shape=NUM_TOWERS)
tower_density_scale_f = ti.field(dtype=float, shape=NUM_TOWERS)
tower_velocity_scale_f = ti.field(dtype=float, shape=NUM_TOWERS)
tower_dissipation_f = ti.field(dtype=float, shape=NUM_TOWERS)
tower_radius_f = ti.field(dtype=float, shape=NUM_TOWERS)
tower_spread_f = ti.field(dtype=float, shape=NUM_TOWERS)

# Per-frame audio data (written from Python each frame)
tower_energy_f = ti.field(dtype=float, shape=NUM_TOWERS)
tower_transient_f = ti.field(dtype=float, shape=NUM_TOWERS)


def init_tower_fields():
    """Copy tower configuration into Taichi fields for GPU access."""
    for t in range(NUM_TOWERS):
        cfg = TOWER_CONFIG[t]
        tower_colors_f[t] = ti.Vector(list(cfg["color"]))
        tower_x_f[t] = cfg["x_pos"]
        tower_density_scale_f[t] = cfg["density_scale"]
        tower_velocity_scale_f[t] = cfg["velocity_scale"]
        tower_dissipation_f[t] = cfg["dissipation"]
        tower_radius_f[t] = cfg["radius"]
        tower_spread_f[t] = cfg["spread"]


# =============================================================================
# 3. PHYSICS KERNELS (GPU)
# =============================================================================

@ti.func
def sample_field(field: ti.template(), x: float, y: float):
    """Bilinear interpolation for smooth field sampling (scalar or vector)."""
    x = ti.max(0.0, ti.min(x, float(RES_X) - 1.0))
    y = ti.max(0.0, ti.min(y, float(RES_Y) - 1.0))
    i0 = int(x)
    j0 = int(y)
    i1 = ti.min(i0 + 1, RES_X - 1)
    j1 = ti.min(j0 + 1, RES_Y - 1)
    fx = x - float(i0)
    fy = y - float(j0)
    v00 = field[i0, j0]
    v10 = field[i1, j0]
    v01 = field[i0, j1]
    v11 = field[i1, j1]
    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return v0 * (1.0 - fy) + v1 * fy


@ti.kernel
def advect_velocity(vf: ti.template(), new_vf: ti.template(), decay: float):
    """Self-advect velocity field with uniform decay."""
    for i, j in vf:
        coord = ti.Vector([float(i), float(j)]) + 0.5
        vel = vf[i, j]
        prev = coord - vel * dt
        prev[0] = ti.max(0.5, ti.min(prev[0], float(RES_X) - 0.5))
        prev[1] = ti.max(0.5, ti.min(prev[1], float(RES_Y) - 0.5))
        new_vf[i, j] = sample_field(vf, prev[0] - 0.5, prev[1] - 0.5) * decay


@ti.kernel
def advect_tower_density_k(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    """Advect tower density along velocity field with per-tower dissipation."""
    for i, j in vf:
        coord = ti.Vector([float(i), float(j)]) + 0.5
        vel = vf[i, j]
        prev = coord - vel * dt
        prev[0] = ti.max(0.5, ti.min(prev[0], float(RES_X) - 0.5))
        prev[1] = ti.max(0.5, ti.min(prev[1], float(RES_Y) - 0.5))
        sampled = sample_field(qf, prev[0] - 0.5, prev[1] - 0.5)
        # Apply per-tower dissipation rate
        result = sampled
        for t in ti.static(range(NUM_TOWERS)):
            result[t] = sampled[t] * tower_dissipation_f[t]
        new_qf[i, j] = result


@ti.kernel
def apply_tower_emissions(frame: int, density_mult: float, music_resp: float):
    """Inject velocity and density for all 5 towers based on audio energy.
    Each tower has a unique force type determined by its index."""
    for i, j in velocity:
        for t in ti.static(range(NUM_TOWERS)):
            energy = tower_energy_f[t] * music_resp
            transient = tower_transient_f[t] * music_resp

            if energy >= 0.02:
                tx = tower_x_f[t] * float(RES_X)
                ty = 0.03 * float(RES_Y)
                dx = float(i) - tx
                dy = float(j) - ty
                dist2 = dx * dx + dy * dy

                r = tower_radius_f[t] * float(RES_X)
                r2_limit = r * r * 4.0

                if dist2 < r2_limit:
                    r_sq = ti.max(r * r, 0.001)
                    falloff = ti.exp(-dist2 / r_sq)

                    d_scale = tower_density_scale_f[t]
                    v_scale = tower_velocity_scale_f[t]
                    spread = tower_spread_f[t]

                    vx = 0.0
                    vy = 0.0
                    d_val = 0.0

                    norm_x = 0.0
                    if r > 0.0:
                        norm_x = ti.max(-1.0, ti.min(dx / r, 1.0))

                    # ---- KICK: Strong upward impulse on transients ----
                    if t == 0:
                        impulse = energy * v_scale * 14.0
                        if transient > 0.1:
                            impulse *= (1.0 + transient * 4.0)
                        angle_rad = norm_x * spread * 3.14159 / 180.0
                        vx = ti.sin(angle_rad) * impulse * 0.3
                        vy = ti.cos(angle_rad) * impulse
                        d_val = energy * d_scale * 0.15 * density_mult
                        if transient > 0.1:
                            d_val *= (1.0 + transient * 2.5)

                    # ---- BASS: Slow heavy continuous upward push ----
                    elif t == 1:
                        push = energy * v_scale * 5.0
                        angle_rad = norm_x * spread * 3.14159 / 180.0
                        vx = ti.sin(angle_rad) * push * 0.15
                        vy = ti.cos(angle_rad) * push
                        d_val = energy * d_scale * 0.12 * density_mult

                    # ---- SNARE: Radial burst on transients ----
                    elif t == 2:
                        if transient > 0.05:
                            dist = ti.sqrt(dist2) + 0.001
                            burst = transient * v_scale * 18.0
                            vx = (dx / dist) * burst
                            vy = (dy / dist) * burst + energy * v_scale * 5.0
                            d_val = transient * d_scale * 0.2 * density_mult
                        else:
                            vy = energy * v_scale * 3.0
                            d_val = energy * d_scale * 0.03 * density_mult

                    # ---- HATS: Chaotic noise turbulence ----
                    elif t == 3:
                        h1 = float((i * 73856093 + frame * 19349663) ^ (j * 83492791))
                        h1 = h1 * 0.0000001
                        h1 = h1 - float(int(h1))
                        h2 = float((j * 73856093 + frame * 83492791) ^ (i * 19349663))
                        h2 = h2 * 0.0000001
                        h2 = h2 - float(int(h2))
                        noise_str = energy * v_scale * 10.0
                        vx = (h1 - 0.5) * noise_str
                        vy = (h2 - 0.3) * noise_str + energy * v_scale * 5.0
                        d_val = energy * d_scale * 0.06 * density_mult

                    # ---- INSTRUMENTS: Smooth flow with gentle centering ----
                    elif t == 4:
                        smooth_v = energy * v_scale * 8.0
                        center_pull = -dx * 0.015 * energy
                        vx = center_pull
                        vy = smooth_v
                        d_val = energy * d_scale * 0.09 * density_mult

                    velocity[i, j] += ti.Vector([vx, vy]) * falloff
                    tower_density[i, j][t] += d_val * falloff


@ti.kernel
def compute_divergence(vf: ti.template(), div: ti.template()):
    """Calculate divergence of velocity field."""
    for i, j in vf:
        wl = vf[ti.max(0, i - 1), j][0]
        wr = vf[ti.min(RES_X - 1, i + 1), j][0]
        wb = vf[i, ti.max(0, j - 1)][1]
        wt = vf[i, ti.min(RES_Y - 1, j + 1)][1]
        div[i, j] = 0.5 * (wr - wl + wt - wb)


@ti.kernel
def pressure_jacobi(p: ti.template(), new_p: ti.template(), div: ti.template()):
    """Jacobi iteration for pressure Poisson equation."""
    for i, j in p:
        pl = p[ti.max(0, i - 1), j]
        pr = p[ti.min(RES_X - 1, i + 1), j]
        pb = p[i, ti.max(0, j - 1)]
        pt = p[i, ti.min(RES_Y - 1, j + 1)]
        new_p[i, j] = (pl + pr + pb + pt - div[i, j]) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), p: ti.template()):
    """Subtract pressure gradient to enforce incompressibility."""
    for i, j in vf:
        pl = p[ti.max(0, i - 1), j]
        pr = p[ti.min(RES_X - 1, i + 1), j]
        pb = p[i, ti.max(0, j - 1)]
        pt = p[i, ti.min(RES_Y - 1, j + 1)]
        vel = vf[i, j]
        mask = 1.0
        if i == 0 or i == RES_X - 1:
            mask = 0.0
        if j == 0 or j == RES_Y - 1:
            mask = 0.0
        vf[i, j] = vel - 0.5 * ti.Vector([pr - pl, pt - pb]) * mask


@ti.kernel
def render_display(shake_x: float, shake_y: float):
    """Map per-tower density to colored pixels with additive blending.
    Camera shake is applied by offsetting sample coordinates."""
    for i, j in pixels:
        si = int(float(i) + shake_x)
        sj = int(float(j) + shake_y)
        si = ti.max(0, ti.min(si, RES_X - 1))
        sj = ti.max(0, ti.min(sj, RES_Y - 1))

        td = tower_density[si, sj]

        r_out = 0.0
        g_out = 0.0
        b_out = 0.0

        for t in ti.static(range(NUM_TOWERS)):
            d = td[t] * 3.0
            if d > 0.005:
                tc_r = tower_colors_f[t][0]
                tc_g = tower_colors_f[t][1]
                tc_b = tower_colors_f[t][2]

                # Dark edge version of the tower color
                dark_r = tc_r * 0.15
                dark_g = tc_g * 0.15
                dark_b = tc_b * 0.15

                # Bright core version (color + white boost, clamped)
                bright_r = ti.min(tc_r + 0.25, 1.0)
                bright_g = ti.min(tc_g + 0.25, 1.0)
                bright_b = ti.min(tc_b + 0.25, 1.0)

                cr = 0.0
                cg = 0.0
                cb = 0.0

                if d < 0.3:
                    # Dark edge: black -> dark color
                    f = d / 0.3
                    cr = dark_r * f
                    cg = dark_g * f
                    cb = dark_b * f
                elif d < 0.7:
                    # Mid: dark color -> full color
                    f = (d - 0.3) / 0.4
                    cr = dark_r + (tc_r - dark_r) * f
                    cg = dark_g + (tc_g - dark_g) * f
                    cb = dark_b + (tc_b - dark_b) * f
                else:
                    # Core: full color -> bright
                    f = ti.min((d - 0.7) / 0.3, 1.0)
                    cr = tc_r + (bright_r - tc_r) * f
                    cg = tc_g + (bright_g - tc_g) * f
                    cb = tc_b + (bright_b - tc_b) * f

                # Additive blending across towers
                r_out += cr
                g_out += cg
                b_out += cb

        pixels[i, j] = ti.Vector([
            ti.min(r_out, 1.0),
            ti.min(g_out, 1.0),
            ti.min(b_out, 1.0)
        ])


# =============================================================================
# 4. AUDIO SYSTEM
# =============================================================================

class StemAnalyzer:
    """Analyzes a single audio stem for energy and transient detection."""

    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath
        self.has_audio = False
        self.data = None
        self.fs = 44100
        self.audio_length = 0
        self.sound = None
        self.channel = None

        self.energy_history = []
        self.prev_energy = 0.0
        self.max_energy = 0.1

        if filepath and os.path.exists(filepath):
            try:
                self.fs, data = wavfile.read(filepath)
                # Convert to mono float
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                # Normalize based on dtype
                if data.dtype == np.int16:
                    data = data.astype(np.float64) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float64) / 2147483648.0
                elif data.dtype not in (np.float32, np.float64):
                    data = data.astype(np.float64)
                    max_val = np.max(np.abs(data))
                    if max_val > 0:
                        data /= max_val
                self.data = data.astype(np.float64)
                self.audio_length = len(data) / self.fs
                self.has_audio = True

                self.sound = pygame.mixer.Sound(filepath)
                self.sound.set_volume(1.0)

                print(f"  [{name}] Loaded: {filepath} ({self.audio_length:.1f}s)")
            except Exception as e:
                print(f"  [{name}] Failed to load {filepath}: {e}")
        else:
            if filepath:
                print(f"  [{name}] Not found: {filepath}")

    def play(self, channel_id):
        """Start playback on a specific mixer channel."""
        if self.has_audio and self.sound:
            self.channel = pygame.mixer.Channel(channel_id)
            self.channel.play(self.sound)

    def get_energy(self, current_time):
        """Returns (energy, transient) normalized 0-1."""
        if not self.has_audio:
            # Small ambient emission so towers are visible without audio
            return 0.06, 0.0

        if current_time >= self.audio_length:
            return 0.0, 0.0

        idx = int(current_time * self.fs)
        window = 2048
        if idx + window >= len(self.data):
            return 0.0, 0.0

        chunk = self.data[idx:idx + window]
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # Adaptive normalization
        if rms > self.max_energy:
            self.max_energy = rms
        energy = min(rms / max(self.max_energy, 0.001), 1.0)

        # Transient detection: positive energy derivative
        transient = max(0.0, energy - self.prev_energy)
        self.prev_energy = energy

        # 3-frame smoothing
        self.energy_history.append(energy)
        if len(self.energy_history) > 3:
            self.energy_history.pop(0)
        smoothed = float(np.mean(self.energy_history))

        return smoothed, float(transient)

    def stop(self):
        if self.channel:
            self.channel.stop()

    def restart(self, channel_id):
        self.energy_history = []
        self.prev_energy = 0.0
        self.max_energy = 0.1
        if self.has_audio and self.sound:
            self.channel = pygame.mixer.Channel(channel_id)
            self.channel.play(self.sound)


class MultiStemAnalyzer:
    """Manages synchronized playback and analysis of multiple audio stems."""

    def __init__(self, stems_dir):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(8)

        print("Loading audio stems...")
        self.stems = []
        self.start_time = time.time()

        for cfg in TOWER_CONFIG:
            name = cfg["name"]
            filepath = os.path.join(stems_dir, f"{name}.wav")
            stem = StemAnalyzer(name, filepath)
            self.stems.append(stem)

        loaded = sum(1 for s in self.stems if s.has_audio)
        if loaded == 0:
            print(f"\nNo stem files found in '{stems_dir}/'")
            print(f"Expected files: {', '.join(c['name'] + '.wav' for c in TOWER_CONFIG)}")
            print("Place WAV stems in the STEMS/ directory and restart.")
            print("Towers will run with ambient emission.\n")
        else:
            print(f"Loaded {loaded}/{NUM_TOWERS} stems.\n")

    def start_playback(self):
        """Start all stems playing simultaneously."""
        self.start_time = time.time()
        for i, stem in enumerate(self.stems):
            stem.play(i)

    def get_all_energies(self):
        """Returns list of (energy, transient) tuples for each tower."""
        current_time = time.time() - self.start_time
        return [stem.get_energy(current_time) for stem in self.stems]

    def stop(self):
        for stem in self.stems:
            stem.stop()

    def restart(self):
        self.start_time = time.time()
        for i, stem in enumerate(self.stems):
            stem.restart(i)

    def pause(self):
        pygame.mixer.pause()

    def unpause(self):
        pygame.mixer.unpause()


# =============================================================================
# 5. CAMERA SYSTEM
# =============================================================================

class Camera:
    """Camera with kick-triggered shake effect."""

    def __init__(self):
        self.shake_x = 0.0
        self.shake_y = 0.0
        self.shake_decay = 0.82

    def update(self, kick_transient):
        """Update camera shake based on kick drum transients."""
        if kick_transient > 0.15:
            intensity = kick_transient * 8.0
            self.shake_x = (np.random.random() - 0.5) * intensity
            self.shake_y = (np.random.random() - 0.5) * intensity * 0.5

        # Decay shake
        self.shake_x *= self.shake_decay
        self.shake_y *= self.shake_decay

        # Kill tiny residual
        if abs(self.shake_x) < 0.1:
            self.shake_x = 0.0
        if abs(self.shake_y) < 0.1:
            self.shake_y = 0.0

    def get_shake(self):
        return self.shake_x, self.shake_y


# =============================================================================
# 6. MAIN LOOP
# =============================================================================

def main():
    # Initialize tower parameter fields on GPU
    init_tower_fields()

    # Initialize audio
    audio = MultiStemAnalyzer(STEMS_DIR)

    # Initialize pygame display
    pygame.init()
    screen = pygame.display.set_mode((RES_X, RES_Y))
    pygame.display.set_caption("Frequency Towers")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    font_large = pygame.font.SysFont("monospace", 24)

    # Camera
    camera = Camera()

    # State
    density_mult = DEFAULT_DENSITY_MULT
    music_resp = DEFAULT_MUSIC_RESP
    vel_decay = 0.985
    show_controls = True
    paused = False
    frame_count = 0

    # Start audio playback
    audio.start_playback()

    print("Frequency Towers - Running")
    print("Controls:")
    print("  Q/A: Decrease/Increase Density")
    print("  W/S: Decrease/Increase Music Responsiveness")
    print("  R/F: Decrease/Increase Velocity Decay")
    print("  T: Reset parameters")
    print("  Space: Restart | P: Pause | H: Toggle HUD | ESC: Quit")

    running = True
    try:
        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_q:
                        density_mult = max(0.05, density_mult - 0.02)
                        print(f"Density: {density_mult:.3f}")
                    elif event.key == pygame.K_a:
                        density_mult = min(2.0, density_mult + 0.02)
                        print(f"Density: {density_mult:.3f}")
                    elif event.key == pygame.K_w:
                        music_resp = max(0.1, music_resp - 0.1)
                        print(f"Music Resp: {music_resp:.2f}")
                    elif event.key == pygame.K_s:
                        music_resp = min(3.0, music_resp + 0.1)
                        print(f"Music Resp: {music_resp:.2f}")
                    elif event.key == pygame.K_r:
                        vel_decay = max(0.95, vel_decay - 0.002)
                        print(f"Vel Decay: {vel_decay:.4f}")
                    elif event.key == pygame.K_f:
                        vel_decay = min(0.999, vel_decay + 0.002)
                        print(f"Vel Decay: {vel_decay:.4f}")
                    elif event.key == pygame.K_t:
                        density_mult = DEFAULT_DENSITY_MULT
                        music_resp = DEFAULT_MUSIC_RESP
                        vel_decay = 0.985
                        print("Reset to defaults")
                    elif event.key == pygame.K_h:
                        show_controls = not show_controls
                    elif event.key == pygame.K_p:
                        paused = not paused
                        if paused:
                            audio.pause()
                            print("Paused")
                        else:
                            audio.unpause()
                            print("Unpaused")
                    elif event.key == pygame.K_SPACE:
                        audio.restart()
                        velocity.fill([0.0, 0.0])
                        tower_density.fill(0)
                        pressure.fill(0.0)
                        divergence.fill(0.0)
                        frame_count = 0
                        paused = False
                        print("Restarted!")

            # --- Paused State ---
            if paused:
                render_display(0.0, 0.0)
                img = pixels.to_numpy()
                img = np.flip(img, axis=1)
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                surface = pygame.surfarray.make_surface(img)
                screen.blit(surface, (0, 0))
                pause_text = font_large.render("PAUSED - Press P to resume", True, (255, 50, 50))
                screen.blit(pause_text, (RES_X // 2 - pause_text.get_width() // 2, 30))
                pygame.display.flip()
                clock.tick(30)
                continue

            # --- 1. Audio Analysis ---
            energies = audio.get_all_energies()
            for t in range(NUM_TOWERS):
                energy, transient = energies[t]
                tower_energy_f[t] = energy
                tower_transient_f[t] = transient

            # Update camera with kick transient
            kick_transient = energies[0][1]
            camera.update(kick_transient)

            # Debug output every ~1 second
            if frame_count % 60 == 0:
                parts = [f"{TOWER_CONFIG[t]['name']}:{energies[t][0]:.2f}" for t in range(NUM_TOWERS)]
                print(f"[{frame_count // 60}s] {' | '.join(parts)}")
            frame_count += 1

            # --- 2. Emission ---
            apply_tower_emissions(frame_count, density_mult, music_resp)

            # --- 3. Advect ---
            advect_velocity(velocity, _new_velocity, vel_decay)
            velocity.copy_from(_new_velocity)

            advect_tower_density_k(velocity, tower_density, _new_tower_density)
            tower_density.copy_from(_new_tower_density)

            # --- 4. Pressure Solve ---
            compute_divergence(velocity, divergence)
            pressure.fill(0)
            for _ in range(p_jacobi_iters):
                pressure_jacobi(pressure, _new_pressure, divergence)
                pressure.copy_from(_new_pressure)
            subtract_gradient(velocity, pressure)

            # --- 5. Render ---
            shake_x, shake_y = camera.get_shake()
            render_display(shake_x, shake_y)

            # Transfer to pygame display
            img = pixels.to_numpy()
            img = np.flip(img, axis=1)  # Y-flip: Taichi j=0 is bottom, pygame y=0 is top
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            surface = pygame.surfarray.make_surface(img)
            screen.blit(surface, (0, 0))

            # --- 6. UI Overlay ---
            if show_controls:
                y_pos = 10
                lines = [
                    f"Density: {density_mult:.3f} (Q/A)",
                    f"Music Resp: {music_resp:.2f} (W/S)",
                    f"Vel Decay: {vel_decay:.4f} (R/F)",
                    "T=Reset | Space=Restart | P=Pause | H=Toggle",
                ]
                for line in lines:
                    text_surf = font.render(line, True, (255, 255, 255))
                    screen.blit(text_surf, (10, y_pos))
                    y_pos += 20

                # Per-tower energy display with color bars
                y_pos += 5
                for t in range(NUM_TOWERS):
                    cfg = TOWER_CONFIG[t]
                    e, tr = energies[t]
                    color = tuple(int(c * 255) for c in cfg["color"])
                    name = cfg["name"].upper()
                    text_surf = font.render(f"{name}: {e:.2f} T:{tr:.2f}", True, color)
                    screen.blit(text_surf, (10, y_pos))
                    # Energy bar
                    bar_x = 230
                    bar_w = int(e * 150)
                    pygame.draw.rect(screen, color, (bar_x, y_pos + 2, bar_w, 14))
                    y_pos += 22

                # Tower name labels at bottom of screen
                for t in range(NUM_TOWERS):
                    cfg = TOWER_CONFIG[t]
                    color = tuple(int(c * 255) for c in cfg["color"])
                    label_x = int(cfg["x_pos"] * RES_X)
                    label = font.render(cfg["name"].upper(), True, color)
                    screen.blit(label, (label_x - label.get_width() // 2, RES_Y - 30))

            pygame.display.flip()
            clock.tick(60)

    finally:
        audio.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
