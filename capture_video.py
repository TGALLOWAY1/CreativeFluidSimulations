"""
Headless video capture of the fluid simulation.
Runs the simulation for a set duration and writes frames to an MP4 via ffmpeg.
"""
import subprocess
import sys
import os
import numpy as np
import time

# Set SDL to use dummy video driver (no display needed for audio)
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import taichi as ti
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq

# Initialize Taichi with CPU (more reliable in headless environments)
ti.init(arch=ti.cpu)

# =============================================================================
# CONFIGURATION (same as fluid_sim.py)
# =============================================================================
RES = 512
dt = 0.02
p_jacobi_iters = 40
force_radius = 0.1
damp = 0.985

density_multiplier = 0.15
music_responsiveness = 1.0
bass_speed_multiplier = 90.0
base_flow_speed = 6.0

AUDIO_FILE = "REF SONGS/TestSong.wav"
CAPTURE_DURATION = 55  # seconds (full audio length)
FPS = 30
OUTPUT_FILE = "simulation_output.mp4"

# =============================================================================
# DATA FIELDS
# =============================================================================
velocity = ti.Vector.field(2, dtype=float, shape=(RES, RES))
_new_velocity = ti.Vector.field(2, dtype=float, shape=(RES, RES))
density_field = ti.field(dtype=float, shape=(RES, RES))
_new_density = ti.field(dtype=float, shape=(RES, RES))
divergence = ti.field(dtype=float, shape=(RES, RES))
pressure = ti.field(dtype=float, shape=(RES, RES))
_new_pressure = ti.field(dtype=float, shape=(RES, RES))
pixels = ti.Vector.field(3, dtype=float, shape=(RES, RES))

# =============================================================================
# PHYSICS KERNELS (identical to fluid_sim.py)
# =============================================================================

@ti.func
def sample_field(field: ti.template(), x: float, y: float):
    x = ti.max(0.0, ti.min(x, RES - 1.0))
    y = ti.max(0.0, ti.min(y, RES - 1.0))
    i0 = int(x)
    j0 = int(y)
    i1 = ti.min(i0 + 1, RES - 1)
    j1 = ti.min(j0 + 1, RES - 1)
    fx = x - i0
    fy = y - j0
    v00 = field[i0, j0]
    v10 = field[i1, j0]
    v01 = field[i0, j1]
    v11 = field[i1, j1]
    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return v0 * (1.0 - fy) + v1 * fy

@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template(), decay: float):
    for i, j in vf:
        coord = ti.Vector([i, j]) + 0.5
        vel = vf[i, j]
        prev_coord = coord - vel * dt
        prev_coord = ti.max(0.5, ti.min(prev_coord, RES - 0.5))
        new_qf[i, j] = sample_field(qf, prev_coord[0] - 0.5, prev_coord[1] - 0.5) * decay

@ti.kernel
def apply_fanned_emission(vf: ti.template(), df: ti.template(),
                          x: float, y: float,
                          r: float, base_speed: float, bass_boost: float,
                          pan: float, spread_width: float, d_val: float):
    for i, j in vf:
        dx = i - x * RES
        dy = j - y * RES
        dist2 = dx*dx + dy*dy
        if dist2 < (r * RES)**2:
            falloff = ti.exp(-dist2 / (r * RES))
            max_dist = r * RES
            if max_dist > 0:
                normalized_x = dx / max_dist
                normalized_x = ti.max(-1.0, ti.min(1.0, normalized_x))
                pan_offset = pan * (spread_width * 0.4)
                cell_angle = pan_offset + (normalized_x * spread_width * 0.5)
                hash_val = float((i * 73856093) ^ (j * 19349663)) * 0.000001
                hash_val = hash_val - int(hash_val)
                random_variation = (hash_val - 0.5) * spread_width * 0.2
                total_angle = cell_angle + random_variation
                angle_rad = total_angle * 3.14159 / 180.0
                speed = base_speed + bass_boost
                v_x = ti.sin(angle_rad) * speed * 1.2
                v_y = ti.cos(angle_rad) * speed
                vf[i, j] += ti.Vector([v_x, v_y]) * falloff
                df[i, j] += d_val * falloff

@ti.kernel
def compute_divergence(vf: ti.template(), div: ti.template()):
    for i, j in vf:
        wl = vf[max(0, i-1), j][0]
        wr = vf[min(RES-1, i+1), j][0]
        wb = vf[i, max(0, j-1)][1]
        wt = vf[i, min(RES-1, j+1)][1]
        div[i, j] = 0.5 * (wr - wl + wt - wb)

@ti.kernel
def pressure_jacobi(p: ti.template(), new_p: ti.template(), div: ti.template()):
    for i, j in p:
        pl = p[max(0, i-1), j]
        pr = p[min(RES-1, i+1), j]
        pb = p[i, max(0, j-1)]
        pt = p[i, min(RES-1, j+1)]
        new_p[i, j] = (pl + pr + pb + pt - div[i, j]) * 0.25

@ti.kernel
def subtract_gradient(vf: ti.template(), p: ti.template()):
    for i, j in vf:
        pl = p[max(0, i-1), j]
        pr = p[min(RES-1, i+1), j]
        pb = p[i, max(0, j-1)]
        pt = p[i, min(RES-1, j+1)]
        vel = vf[i, j]
        mask = 1.0
        if i == 0 or i == RES-1: mask = 0.0
        if j == 0 or j == RES-1: mask = 0.0
        vf[i, j] = vel - 0.5 * ti.Vector([pr - pl, pt - pb]) * mask

@ti.kernel
def render_display(df: ti.template(), density_scale: float):
    for i, j in pixels:
        d = df[i, j] * density_scale
        base_color = ti.Vector([0.0, 0.02, 0.05])
        light_color = ti.Vector([0.0, 0.7, 0.85])
        hot_color = ti.Vector([0.2, 0.9, 1.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        if d < 0.3:
            c = base_color + (light_color - base_color) * (d / 0.3)
        elif d < 0.7:
            c = light_color + (hot_color - light_color) * ((d - 0.3) / 0.4)
        else:
            c = hot_color
        pixels[i, j] = c

# =============================================================================
# AUDIO ANALYZER (simplified - no playback needed)
# =============================================================================
class AudioAnalyzer:
    def __init__(self, filepath):
        self.fs, self.data = wavfile.read(filepath)
        if len(self.data.shape) > 1:
            self.stereo_data = self.data.copy()
            self.data = self.data.mean(axis=1)
        else:
            self.stereo_data = np.column_stack([self.data, self.data])
        self.data = self.data / np.max(np.abs(self.data))
        max_val = np.max(np.abs(self.stereo_data))
        if max_val > 0:
            self.stereo_data = self.stereo_data / max_val
        self.audio_length = len(self.data) / self.fs
        self.bass_history = []
        self.treble_history = []
        self.stereo_spread_history = []
        self.current_pan = 0.0
        self.max_bass = 0.0
        self.max_treble = 0.0
        self.max_stereo_spread = 0.0
        print(f"Loaded {filepath}. Duration: {self.audio_length:.2f}s, Sample rate: {self.fs}")

    def get_energy(self, sim_time):
        if sim_time >= self.audio_length:
            return 0.0, 0.0, 0.0, 0.0
        idx = int(sim_time * self.fs)
        window = 4096
        if idx + window >= len(self.data):
            return 0.0, 0.0, 0.0, 0.0
        chunk = self.data[idx:idx+window]
        stereo_chunk = self.stereo_data[idx:idx+window]
        rms = np.sqrt(np.mean(chunk**2))
        spectrum = np.abs(rfft(chunk))
        freqs = rfftfreq(window, 1 / self.fs)

        bass_mask = (freqs >= 40) & (freqs <= 200)
        bass_spectrum = spectrum[bass_mask] if np.any(bass_mask) else np.array([0.0])
        bass_peak = np.max(bass_spectrum) if len(bass_spectrum) > 0 else 0.0
        bass_mean = np.mean(bass_spectrum) if len(bass_spectrum) > 0 else 0.0
        bass_raw = (bass_peak * 0.7 + bass_mean * 0.3) * 2.0 + rms * 1.5

        high_mask = (freqs >= 2000) & (freqs <= 10000)
        high_spectrum = spectrum[high_mask] if np.any(high_mask) else np.array([0.0])
        high_peak = np.max(high_spectrum) if len(high_spectrum) > 0 else 0.0
        high_mean = np.mean(high_spectrum) if len(high_spectrum) > 0 else 0.0
        high_raw = (high_peak * 0.6 + high_mean * 0.4) * 1.5 + rms * 0.8

        if bass_raw > self.max_bass: self.max_bass = bass_raw
        if high_raw > self.max_treble: self.max_treble = high_raw

        self.bass_history.append(bass_raw)
        self.treble_history.append(high_raw)
        if len(self.bass_history) > 3:
            self.bass_history.pop(0)
            self.treble_history.pop(0)

        bass_smoothed = np.mean(self.bass_history)
        treble_smoothed = np.mean(self.treble_history)

        # Stereo spread
        if len(stereo_chunk.shape) > 1 and stereo_chunk.shape[1] >= 2:
            left_rms = np.sqrt(np.mean(stereo_chunk[:, 0]**2))
            right_rms = np.sqrt(np.mean(stereo_chunk[:, 1]**2))
            total_energy = left_rms + right_rms
            if total_energy > 0:
                pan = (right_rms - left_rms) / total_energy
                stereo_spread_raw = abs(pan)
                self.current_pan = pan
            else:
                stereo_spread_raw = 0.0
                self.current_pan = 0.0
        else:
            stereo_spread_raw = 0.0
            self.current_pan = 0.0

        if stereo_spread_raw > self.max_stereo_spread:
            self.max_stereo_spread = stereo_spread_raw

        self.stereo_spread_history.append(stereo_spread_raw)
        if len(self.stereo_spread_history) > 3:
            self.stereo_spread_history.pop(0)
        stereo_spread_smoothed = np.mean(self.stereo_spread_history)

        bass_scale = max(self.max_bass, 0.1)
        treble_scale = max(self.max_treble, 0.1)
        stereo_scale = max(self.max_stereo_spread, 0.01)

        bass_energy = min(bass_smoothed / bass_scale, 1.0)
        high_energy = min(treble_smoothed / treble_scale, 1.0)
        stereo_spread = min(stereo_spread_smoothed / stereo_scale, 1.0)

        return bass_energy, high_energy, stereo_spread, self.current_pan


# =============================================================================
# MAIN - Headless capture
# =============================================================================
def main():
    audio = AudioAnalyzer(AUDIO_FILE)

    density_mult = 0.5
    music_resp = 1.0
    bass_speed = 100.0
    base_speed = 3.0
    decay_rate = 0.99

    total_frames = CAPTURE_DURATION * FPS
    frame_interval = 1.0 / FPS  # simulated time per frame

    print(f"Capturing {CAPTURE_DURATION}s at {FPS}fps ({total_frames} frames) to {OUTPUT_FILE}")

    # Start ffmpeg process to encode frames
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{RES}x{RES}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-i", AUDIO_FILE,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-pix_fmt", "yuv420p",
        OUTPUT_FILE,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    start_time = time.time()
    for frame_idx in range(total_frames):
        sim_time = frame_idx * frame_interval

        # Get audio energy at this simulated time
        bass, treble, stereo_spread, pan = audio.get_energy(sim_time)

        # Apply emission
        total_energy = bass + treble * 0.5
        if total_energy > 0.05:
            source_x = 0.5
            source_y = 0.02
            responsive_bass = bass * music_resp
            responsive_treble = treble * music_resp
            responsive_stereo = stereo_spread * music_resp
            base_flow_y = base_speed
            bass_boost_y = responsive_bass * bass_speed
            base_spread = 30.0
            max_spread_angle = 60.0
            spread_width = base_spread + (responsive_stereo * (max_spread_angle - base_spread))
            base_density = 0.01
            d_val = (base_density + (responsive_bass * 0.3) + (responsive_treble * 0.15)) * density_mult
            radius = 0.03 + (responsive_bass * 0.05)

            apply_fanned_emission(velocity, density_field, source_x, source_y,
                                  radius, base_flow_y, bass_boost_y,
                                  pan, spread_width, d_val)

        # Physics steps
        advect(velocity, velocity, _new_velocity, decay_rate)
        velocity.copy_from(_new_velocity)
        advect(velocity, density_field, _new_density, decay_rate)
        density_field.copy_from(_new_density)
        compute_divergence(velocity, divergence)
        pressure.fill(0)
        for _ in range(p_jacobi_iters):
            pressure_jacobi(pressure, _new_pressure, divergence)
            pressure.copy_from(_new_pressure)
        subtract_gradient(velocity, pressure)

        # Render
        render_display(density_field, density_mult * 3.0)

        # Extract pixel data and write to ffmpeg
        img = pixels.to_numpy()
        # img shape is (RES, RES, 3) with values 0-1, need uint8
        # Flip vertically (taichi has origin at bottom-left, video at top-left)
        img = np.flipud(img)
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        proc.stdin.write(img_uint8.tobytes())

        if (frame_idx + 1) % FPS == 0:
            elapsed = time.time() - start_time
            sim_sec = (frame_idx + 1) // FPS
            print(f"  {sim_sec}/{CAPTURE_DURATION}s captured ({elapsed:.1f}s elapsed) "
                  f"- Bass: {bass:.3f}, Treble: {treble:.3f}")

    proc.stdin.close()
    proc.wait()
    elapsed = time.time() - start_time
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nDone! {OUTPUT_FILE} ({file_size:.1f} MB) captured in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
