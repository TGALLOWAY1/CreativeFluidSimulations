import taichi as ti
import numpy as np
import time
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import pygame

# Initialize Taichi (Auto-detects GPU: CUDA, Metal, or Vulkan)
ti.init(arch=ti.gpu)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
RES = 512              # Simulation Resolution (Try 1024 for high-end GPUs)
dt = 0.02              # Time step (reduced for slower movement)
p_jacobi_iters = 40    # Pressure iterations (Higher = more "swirly", slower)
force_radius = 0.1
damp = 0.985           # Density decay (faster decay when music stops)

# Tunable parameters (will be controlled by GUI sliders)
density_multiplier = 0.15      # Overall density scale
music_responsiveness = 1.0     # How much smoke responds to music (0-2)
bass_speed_multiplier =90.0   # How much bass affects speed
base_flow_speed = 6.0          # Base upward velocity

AUDIO_FILE = "REF SONGS/TestSong.wav" # <--- PUT YOUR SONG HERE

# =============================================================================
# 2. DATA FIELDS (The Grid)
# =============================================================================
# Velocity vector field (u, v)
velocity = ti.Vector.field(2, dtype=float, shape=(RES, RES))
_new_velocity = ti.Vector.field(2, dtype=float, shape=(RES, RES))

# Density/Color field (scalar)
density = ti.field(dtype=float, shape=(RES, RES))
_new_density = ti.field(dtype=float, shape=(RES, RES))

# Helper fields for the solver
divergence = ti.field(dtype=float, shape=(RES, RES))
pressure = ti.field(dtype=float, shape=(RES, RES))
_new_pressure = ti.field(dtype=float, shape=(RES, RES))

# Visualization buffer (RGB pixels)
pixels = ti.Vector.field(3, dtype=float, shape=(RES, RES))

# =============================================================================
# 3. PHYSICS KERNELS (Running on GPU)
# =============================================================================

@ti.func
def sample_field(field: ti.template(), x: float, y: float):
    """Bilinear interpolation for fields (works with both scalar and vector fields)"""
    # Clamp to valid range
    x = ti.max(0.0, ti.min(x, RES - 1.0))
    y = ti.max(0.0, ti.min(y, RES - 1.0))
    
    # Get integer coordinates
    i0 = int(x)
    j0 = int(y)
    i1 = ti.min(i0 + 1, RES - 1)
    j1 = ti.min(j0 + 1, RES - 1)
    
    # Get fractional parts
    fx = x - i0
    fy = y - j0
    
    # Sample the 4 corners
    v00 = field[i0, j0]
    v10 = field[i1, j0]
    v01 = field[i0, j1]
    v11 = field[i1, j1]
    
    # Bilinear interpolation
    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return v0 * (1.0 - fy) + v1 * fy

@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template(), decay: float):
    """
    Move quantities (qf) along the velocity field (vf).
    Solves: q(x, t+dt) = q(x - u*dt, t)
    """
    for i, j in vf:
        coord = ti.Vector([i, j]) + 0.5
        vel = vf[i, j]
        prev_coord = coord - vel * dt
        
        # Clamp coordinates to grid boundaries
        prev_coord = ti.max(0.5, ti.min(prev_coord, RES - 0.5))
        
        # Linear interpolation (Bi-lerp) for smooth movement with decay
        new_qf[i, j] = sample_field(qf, prev_coord[0] - 0.5, prev_coord[1] - 0.5) * decay

@ti.kernel
def apply_impulse(vf: ti.template(), df: ti.template(), 
                  x: float, y: float, 
                  r: float, v_x: float, v_y: float, d_val: float):
    """Injects velocity and density at a specific point (Mouse or Audio Force)"""
    for i, j in vf:
        dx = i - x * RES
        dy = j - y * RES
        dist2 = dx*dx + dy*dy
        
        if dist2 < (r * RES)**2:
            falloff = ti.exp(-dist2 / (r * RES))
            vf[i, j] += ti.Vector([v_x, v_y]) * falloff
            df[i, j] += d_val * falloff

@ti.kernel
def apply_fanned_emission(vf: ti.template(), df: ti.template(),
                          x: float, y: float,
                          r: float, base_speed: float, bass_boost: float,
                          pan: float, spread_width: float, d_val: float):
    """Emits smoke with variable angles to create a fan pattern"""
    for i, j in vf:
        dx = i - x * RES
        dy = j - y * RES
        dist2 = dx*dx + dy*dy
        
        if dist2 < (r * RES)**2:
            falloff = ti.exp(-dist2 / (r * RES))
            
            # Calculate angle for this cell based on horizontal position
            # Cells to the left get negative angles, right get positive
            # Use dx to determine the angle offset from center
            max_dist = r * RES
            if max_dist > 0:
                # Normalize horizontal position (-1 to +1) based on radius
                normalized_x = dx / max_dist
                # Clamp to prevent extreme angles
                normalized_x = ti.max(-1.0, ti.min(1.0, normalized_x))
                
                # Pan shifts the center direction
                pan_offset = pan * (spread_width * 0.4)
                # Each cell gets an angle based on its position
                # Use full spread_width range for maximum fan effect
                cell_angle = pan_offset + (normalized_x * spread_width * 0.5)
                
                # Add pseudo-random variation using cell indices for natural spread
                # Use hash of cell position for deterministic but varied angles
                hash_val = float((i * 73856093) ^ (j * 19349663)) * 0.000001
                hash_val = hash_val - int(hash_val)  # Get fractional part (0-1)
                random_variation = (hash_val - 0.5) * spread_width * 0.2
                total_angle = cell_angle + random_variation
                
                # Convert angle to velocity components
                angle_rad = total_angle * 3.14159 / 180.0  # degrees to radians
                speed = base_speed + bass_boost
                
                # Calculate velocity with proper fanning
                # Make horizontal component much stronger for visible fan
                v_x = ti.sin(angle_rad) * speed * 1.2  # Much stronger horizontal
                v_y = ti.cos(angle_rad) * speed  # Vertical component
                
                vf[i, j] += ti.Vector([v_x, v_y]) * falloff
                df[i, j] += d_val * falloff

@ti.kernel
def apply_shockwave(vf: ti.template(), cx: float, cy: float,
                    strength: float, ring_radius: float, ring_width: float):
    """Radial outward velocity burst — expanding ring that pushes existing smoke."""
    for i, j in vf:
        dx = float(i) - cx * RES
        dy = float(j) - cy * RES
        dist = ti.sqrt(dx * dx + dy * dy)

        # Ring-shaped Gaussian falloff: peaks at ring_radius, decays with ring_width
        ring_factor = ti.exp(-((dist - ring_radius) ** 2) / (2.0 * ring_width * ring_width))

        if dist > 0.5 and ring_factor > 0.01:
            # Radial outward direction
            nx = dx / dist
            ny = dy / dist
            push = strength * ring_factor
            vf[i, j] += ti.Vector([nx, ny]) * push

@ti.kernel
def apply_edge_turbulence(vf: ti.template(), df: ti.template(),
                          treble_strength: float, time_seed: int):
    """Inject high-frequency velocity perturbations at smoke edges (density gradients)."""
    for i, j in vf:
        if i < 2 or i >= RES - 2 or j < 2 or j >= RES - 2:
            continue

        # Density gradient magnitude — identifies smoke edges
        dx_grad = df[i + 1, j] - df[i - 1, j]
        dy_grad = df[i, j + 1] - df[i, j - 1]
        grad_mag = ti.sqrt(dx_grad * dx_grad + dy_grad * dy_grad)

        if grad_mag > 0.01:
            # Spatial hash for pseudo-random direction (deterministic per cell)
            hash1 = float((i * 73856093) ^ (j * 19349663)) * 0.0000001
            hash1 = hash1 - ti.floor(hash1)
            hash2 = float((i * 83492791) ^ (j * 47420677)) * 0.0000001
            hash2 = hash2 - ti.floor(hash2)

            # Time-varying hash for sizzle — changes every frame
            t_hash = float((time_seed * 12345) ^ (i * 7919) ^ (j * 6271)) * 0.0000001
            t_hash = t_hash - ti.floor(t_hash)

            # Combine spatial + temporal randomness for rapidly changing perturbation
            perturb_x = (hash1 + t_hash - 1.0) * 2.0
            perturb_y = (hash2 + t_hash - 1.0) * 2.0

            # Scale by gradient magnitude (edges only) and treble energy
            scale = grad_mag * treble_strength
            vf[i, j] += ti.Vector([perturb_x, perturb_y]) * scale

@ti.kernel
def compute_divergence(vf: ti.template(), div: ti.template()):
    """Calculate how much fluid is entering/leaving each cell."""
    for i, j in vf:
        # Neighbor lookups (with boundary checks)
        wl = vf[max(0, i-1), j][0]
        wr = vf[min(RES-1, i+1), j][0]
        wb = vf[i, max(0, j-1)][1]
        wt = vf[i, min(RES-1, j+1)][1]
        
        div[i, j] = 0.5 * (wr - wl + wt - wb)

@ti.kernel
def pressure_jacobi(p: ti.template(), new_p: ti.template(), div: ti.template()):
    """Solve Poisson equation for pressure."""
    for i, j in p:
        pl = p[max(0, i-1), j]
        pr = p[min(RES-1, i+1), j]
        pb = p[i, max(0, j-1)]
        pt = p[i, min(RES-1, j+1)]
        
        new_p[i, j] = (pl + pr + pb + pt - div[i, j]) * 0.25

@ti.kernel
def subtract_gradient(vf: ti.template(), p: ti.template()):
    """Subtract pressure gradient from velocity to enforce incompressibility."""
    for i, j in vf:
        pl = p[max(0, i-1), j]
        pr = p[min(RES-1, i+1), j]
        pb = p[i, max(0, j-1)]
        pt = p[i, min(RES-1, j+1)]
        
        vel = vf[i, j]
        mask = 1.0
        
        # Boundary conditions (Pure Neumman - walls reflect)
        if i == 0 or i == RES-1: mask = 0.0
        if j == 0 or j == RES-1: mask = 0.0

        vf[i, j] = vel - 0.5 * ti.Vector([pr - pl, pt - pb]) * mask

@ti.kernel
def render_display(df: ti.template(), density_scale: float):
    """Map density to color palette (Dark Blue -> Cyan)"""
    for i, j in pixels:
        d = df[i, j] * density_scale  # Scale density for visualization
        
        # Color Palette - adjusted to prevent all-white
        # Base: Dark Blue/Black
        base_color = ti.Vector([0.0, 0.02, 0.05])
        # Highlight: Cyan/Electric Blue
        light_color = ti.Vector([0.0, 0.7, 0.85])
        # Hot: Bright Cyan (not pure white)
        hot_color = ti.Vector([0.2, 0.9, 1.0])
        
        c = ti.Vector([0.0, 0.0, 0.0])
        
        # Adjusted thresholds to prevent saturation
        if d < 0.3:
            c = base_color + (light_color - base_color) * (d / 0.3)
        elif d < 0.7:
            c = light_color + (hot_color - light_color) * ((d - 0.3) / 0.4)
        else:
            # Cap at hot_color, don't go to pure white
            c = hot_color
            
        pixels[i, j] = c

# =============================================================================
# 4. AUDIO PROCESSOR
# =============================================================================
class AudioAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath  # Store filepath for restart
        try:
            # Initialize pygame mixer for audio playback
            pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            
            # Load audio data for analysis
            self.fs, self.data = wavfile.read(filepath)
            # Preserve stereo channels for stereo spread analysis
            if len(self.data.shape) > 1:
                self.stereo_data = self.data.copy()  # Keep stereo for spread analysis
                self.data = self.data.mean(axis=1) # Convert to mono for frequency analysis
            else:
                self.stereo_data = np.column_stack([self.data, self.data])  # Duplicate for mono files
            self.data = self.data / np.max(np.abs(self.data)) # Normalize
            # Normalize stereo data
            max_val = np.max(np.abs(self.stereo_data))
            if max_val > 0:
                self.stereo_data = self.stereo_data / max_val
            
            # Load and play audio with pygame Sound (better WAV support)
            self.sound = pygame.mixer.Sound(filepath)
            # Set volume to maximum (1.0)
            self.sound.set_volume(1.0)
            # Play the sound
            channel = self.sound.play(loops=0)
            
            self.has_audio = True
            self.audio_length = len(self.data) / self.fs
            self.start_time = time.time()
            print(f"Loaded {filepath} successfully. Duration: {self.audio_length:.2f}s")
            print(f"Audio playback started. Volume: {self.sound.get_volume()}")
        except Exception as e:
            import traceback
            print(f"Could not load audio ({e}). Running in silent mode.")
            traceback.print_exc()
            self.has_audio = False
            self.audio_length = 0
            self.start_time = time.time()
            
        self.last_idx = 0
        self.bass_history = []
        self.treble_history = []
        self.stereo_spread_history = []
        self.current_pan = 0.0  # Current pan position (-1 to +1)
        # Track max values for adaptive normalization
        self.max_bass = 0.0
        self.max_treble = 0.0
        self.max_stereo_spread = 0.0
        # Transient/onset detection (unsmoothed frame-to-frame delta)
        self.prev_bass_raw = 0.0
        self.max_bass_delta = 0.0
        
    def get_energy(self):
        if not self.has_audio:
            # Mock beat if no audio
            t = time.time() - self.start_time
            return (np.sin(t * 10) > 0.8) * 0.8, 0.0, 0.0, 0.0, 0.0
        
        # Track playback time manually (Sound objects don't have get_pos())
        current_time = time.time() - self.start_time
        
        # Check if audio has finished playing
        if current_time >= self.audio_length:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        idx = int(current_time * self.fs)
        
        # Larger window for better frequency resolution
        window = 4096
        if idx + window >= len(self.data):
            return 0.0, 0.0, 0.0, 0.0, 0.0 # End of song
            
        chunk = self.data[idx:idx+window]
        stereo_chunk = self.stereo_data[idx:idx+window] if idx + window < len(self.stereo_data) else self.stereo_data[idx:]
        
        # Calculate RMS (overall loudness) for beat detection
        rms = np.sqrt(np.mean(chunk**2))
        
        # Frequency analysis
        spectrum = np.abs(rfft(chunk))
        freqs = rfftfreq(window, 1 / self.fs)
        
        # Bass Range (40Hz - 200Hz) - focused on kick drum range
        bass_mask = (freqs >= 40) & (freqs <= 200)
        bass_spectrum = spectrum[bass_mask] if np.any(bass_mask) else np.array([0.0])
        # Use peak energy instead of mean for more dramatic response
        bass_peak = np.max(bass_spectrum) if len(bass_spectrum) > 0 else 0.0
        bass_mean = np.mean(bass_spectrum) if len(bass_spectrum) > 0 else 0.0
        # Combine peak and mean (reduced multipliers to prevent capping)
        bass_raw = (bass_peak * 0.7 + bass_mean * 0.3) * 2.0 + rms * 1.5

        # Transient detection: unsmoothed frame-to-frame increase (bypasses smoothing)
        bass_delta = max(0.0, bass_raw - self.prev_bass_raw)
        self.prev_bass_raw = bass_raw
        if bass_delta > self.max_bass_delta:
            self.max_bass_delta = bass_delta

        # Treble Range (2000Hz - 10000Hz) - for hi-hats and cymbals
        high_mask = (freqs >= 2000) & (freqs <= 10000)
        high_spectrum = spectrum[high_mask] if np.any(high_mask) else np.array([0.0])
        high_peak = np.max(high_spectrum) if len(high_spectrum) > 0 else 0.0
        high_mean = np.mean(high_spectrum) if len(high_spectrum) > 0 else 0.0
        high_raw = (high_peak * 0.6 + high_mean * 0.4) * 1.5 + rms * 0.8
        
        # Track maximum values for adaptive normalization
        if bass_raw > self.max_bass:
            self.max_bass = bass_raw
        if high_raw > self.max_treble:
            self.max_treble = high_raw
        
        # Store history for smoothing
        self.bass_history.append(bass_raw)
        self.treble_history.append(high_raw)
        if len(self.bass_history) > 2:
            self.bass_history.pop(0)
        if len(self.treble_history) > 3:
            self.treble_history.pop(0)
        
        # Use recent average for smoother response
        bass_smoothed = np.mean(self.bass_history) if self.bass_history else bass_raw
        treble_smoothed = np.mean(self.treble_history) if self.treble_history else high_raw
        
        # Calculate stereo spread (left vs right channel difference)
        if len(stereo_chunk.shape) > 1 and stereo_chunk.shape[1] >= 2:
            left_channel = stereo_chunk[:, 0]
            right_channel = stereo_chunk[:, 1]
            
            # Calculate RMS for each channel
            left_rms = np.sqrt(np.mean(left_channel**2))
            right_rms = np.sqrt(np.mean(right_channel**2))
            
            # Stereo spread: difference between channels normalized by total energy
            total_energy = left_rms + right_rms
            if total_energy > 0:
                # Pan value: -1 (fully left) to +1 (fully right)
                pan = (right_rms - left_rms) / total_energy
                # Spread: how much difference (0 = mono, 1 = fully panned)
                stereo_spread_raw = abs(pan)
                # Store pan for angle calculation
                self.current_pan = pan
            else:
                stereo_spread_raw = 0.0
                self.current_pan = 0.0
        else:
            stereo_spread_raw = 0.0
            self.current_pan = 0.0
        
        # Track max stereo spread
        if stereo_spread_raw > self.max_stereo_spread:
            self.max_stereo_spread = stereo_spread_raw
        
        # Store history for smoothing
        self.stereo_spread_history.append(stereo_spread_raw)
        if len(self.stereo_spread_history) > 3:
            self.stereo_spread_history.pop(0)
        
        stereo_spread_smoothed = np.mean(self.stereo_spread_history) if self.stereo_spread_history else stereo_spread_raw
        
        # Normalize using adaptive scaling (use max seen so far, with minimum threshold)
        # This prevents everything from capping at 1.0
        bass_scale = max(self.max_bass, 0.1)  # Minimum scale to prevent division issues
        treble_scale = max(self.max_treble, 0.1)
        stereo_scale = max(self.max_stereo_spread, 0.01)
        
        bass_energy = min(bass_smoothed / bass_scale, 1.0)
        high_energy = min(treble_smoothed / treble_scale, 1.0)
        stereo_spread = min(stereo_spread_smoothed / stereo_scale, 1.0)

        # Normalize transient (unsmoothed, raw spike for kick punch)
        transient_scale = max(self.max_bass_delta, 0.1)
        transient = min(bass_delta / transient_scale, 1.0)

        return bass_energy, high_energy, stereo_spread, self.current_pan, transient
    
    def stop(self):
        """Stop audio playback"""
        if self.has_audio:
            self.sound.stop()
    
    def restart(self):
        """Restart audio playback from the beginning"""
        if self.has_audio:
            self.sound.stop()
            self.sound = pygame.mixer.Sound(self.filepath)
            self.sound.set_volume(1.0)
            self.sound.play(loops=0)
            self.start_time = time.time()
            self.bass_history = []
            self.treble_history = []
            self.stereo_spread_history = []
            self.max_bass = 0.0
            self.max_treble = 0.0
            self.max_stereo_spread = 0.0
            self.prev_bass_raw = 0.0
            self.max_bass_delta = 0.0
            print("Audio restarted")
    
    def pause(self):
        """Pause audio playback"""
        if self.has_audio:
            pygame.mixer.pause()
    
    def unpause(self):
        """Unpause audio playback"""
        if self.has_audio:
            pygame.mixer.unpause()

# =============================================================================
# 5. MAIN LOOP
# =============================================================================

def main():
    gui = ti.GUI("Audio Smoke", res=(RES, RES))
    audio = AudioAnalyzer(AUDIO_FILE)
    
    # Tunable parameters (will be controlled by sliders)
    density_mult = 0.5
    music_resp = 1.0
    bass_speed = 100.0
    base_speed = 3.0
    decay_rate = 0.99
    
    # Shockwave state (triggered by kick/snare transients)
    shockwave_active = False
    shockwave_radius = 0.0
    shockwave_strength = 0.0
    SHOCKWAVE_THRESHOLD = 0.4
    SHOCKWAVE_INITIAL_STRENGTH = 25.0
    SHOCKWAVE_EXPAND_SPEED = 8.0
    SHOCKWAVE_RING_WIDTH = 15.0
    SHOCKWAVE_DECAY = 0.85

    print("Starting simulation... (Press ESC to exit)")
    print("Keyboard Controls:")
    print("  Q/A: Decrease/Increase Density Multiplier")
    print("  W/S: Decrease/Increase Music Responsiveness")
    print("  E/D: Decrease/Increase Bass Speed Multiplier")
    print("  R/F: Decrease/Increase Decay Rate")
    print("  T: Reset all to defaults")
    print("  SPACE: Restart song and animation")
    print("  P: Pause/Unpause")
    
    frame_count = 0
    show_controls = True
    paused = False
    try:
        while gui.running:
            # Handle keyboard input for real-time tuning
            for e in gui.get_events(ti.GUI.PRESS):
                if e.key == ti.GUI.ESCAPE:
                    break
                elif e.key == 'q':
                    density_mult = max(0.0, density_mult - 0.01)
                    print(f"Density: {density_mult:.3f}")
                elif e.key == 'a':
                    density_mult = min(1.0, density_mult + 0.01)
                    print(f"Density: {density_mult:.3f}")
                elif e.key == 'w':
                    music_resp = max(0.0, music_resp - 0.1)
                    print(f"Music Resp: {music_resp:.2f}")
                elif e.key == 's':
                    music_resp = min(2.0, music_resp + 0.1)
                    print(f"Music Resp: {music_resp:.2f}")
                elif e.key == 'e':
                    bass_speed = max(0.0, bass_speed - 5.0)
                    print(f"Bass Speed: {bass_speed:.1f}")
                elif e.key == 'd':
                    bass_speed = min(100.0, bass_speed + 5.0)
                    print(f"Bass Speed: {bass_speed:.1f}")
                elif e.key == 'r':
                    decay_rate = max(0.95, decay_rate - 0.002)
                    print(f"Decay Rate: {decay_rate:.4f}")
                elif e.key == 'f':
                    decay_rate = min(0.999, decay_rate + 0.002)
                    print(f"Decay Rate: {decay_rate:.4f}")
                elif e.key == 't':
                    density_mult = 0.5
                    music_resp = 1.0
                    bass_speed = 100.0
                    decay_rate = 0.99
                    print("Reset to defaults")
                elif e.key == 'h':
                    show_controls = not show_controls
                elif e.key == 'p' or e.key == 'P':
                    # Pause/Unpause
                    paused = not paused
                    if paused:
                        audio.pause()
                        print("Paused")
                    else:
                        audio.unpause()
                        print("Unpaused")
                elif e.key == ' ' or e.key == ti.GUI.SPACE:
                    # Restart song and animation
                    audio.restart()
                    # Clear all simulation fields
                    velocity.fill([0.0, 0.0])
                    density.fill(0.0)
                    pressure.fill(0.0)
                    divergence.fill(0.0)
                    frame_count = 0
                    paused = False
                    shockwave_active = False
                    shockwave_radius = 0.0
                    shockwave_strength = 0.0
                    print("Animation and song restarted!")
            
            # Skip simulation if paused
            if paused:
                # Still render the current state
                render_display(density, density_mult * 3.0)
                gui.set_image(pixels)
                if show_controls:
                    gui.text("PAUSED - Press P to resume", (0.01, 0.77), font_size=20, color=0xFF0000)
                gui.show()
                continue
            
            # 1. Get Audio Data
            bass, treble, stereo_spread, pan, transient = audio.get_energy()
            
            # Debug output every 60 frames (~1 second at 60fps)
            if frame_count % 60 == 0:
                print(f"Bass: {bass:.3f}, Treble: {treble:.3f}, Stereo: {stereo_spread:.3f}, Density: {density_mult:.3f}, Resp: {music_resp:.3f}")
            frame_count += 1
            
            # 2. Apply Inputs
            # Mouse interaction disabled - only audio-driven smoke
            
            # B. BOTTOM SMOKE EMITTER
            # Only emit if there's significant audio energy
            total_energy = bass + treble * 0.5
            if total_energy > 0.05:  # Threshold to prevent emission when music stops
                # Source Position: Bottom Center
                source_x = 0.5
                source_y = 0.02
                
                # Apply music responsiveness multiplier
                responsive_bass = bass * music_resp
                responsive_treble = treble * music_resp
                responsive_stereo = stereo_spread * music_resp
                
                # Velocity (Upward Force)
                base_flow_y = base_speed
                bass_boost_y = responsive_bass * bass_speed
                
                # Emission angle system: stereo controls spread width, pan controls direction
                # Base spread: always some diffusion (minimum 30 degrees for visible fan)
                base_spread = 30.0  # Minimum spread in degrees - much more visible
                
                # Stereo spread controls the total width of the emission cone
                # At max stereo (1.0), spread reaches 60 degrees total (±30 degrees)
                # At min stereo (0.0), spread is still base_spread for visible fan
                max_spread_angle = 60.0  # Maximum total spread - wider fan
                spread_width = base_spread + (responsive_stereo * (max_spread_angle - base_spread))
                
                # Density (Amount of Smoke) — transient adds brightness flash on kicks
                base_density = 0.01
                d_val = (base_density + (responsive_bass * 0.3) + (responsive_treble * 0.15) + transient * 0.5) * density_mult

                # Dynamic Radius - expands with bass + transient spike
                radius = 0.03 + (responsive_bass * 0.05) + (transient * 0.04)
                
                # Debug: print spread info occasionally
                if frame_count % 60 == 0:
                    print(f"Spread width: {spread_width:.1f}°, Pan: {pan:.3f}, Stereo: {stereo_spread:.3f}")
                
                # Use fanned emission kernel that varies angle per cell
                apply_fanned_emission(velocity, density, source_x, source_y,
                                     radius, base_flow_y, bass_boost_y,
                                     pan, spread_width, d_val)

            # Shockwave: radial pressure wave on kick/snare transients
            if transient > SHOCKWAVE_THRESHOLD and not shockwave_active:
                shockwave_active = True
                shockwave_radius = 0.0
                shockwave_strength = SHOCKWAVE_INITIAL_STRENGTH * transient

            if shockwave_active:
                shockwave_radius += SHOCKWAVE_EXPAND_SPEED
                apply_shockwave(velocity, 0.5, 0.02,
                                shockwave_strength, shockwave_radius, SHOCKWAVE_RING_WIDTH)
                shockwave_strength *= SHOCKWAVE_DECAY
                if shockwave_strength < 0.5:
                    shockwave_active = False

            # Treble edge turbulence: sizzle at smoke boundaries
            if treble > 0.1:
                apply_edge_turbulence(velocity, density, treble * 8.0, frame_count)

            # 3. Physics Steps
            # Advect Velocity
            advect(velocity, velocity, _new_velocity, decay_rate)
            velocity.copy_from(_new_velocity)
            
            # Advect Density
            advect(velocity, density, _new_density, decay_rate)
            density.copy_from(_new_density)
            
            # Compute Divergence
            compute_divergence(velocity, divergence)
            
            # Solve Pressure (Jacobi Iteration)
            pressure.fill(0) # Reset pressure guess
            for _ in range(p_jacobi_iters):
                pressure_jacobi(pressure, _new_pressure, divergence)
                pressure.copy_from(_new_pressure)
                
            # Subtract Gradient (Enforce Incompressibility)
            subtract_gradient(velocity, pressure)

            # 4. Render
            render_display(density, density_mult * 3.0)  # Scale for visualization
            gui.set_image(pixels)
            
            # Display current parameter values on screen
            if show_controls:
                gui.text("Density: {:.3f} (Q/A)".format(density_mult), (0.01, 0.95), font_size=15, color=0xFFFFFF)
                gui.text("Music Resp: {:.2f} (W/S)".format(music_resp), (0.01, 0.92), font_size=15, color=0xFFFFFF)
                gui.text("Bass Speed: {:.1f} (E/D)".format(bass_speed), (0.01, 0.89), font_size=15, color=0xFFFFFF)
                gui.text("Decay: {:.4f} (R/F)".format(decay_rate), (0.01, 0.86), font_size=15, color=0xFFFFFF)
                gui.text("Bass: {:.3f} Treble: {:.3f} Stereo: {:.3f}".format(bass, treble, stereo_spread), (0.01, 0.83), font_size=15, color=0x00FFFF)
                gui.text("SPACE: Restart | P: Pause | H: Toggle Help", (0.01, 0.80), font_size=15, color=0xFFFF00)
            
            gui.show()
    finally:
        # Clean up audio when exiting
        audio.stop()
        pygame.mixer.quit()

if __name__ == "__main__":
    main()
