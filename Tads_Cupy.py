import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np # Still needed for .get() results and some matplotlib interactions

# Try importing CuPy and CuPyX signal
try:
    import cupy as cp
    from cupyx.scipy import signal as cusignal
    print("CuPy found and imported successfully.")
    # Set the current GPU device (optional, usually defaults to device 0)
    # cp.cuda.Device(0).use()
except ImportError:
    messagebox.showerror("Import Error",
                         "CuPy or cupyx.scipy.signal not found.\n"
                         "Please install CuPy matching your CUDA version.\n"
                         "Example: pip install cupy-cuda11x\n\n"
                         "Falling back to NumPy (performance will be CPU-bound).")
    # Fallback to NumPy if CuPy is not available
    cp = np
    from scipy import signal as cusignal # Use SciPy signal as fallback
    IS_CUPY_AVAILABLE = False
else:
    IS_CUPY_AVAILABLE = True


import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import threading

class TADSSimulator:
    """
    A TADS-like 2D field with wave speed modulated by tension,
    plus a collision nonlinearity, optimized for producing atom-like structures.
    Uses CuPy for GPU acceleration.
    """
    def __init__(self, grid_size=100):
        self.grid_size = grid_size

        # Default parameters - set to the values that produce interesting atom-like behavior
        self.dt = 0.1
        self.damping = 0.000
        self.base_wave_speed_sq = 2.0
        self.tension_wave_factor = 16.279
        self.collision_threshold = 0.328
        self.collision_strength = 0.2  # We'll adjust this during the simulation
        self.noise_level = 0.0000
        self.init_std_dev = 0.1
        self.max_energy = 505.0

        # Kernel needs to be a NumPy array for convolve2d even with CuPyX?
        # Let's try keeping it as CuPy array directly if cusignal handles it
        self.laplacian_kernel = cp.array([[0, 1, 0],
                                          [1, -4, 1],
                                          [0, 1, 0]], dtype=cp.float32)

        self.step_count = 0
        self.has_collapsed = False
        self.particle_centers = [] # Store particle centers on CPU

        # Initialize fields on the GPU
        self._initialize_fields()

    def _initialize_fields(self):
        """Initializes or re-initializes the fields on the GPU."""
        print(f"Initializing fields with grid size: {self.grid_size}x{self.grid_size}")
        if IS_CUPY_AVAILABLE:
            cp.random.seed(42) # Seed the GPU random generator
        else:
            np.random.seed(42) # Seed the CPU random generator (fallback)

        self.phi = (cp.random.rand(self.grid_size, self.grid_size, dtype=cp.float32) - 0.5) * 2 * self.init_std_dev
        self.phi_old = cp.copy(self.phi)

        # We'll store the local wave speed for display
        self.local_wave_speed_sq = cp.full((self.grid_size, self.grid_size),
                                          self.base_wave_speed_sq, dtype=cp.float32)
        print("Fields initialized.")


    def create_gaussian_perturbation(self, location=None, radius=5, strength=0.5):
        """Create a Gaussian perturbation in the field (Operates on GPU array)"""
        if location is None:
            # Default to center
            x_c, y_c = self.grid_size // 2, self.grid_size // 2
        else:
            x_c, y_c = location

        # Create coordinate grids on the GPU
        y_gpu, x_gpu = cp.indices((self.grid_size, self.grid_size), dtype=cp.float32)

        # Calculate Gaussian values on the GPU
        dist_sq = (x_gpu - x_c)**2 + (y_gpu - y_c)**2
        variance = (radius / 2)**2
        gaussian_vals = strength * cp.exp(-dist_sq / (2 * variance))

        # Add the perturbation to the field
        self.phi += gaussian_vals

        # Apply boundary condition or clipping if needed (optional)
        # cp.clip(self.phi, -some_max, some_max, out=self.phi) # Example

        self.phi_old = cp.copy(self.phi)
        print(f"Gaussian perturbation created at ({x_c}, {y_c})")


    def step(self):
        """Perform one timestep of the simulation with tracking of emergent structures (GPU accelerated)."""

        # 1. Compute Laplacian using CuPyX's convolve2d
        # Ensure kernel is cp.ndarray if cusignal expects it
        laplacian_phi = cusignal.convolve2d(self.phi, self.laplacian_kernel, mode='same', boundary='wrap')

        # 2. Compute tension map => local wave speed (all CuPy operations)
        tension_map = cp.abs(self.phi)**2
        local_speed = self.base_wave_speed_sq / (1.0 + self.tension_wave_factor * tension_map + 1e-6)
        self.local_wave_speed_sq = local_speed # Store for visualization

        # 3. Collision term (all CuPy operations)
        collision_effect = cp.zeros_like(self.phi)
        # Create boolean mask on GPU
        active_regions = tension_map > (self.collision_threshold**2)
        # Apply effect only to active regions using the mask
        collision_effect[active_regions] = self.collision_strength * self.phi[active_regions]**3

        # 4. Update (all CuPy operations)
        velocity = self.phi - self.phi_old
        phi_new = (
            self.phi
            + (1 - self.damping * self.dt) * velocity
            + self.dt**2 * (local_speed * laplacian_phi + collision_effect)
        )

        # Add noise if level is > 0 (GPU random numbers)
        if self.noise_level > 0:
             phi_new += cp.sqrt(self.dt) * self.noise_level * cp.random.randn(self.grid_size, self.grid_size, dtype=cp.float32)


        # 5. Compute new velocity for energy clamp (all CuPy operations)
        velocity_new = phi_new - self.phi
        energy_phi = cp.sum(phi_new**2)
        energy_vel = cp.sum(velocity_new**2)
        total_energy = energy_phi + energy_vel

        # 6. Global energy clamp (conditional scaling on GPU)
        # Use cp.sqrt and scalar division/multiplication which are efficient on GPU
        if total_energy > self.max_energy:
            # Calculate scale factor on GPU (scalar result is transferred implicitly if needed)
            scale = cp.sqrt(self.max_energy / total_energy)
            phi_new *= scale
            # Recompute velocity_new if clamping happened, or scale existing one
            # velocity_new *= scale # Simpler if velocity_new is already computed

        # 7. Track particle formation (CPU part - potentially slow for large grids/many particles)
        # Transfer necessary data (phi_new) from GPU to CPU for this step
        self.track_particles(phi_new) # Pass CuPy array

        # finalize update on GPU
        self.phi_old = self.phi
        self.phi = phi_new
        self.step_count += 1

    def track_particles(self, phi_new_gpu):
        """Track emergence of particle-like structures. Operates mostly on CPU after data transfer."""
        # Get the field data from GPU to CPU RAM
        phi_new_cpu = cp.asnumpy(phi_new_gpu) # Efficient transfer

        # Find local maxima above threshold as potential particles (CPU loop)
        # This part remains largely the same, but operates on phi_new_cpu
        threshold = 0.3 # CPU comparison
        maxima = [] # Store results on CPU
        # Iterate using CPU loops - might be a bottleneck for huge grids
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                center_val = phi_new_cpu[i, j]
                center_abs = abs(center_val) # CPU abs

                if center_abs > threshold:
                    is_max = True
                    # Check 8 neighbors
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            if center_abs < abs(phi_new_cpu[i + di, j + dj]): # CPU comparison
                                is_max = False
                                break
                        if not is_max:
                            break
                    if is_max:
                        maxima.append((j, i, center_val)) # Store (x, y, value) - CPU list

        # Check for collapse transition
        # Calculate max amplitude on GPU, get scalar result
        max_amplitude = cp.max(cp.abs(phi_new_gpu)).item() # .item() gets scalar value

        if max_amplitude > 0.7 and not self.has_collapsed:
            self.has_collapsed = True
            print(f"Field collapsed at step {self.step_count} (Max Amp: {max_amplitude:.3f})")

        # Update particle centers (stored on CPU)
        self.particle_centers = maxima

    def reset(self, grid_size=None):
        """Reset the field, step count, and re-initialize fields potentially with a new size."""
        if grid_size is not None:
            if isinstance(grid_size, int) and grid_size > 0:
                print(f"Resetting simulator with new grid size: {grid_size}")
                self.grid_size = grid_size
            else:
                print(f"Invalid grid_size provided to reset: {grid_size}. Keeping old size: {self.grid_size}")

        # Re-initialize fields on GPU with current grid_size
        self._initialize_fields()

        self.step_count = 0
        self.has_collapsed = False
        self.particle_centers = [] # Clear CPU list
        print(f"Simulation field reset.")


class TADS3DGUI:
    """
    Enhanced GUI with 3D visualization of TADS field, using CuPy backend.
    Includes grid size control.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("TADS 3D Simulator (CuPy Accelerated)")

        # Initial Grid Size
        self.current_grid_size = 100 # Default

        # Create the simulator
        self.simulator = TADSSimulator(grid_size=self.current_grid_size)

        # We'll run it in a separate thread
        self.running = False
        self.update_interval = 30 # ms - Faster updates might be possible with GPU

        # Whether we're in collapse mode or structure mode
        self.collapse_mode = True

        # We will share the same set of parameter sliders for both
        self.slider_labels = {}

        # Setup GUI components
        self.setup_gui()
        # Create initial plot meshgrid
        self._create_plot_meshgrid()
        # Initial plot update
        self.update_plot()


    def _create_plot_meshgrid(self):
         """Creates the X, Y meshgrid for plotting based on the current grid size."""
         print(f"Creating plot meshgrid for size: {self.simulator.grid_size}")
         # Use NumPy for meshgrid as Matplotlib works best with NumPy arrays
         # If performance becomes an issue, could create on GPU then .get()
         x_np = np.arange(0, self.simulator.grid_size)
         y_np = np.arange(0, self.simulator.grid_size)
         self.X_cpu, self.Y_cpu = np.meshgrid(x_np, y_np)


    def setup_gui(self):
        # Create a main frame that fills the window
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left side for visualizations
        vis_frame = tk.Frame(main_frame)
        vis_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side for controls
        controls_frame = tk.Frame(main_frame, width=250) # Give controls fixed width
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        controls_frame.pack_propagate(False) # Prevent resizing

        # --- Visualization Setup ---
        plot_frame = tk.Frame(vis_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = plt.Figure(figsize=(10, 8))
        self.gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 2])

        # 2D field visualization
        self.ax_field = self.fig.add_subplot(self.gs[0, 0])
        # Initial dummy data for imshow - use numpy for placeholder
        dummy_field = np.zeros((self.simulator.grid_size, self.simulator.grid_size))
        self.im_field = self.ax_field.imshow(dummy_field, cmap='viridis',
                                             vmin=-1.0, vmax=1.0, interpolation='nearest')
        self.ax_field.set_title("TADS Field")
        self.fig.colorbar(self.im_field, ax=self.ax_field, fraction=0.046, pad=0.04)

        # 2D speed visualization
        self.ax_speed = self.fig.add_subplot(self.gs[0, 1])
        dummy_speed = np.full_like(dummy_field, self.simulator.base_wave_speed_sq)
        self.im_speed = self.ax_speed.imshow(dummy_speed, cmap='magma',
                                             vmin=0, vmax=self.simulator.base_wave_speed_sq,
                                             interpolation='nearest')
        self.ax_speed.set_title("Wave Speed²")
        self.fig.colorbar(self.im_speed, ax=self.ax_speed, fraction=0.046, pad=0.04)

        # 3D surface plot
        self.ax_3d = self.fig.add_subplot(self.gs[1, :], projection='3d')
        # Placeholder surface - needs X_cpu, Y_cpu created first
        # self._create_plot_meshgrid() # Create initial meshgrid before plotting
        # self.surf = self.ax_3d.plot_surface(self.X_cpu, self.Y_cpu, dummy_field, cmap=cm.coolwarm,
        #                                    linewidth=0, antialiased=False)
        self.ax_3d.set_title("TADS Field Surface")
        self.ax_3d.set_zlim(-2.0, 2.0)

        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Status bar
        status_frame = tk.Frame(vis_frame)
        status_frame.pack(fill=tk.X, pady=(0, 5), padx=10)
        self.status_var = tk.StringVar(value="Ready. Steps: 0")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X)

        # --- Controls Setup ---
        control_label = ttk.Label(controls_frame, text="Simulation Controls", font=("Arial", 12, "bold"))
        control_label.pack(pady=10)

        # --- Grid Size Control ---
        grid_size_frame = ttk.Frame(controls_frame)
        grid_size_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(grid_size_frame, text="Grid Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.grid_size_var = tk.StringVar(value=str(self.current_grid_size))
        grid_size_entry = ttk.Entry(grid_size_frame, textvariable=self.grid_size_var, width=6)
        grid_size_entry.pack(side=tk.LEFT, padx=5)
        apply_grid_button = ttk.Button(grid_size_frame, text="Apply", command=self.apply_grid_size, width=6)
        apply_grid_button.pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.start_stop_button = ttk.Button(button_frame, text="Start", command=self.toggle_simulation)
        self.start_stop_button.pack(fill=tk.X, pady=3)

        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset)
        self.reset_button.pack(fill=tk.X, pady=3)

        self.mode_button = ttk.Button(button_frame, text="Phase: Collapse", command=self.toggle_mode)
        self.mode_button.pack(fill=tk.X, pady=3)

        self.create_perturbation_button = ttk.Button(button_frame, text="Create Perturbation",
                                                      command=self.create_perturbation)
        self.create_perturbation_button.pack(fill=tk.X, pady=3)

        # Parameters label
        param_label = ttk.Label(controls_frame, text="Parameters", font=("Arial", 12, "bold"))
        param_label.pack(pady=(20, 10))

        # Parameter sliders frame
        slider_frame = ttk.Frame(controls_frame)
        slider_frame.pack(fill=tk.X, padx=10)

        # --- Parameter Sliders ---
        # Collision Strength
        self._add_slider(slider_frame, "Collision Str:", 'coll_str', 0.0, 1.0, self.simulator.collision_strength, 3, 'collision_strength')
        # Tension Factor
        self._add_slider(slider_frame, "Tension Factor:", 'tension', 0.0, 100.0, self.simulator.tension_wave_factor, 3, 'tension_wave_factor')
        # Collision Threshold
        self._add_slider(slider_frame, "Collision Thresh:", 'coll_thresh', 0.1, 1.5, self.simulator.collision_threshold, 3, 'collision_threshold')
        # Base Speed^2
        self._add_slider(slider_frame, "Base Speed²:", 'base_speed', 0.01, 5.0, self.simulator.base_wave_speed_sq, 3, 'base_wave_speed_sq')
        # Max Energy
        self._add_slider(slider_frame, "Max Energy:", 'max_energy', 10.0, 1000.0, self.simulator.max_energy, 1, 'max_energy')

        # Add info text
        info_frame = ttk.Frame(controls_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=(30, 10))
        info_text = "Red/Blue markers show pos/neg field maxima (particles)."
        info_label = ttk.Label(info_frame, text=info_text, wraplength=200, justify=tk.LEFT)
        info_label.pack(fill=tk.X)


    def _add_slider(self, parent, label_text, label_key, from_, to_, initial_value, precision, sim_attribute_name):
        """Helper to add a parameter slider row."""
        slider_row = ttk.Frame(parent)
        slider_row.pack(fill=tk.X, pady=(8, 2)) # Add more vertical space
        ttk.Label(slider_row, text=label_text, width=15, anchor="w").pack(side=tk.LEFT) # Increased width and anchored left
        lbl_widget = ttk.Label(slider_row, text=f"{initial_value:.{precision}f}", width=7, anchor="e") # Anchored right
        lbl_widget.pack(side=tk.RIGHT, padx=(5,0))
        self.slider_labels[label_key] = lbl_widget

        slider_row_scale = ttk.Frame(parent)
        slider_row_scale.pack(fill=tk.X)
        var = tk.DoubleVar(value=initial_value)
        scale = ttk.Scale(slider_row_scale, variable=var, from_=from_, to=to_, orient='horizontal',
                          command=lambda val, attr=sim_attribute_name, v=var: self.update_single_param(val, attr, v))
        scale.pack(fill=tk.X, pady=(0, 5))
        # Use trace_add to link variable changes to label updates
        var.trace_add("write", lambda *_, v=var, lbl=lbl_widget, p=precision: self._update_slider_label(v, lbl, p))

        # Store the variable itself if needed later, e.g., to reset its value
        setattr(self, f"{label_key}_var", var)


    def _update_slider_label(self, var, label_widget, precision):
        """Update the text label associated with a slider."""
        try:
            label_widget.config(text=f"{var.get():.{precision}f}")
        except tk.TclError: # Handle case where widget might be destroyed during closing
             pass


    def update_single_param(self, value_str, sim_attribute_name, tk_var):
        """Update a single simulator parameter when its slider changes."""
        try:
            value = float(value_str)
            setattr(self.simulator, sim_attribute_name, value)
            # Optional: print(f"Set {sim_attribute_name} to {value:.3f}")
        except ValueError:
            print(f"Warning: Invalid value for {sim_attribute_name}: {value_str}")


    def update_plot(self):
        # Get data from GPU to CPU for plotting
        try:
            field_cpu = self.simulator.phi.get()
            speed_cpu = self.simulator.local_wave_speed_sq.get()
            max_speed = self.simulator.base_wave_speed_sq # This is scalar, no .get() needed
            max_amp = cp.max(cp.abs(self.simulator.phi)).item() # Calc on GPU, get scalar item
        except Exception as e:
             print(f"Error getting data from GPU: {e}")
             # Optionally, handle the error, e.g., by skipping the update
             # Or try using cp.asnumpy()
             try:
                 field_cpu = cp.asnumpy(self.simulator.phi)
                 speed_cpu = cp.asnumpy(self.simulator.local_wave_speed_sq)
                 max_speed = self.simulator.base_wave_speed_sq
                 max_amp = cp.asnumpy(cp.max(cp.abs(self.simulator.phi)))
             except Exception as e_inner:
                 print(f"Fallback cp.asnumpy also failed: {e_inner}")
                 return # Cannot update plot if data transfer fails


        # Determine plot limits dynamically or use fixed ones
        field_lim = max(1.0, max_amp * 1.1) # Adjust ylim based on max amplitude but at least +/- 1.0
        speed_lim = max(0.1, max_speed)    # Ensure vmax is not zero


        # --- Update 2D Field Plot ---
        self.ax_field.clear() # Clear previous drawings including markers
        self.im_field = self.ax_field.imshow(field_cpu, cmap='viridis', vmin=-field_lim, vmax=field_lim, interpolation='nearest')
        self.ax_field.set_title("TADS Field")
        self.ax_field.set_xticks([]) # Hide axes ticks
        self.ax_field.set_yticks([])

        # Add particle markers (using CPU coordinates from self.simulator.particle_centers)
        if self.simulator.particle_centers:
            xs, ys, vals = zip(*self.simulator.particle_centers)
            colors = ['r' if v > 0 else 'b' for v in vals]
            # Scale size maybe? s= [abs(v)*20 for v in vals] # Example
            self.ax_field.scatter(xs, ys, c=colors, marker='o', s=50, alpha=0.7, edgecolors='w', linewidths=0.5) # Use scatter for better control


        # --- Update 2D Speed Plot ---
        self.ax_speed.clear()
        self.im_speed = self.ax_speed.imshow(speed_cpu, cmap='magma', vmin=0, vmax=speed_lim, interpolation='nearest')
        self.ax_speed.set_title("Wave Speed²")
        self.ax_speed.set_xticks([]) # Hide axes ticks
        self.ax_speed.set_yticks([])

        # --- Update 3D Surface Plot ---
        # Important: Ensure self.X_cpu and self.Y_cpu match dimensions of field_cpu
        if self.X_cpu.shape == field_cpu.shape and self.Y_cpu.shape == field_cpu.shape:
             self.ax_3d.clear()
             self.surf = self.ax_3d.plot_surface(self.X_cpu, self.Y_cpu, field_cpu, cmap=cm.coolwarm,
                                                 linewidth=0, antialiased=False, rstride=5, cstride=5) # Adjust stride for performance
             self.ax_3d.set_zlim(-field_lim, field_lim) # Dynamic z-limit
             self.ax_3d.set_title("TADS Field Surface")
             self.ax_3d.set_xticks([]) # Hide axes ticks/labels for clarity
             self.ax_3d.set_yticks([])
             # self.ax_3d.set_zticks([]) # Optional: hide z-axis ticks too
             self.ax_3d.view_init(elev=30, azim=self.simulator.step_count % 360) # Rotate view slowly
        else:
            print(f"Plot update skipped: Mismatch between meshgrid shape {self.X_cpu.shape} and field shape {field_cpu.shape}")
            # This can happen if grid size changed but meshgrid wasn't updated yet


        # --- Update Status ---
        particles = len(self.simulator.particle_centers)
        status_text = f"Steps: {self.simulator.step_count}, Particles: {particles}, Max Amp: {max_amp:.3f}"
        if not IS_CUPY_AVAILABLE:
            status_text += " (Running on CPU)"
        self.status_var.set(status_text)


        # --- Redraw Canvas ---
        try:
            # self.fig.tight_layout(pad=2.0) # Recalculate layout - can be slow
            self.canvas.draw_idle()
        except tk.TclError as e:
             # This can happen if the window is closed while drawing
             print(f"Tkinter error during canvas draw: {e}")
             pass
        except Exception as e:
             print(f"General error during canvas draw: {e}")
             pass

    def simulation_loop(self):
        last_update_time = time.perf_counter()
        target_interval = self.update_interval / 1000.0

        while self.running:
            t_start = time.perf_counter()

            # --- Run one simulation step ---
            self.simulator.step()

            # --- Mode-specific logic ---
            if self.simulator.has_collapsed and not self.collapse_mode:
                # If collapsed and we want structures, turn off collision *after* collapse
                if self.simulator.collision_strength != 0.0:
                    print("Structure phase: Setting collision strength to 0 post-collapse.")
                    self.simulator.collision_strength = 0.0
                    # Update the slider variable on the main thread
                    self.root.after(0, lambda: self.coll_str_var.set(0.0))


            t_sim_done = time.perf_counter()

            # --- Schedule plot update ---
            # Update plots periodically, not necessarily every single step
            current_time = time.perf_counter()
            if current_time - last_update_time >= target_interval:
                 # Schedule the plot update to run in the main Tkinter thread
                 self.root.after(0, self.update_plot)
                 last_update_time = current_time


            # --- Control the simulation speed / sleep ---
            t_end = time.perf_counter()
            elapsed = t_end - t_start
            # We don't necessarily need to sleep if sim step is fast enough
            # A small sleep prevents 100% CPU/GPU usage if steps are very quick
            sleep_time = max(0, (target_interval / 10.0) - elapsed) # Sleep fraction of update interval
            if sleep_time > 0.001: # Only sleep if it's meaningful
                 time.sleep(sleep_time)


        print("Simulation loop finished.")


    def toggle_simulation(self):
        if not self.running:
            if not IS_CUPY_AVAILABLE:
                 messagebox.showwarning("Performance Warning", "CuPy not found. Simulation will run on the CPU and might be slow.")
            self.running = True
            self.start_stop_button.config(text="Stop")
            # Start the simulation thread
            self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
            self.sim_thread.start()
            print("Simulation started.")
        else:
            self.running = False
            self.start_stop_button.config(text="Start")
            print("Simulation stopping...")
            # Thread will exit naturally because self.running is False
            # Optional: Add join with timeout if needed, but daemon=True should handle it.
            # if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
            #     self.sim_thread.join(timeout=0.5) # Wait briefly for thread to finish


    def apply_grid_size(self):
        """Applies the grid size entered in the Entry widget."""
        was_running = self.running
        if self.running:
            self.toggle_simulation() # Stop simulation
            time.sleep(self.update_interval / 1000.0 * 2) # Wait a bit

        try:
            new_size_str = self.grid_size_var.get()
            new_size = int(new_size_str)
            if new_size < 10 or new_size > 1024: # Add reasonable limits
                messagebox.showerror("Invalid Size", "Grid size must be an integer between 10 and 1024.")
                # Restore previous value if needed
                self.grid_size_var.set(str(self.current_grid_size))
                return
            if new_size != self.current_grid_size:
                 print(f"Applying new grid size: {new_size}")
                 self.current_grid_size = new_size
                 # Reset the simulator with the new size
                 self.simulator.reset(grid_size=new_size)
                 # Recreate the plot meshgrid for the new size
                 self._create_plot_meshgrid()
                 # Update the plots immediately to reflect the reset state and new size
                 self.update_plot()
                 # Clear collapse state etc.
                 self.reset_simulation_state()
            else:
                 print("Grid size unchanged.")

        except ValueError:
            messagebox.showerror("Invalid Input", "Grid size must be an integer.")
            self.grid_size_var.set(str(self.current_grid_size)) # Restore previous valid value

        if was_running:
            self.toggle_simulation() # Restart simulation if it was running


    def reset_simulation_state(self):
        """Resets UI elements and flags related to simulation state (like collapse mode)."""
        self.collapse_mode = True
        self.mode_button.config(text="Phase: Collapse")
        # Reset collision strength to default only if it's structure mode related reset?
        # Let's reset it to the slider's current value, assuming user might have adjusted it
        default_coll_str = self.coll_str_var.get() # Get current slider value
        # Or always reset to a fixed default: default_coll_str = 0.2
        self.simulator.collision_strength = default_coll_str
        self.coll_str_var.set(default_coll_str) # Ensure slider matches


    def reset(self):
        """Resets the simulation to its initial state with current parameters and grid size."""
        was_running = self.running
        if self.running:
            self.toggle_simulation() # Stop simulation
            # Wait a bit longer to ensure thread stops, especially if GPU tasks are involved
            time.sleep(self.update_interval / 1000.0 * 3)


        print("Resetting simulation...")
        # Reset the simulator using its current grid size
        self.simulator.reset() # grid_size=None uses the existing simulator.grid_size
        # Recreate meshgrid (might be redundant if size didn't change, but safe)
        self._create_plot_meshgrid()
         # Reset UI state like collapse mode and button text
        self.reset_simulation_state()
        # Update the plots to show the reset state
        self.update_plot()
        print("Simulation reset complete.")

        if was_running:
             # Brief pause before restarting
             time.sleep(0.1)
             self.toggle_simulation() # Restart simulation if it was running


    def toggle_mode(self):
        """Toggle between collapse mode and structure formation mode"""
        self.collapse_mode = not self.collapse_mode
        if self.collapse_mode:
            self.mode_button.config(text="Phase: Collapse")
            # Set collision strength to drive collapse - use slider value or a default?
            # Let's restore the value currently shown on the slider
            current_slider_coll_str = self.coll_str_var.get()
            self.simulator.collision_strength = current_slider_coll_str
            print(f"Mode: Collapse. Collision Strength set to slider value: {current_slider_coll_str:.3f}")
            # Optionally, force a minimum value for collapse:
            # if current_slider_coll_str < 0.1: # Example threshold
            #    self.coll_str_var.set(0.2)
            #    self.simulator.collision_strength = 0.2
            #    print("Mode: Collapse. Collision Strength forced to 0.2")

        else: # Structure mode
            self.mode_button.config(text="Phase: Structure")
            # If already collapsed, set collision strength to 0 immediately
            if self.simulator.has_collapsed:
                print("Mode: Structure (post-collapse). Setting Collision Strength to 0.0")
                self.simulator.collision_strength = 0.0
                self.coll_str_var.set(0.0) # Update slider too
            else:
                # If not yet collapsed, leave collision strength as is (it might be needed to *reach* collapse first)
                 print("Mode: Structure (pre-collapse). Collision Strength unchanged.")


    def create_perturbation(self):
        """Create a Gaussian perturbation in the field at a random location"""
        # Choose a random location (on CPU, indices are small)
        margin = max(10, self.simulator.grid_size // 10) # Keep away from edges
        try:
            x = np.random.randint(margin, self.simulator.grid_size - margin)
            y = np.random.randint(margin, self.simulator.grid_size - margin)
        except ValueError: # Handle case where grid_size is too small for margin
            x = self.simulator.grid_size // 2
            y = self.simulator.grid_size // 2

        # Call the simulator's method (which operates on GPU array)
        self.simulator.create_gaussian_perturbation(location=(x, y), radius=5, strength=1.5) # Increased strength

        # Update the plot immediately to show the perturbation
        self.update_plot()

    def on_closing(self):
        """Handles window closing event."""
        print("Closing application...")
        self.running = False # Signal simulation thread to stop
        try:
            if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
                print("Waiting for simulation thread to finish...")
                self.sim_thread.join(timeout=1.0) # Wait max 1 second for thread
        except Exception as e:
            print(f"Error joining simulation thread: {e}")

        plt.close(self.fig) # Close the matplotlib figure
        self.root.destroy() # Destroy the Tkinter window
        print("Application closed.")

if __name__ == "__main__":
    root = tk.Tk()
    # Apply a theme for better visuals if available (e.g., 'clam', 'alt', 'default', 'classic')
    try:
        style = ttk.Style(root)
        # print(style.theme_names()) # See available themes
        style.theme_use('clam') # 'clam' often looks good
    except tk.TclError:
        print("Failed to set ttk theme.")

    app = TADS3DGUI(root)
    root.geometry("1300x850")  # Set a larger default window size
    root.protocol("WM_DELETE_WINDOW", app.on_closing) # Ensure clean shutdown
    root.mainloop()