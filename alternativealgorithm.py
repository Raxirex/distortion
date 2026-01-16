import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import math

class DistortEngine:
    
    @staticmethod
    def create_identity(nx, ny):
        """Creates a normalized 0.0-1.0 coordinate grid."""
        x = np.linspace(0, 1, nx, dtype=np.float32)
        y = np.linspace(0, 1, ny, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        return np.stack([xv, yv], axis=-1)

    @staticmethod
    def apply_brush(mesh, cx, cy, radius, strength, tool="Push", dx=0, dy=0):
        """Vectorized brush application using NumPy broadcasting."""
        # Extract coordinate channels
        U = mesh[:, :, 0]
        V = mesh[:, :, 1]
        
        # Calculate distance matrix from cursor
        diff_x = U - cx
        diff_y = V - cy
        dist_sq = diff_x**2 + diff_y**2
        rad_sq = radius**2
        
        # Create boolean mask for points inside the brush
        mask = dist_sq < rad_sq
        if not np.any(mask): 
            return mesh

        # Weighting: 1.0 at center, 0.0 at edge
        dist = np.sqrt(dist_sq[mask])
        w = (1.0 - (dist / radius)) * strength

        if tool == "Push":
            U[mask] += dx * w
            V[mask] += dy * w
            
        elif tool == "Vortex":
            angle = w * 4.0  # Rotation strength
            s, c = np.sin(angle), np.cos(angle)
            # Relative rotation
            rx = diff_x[mask] * c - diff_y[mask] * s
            ry = diff_x[mask] * s + diff_y[mask] * c
            U[mask] = cx + rx
            V[mask] = cy + ry
            
        elif tool == "Ripple":
            freq = 40.0 # Wave frequency
            shift = np.sin(dist * freq) * w * 0.03
            norm = np.maximum(dist, 1e-6)
            U[mask] += (diff_x[mask] / norm) * shift
            V[mask] += (diff_y[mask] / norm) * shift
            
        elif tool == "Bloat":
            # Radial expansion
            U[mask] += (diff_x[mask] / radius) * w * 0.1
            V[mask] += (diff_y[mask] / radius) * w * 0.1
            
        elif tool == "Pinch":
            # Radial contraction
            U[mask] -= (diff_x[mask] / radius) * w * 0.1
            V[mask] -= (diff_y[mask] / radius) * w * 0.1
            
        return mesh

class MeshEditor(tk.Toplevel):
    """The visual editing window with interactive canvas."""
    def __init__(self, parent, mesh_res=1000):
        super().__init__(parent)
        self.title(f"High-Res Mesh Editor ({mesh_res}x{mesh_res})")
        self.geometry("1200x850")
        self.app = parent
        
        # Resolution State
        self.res = mesh_res
        self.mesh = DistortEngine.create_identity(self.res, self.res)
        
        # Interaction State
        self.tool = tk.StringVar(value="Push")
        self.brush_size = tk.DoubleVar(value=0.15)
        self.strength = tk.DoubleVar(value=0.4)
        self.last_mouse = None
        
        self._setup_ui()
        self.bind("<Configure>", lambda e: self.draw_mesh())

    def _setup_ui(self):
        # Control Panel
        sidebar = ttk.Frame(self, padding=15)
        sidebar.pack(side="left", fill="y")
        
        ttk.Label(sidebar, text="BRUSH SETTINGS", font=('Helvetica', 10, 'bold')).pack(pady=(0, 10))
        
        ttk.Label(sidebar, text="Tool Type:").pack(anchor="w")
        tools = ["Push", "Vortex", "Ripple", "Bloat", "Pinch"]
        ttk.Combobox(sidebar, textvariable=self.tool, values=tools, state="readonly").pack(fill="x", pady=5)
        
        ttk.Label(sidebar, text="Brush Size:").pack(anchor="w", pady=(10, 0))
        ttk.Scale(sidebar, from_=0.01, to=0.6, variable=self.brush_size).pack(fill="x")
        
        ttk.Label(sidebar, text="Strength:").pack(anchor="w", pady=(10, 0))
        ttk.Scale(sidebar, from_=0.01, to=1.0, variable=self.strength).pack(fill="x")
        
        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", pady=20)
        
        ttk.Button(sidebar, text="Reset Grid", command=self.reset_mesh).pack(fill="x", pady=5)
        ttk.Button(sidebar, text="Apply Warp", command=self.send_to_app).pack(fill="x", pady=5)
        
        ttk.Label(sidebar, text="Tip: Click & Drag on grid", foreground="gray").pack(side="bottom")

        # Interaction Canvas
        self.canvas = tk.Canvas(self, bg="#121212", highlightthickness=0)
        self.canvas.pack(side="right", fill="both", expand=True)
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: (setattr(self, 'last_mouse', None), self.send_to_app()))

    def reset_mesh(self):
        self.mesh = DistortEngine.create_identity(self.res, self.res)
        self.draw_mesh()
        self.send_to_app()

    def on_click(self, event):
        self.last_mouse = (event.x, event.y)

    def on_drag(self, event):
        if self.last_mouse is None: return
        
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        # Normalize current and last mouse positions (0 to 1)
        nx, ny = event.x / cw, event.y / ch
        lx, ly = self.last_mouse[0] / cw, self.last_mouse[1] / ch
        
        # Calculate normalized movement vector
        dx, dy = nx - lx, ny - ly
        
        # Apply the high-speed vectorized math
        self.mesh = DistortEngine.apply_brush(
            self.mesh, nx, ny, 
            self.brush_size.get(), 
            self.strength.get(),
            tool=self.tool.get(),
            dx=dx, dy=dy
        )
        
        self.last_mouse = (event.x, event.y)
        self.draw_mesh()

    def draw_mesh(self):
        """Draws a visually responsive proxy grid while keeping the 1000x1000 math."""
        self.canvas.delete("grid")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        # Visualization step: Only draw every Nth point to prevent Tkinter hanging
        # 25-30 lines is enough for visual feedback
        v_step = self.res // 30
        
        # Drawing Vertical Grid Lines
        for i in range(0, self.res, v_step):
            line_pts = []
            for j in range(0, self.res, v_step):
                px = self.mesh[j, i, 0] * cw
                py = self.mesh[j, i, 1] * ch
                line_pts.extend([px, py])
            self.canvas.create_line(line_pts, fill="#33ffcc", width=1, tags="grid", capstyle="round", smooth=True)

        # Drawing Horizontal Grid Lines
        for j in range(0, self.res, v_step):
            line_pts = []
            for i in range(0, self.res, v_step):
                px = self.mesh[j, i, 0] * cw
                py = self.mesh[j, i, 1] * ch
                line_pts.extend([px, py])
            self.canvas.create_line(line_pts, fill="#33ffcc", width=1, tags="grid", capstyle="round", smooth=True)

    def send_to_app(self):
        self.app.update_warp(self.mesh)

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vectorized Warp Engine Pro")
        self.geometry("1000x700")
        
        self.cv_img = None
        self.mesh = None
        
        self._setup_ui()

    def _setup_ui(self):
        top_bar = ttk.Frame(self, padding=10)
        top_bar.pack(side="top", fill="x")
        
        ttk.Button(top_bar, text="Step 1: Load Image", command=self.load_image).pack(side="left", padx=5)
        ttk.Button(top_bar, text="Step 2: Edit Warp Grid", command=self.open_editor).pack(side="left", padx=5)
        ttk.Button(top_bar, text="Step 3: Save JPG", command=self.save_image).pack(side="right", padx=5)
        
        # Image Display Area
        self.img_display = ttk.Label(self, text="Please Load an Image to Begin")
        self.img_display.pack(expand=True, fill="both")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.webp")])
        if path:
            # Use OpenCV to read image
            self.cv_img = cv2.imread(path)
            if self.cv_img is None:
                messagebox.showerror("Error", "Could not read image file.")
                return
            self.update_warp()

    def open_editor(self):
        if self.cv_img is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return
        MeshEditor(self)

    def update_warp(self, mesh=None):
        if self.cv_img is None: return
        
        self.mesh = mesh
        h, w = self.cv_img.shape[:2]
        
        if mesh is not None:
            # BILINEAR RESIZE OPTIMIZATION
            # Convert the 1000x1000 mesh into a full-resolution pixel map
            # This is significantly faster than calculating individual triangles.
            map_x = cv2.resize(mesh[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR) * (w - 1)
            map_y = cv2.resize(mesh[:, :, 1], (w, h), interpolation=cv2.INTER_LINEAR) * (h - 1)
            
            # Execute Warp
            warped = cv2.remap(self.cv_img, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
        else:
            warped = self.cv_img

        self.last_result = warped
        self.show_image(warped)

    def show_image(self, cv_img):
        # Resize for display (keep aspect ratio)
        h, w = cv_img.shape[:2]
        display_h, display_w = 600, 800
        scale = min(display_w/w, display_h/h)
        
        small = cv2.resize(cv_img, (int(w*scale), int(h*scale)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.img_display.config(image=img_tk, text="")
        self.img_display.image = img_tk

    def save_image(self):
        if not hasattr(self, 'last_result'): return
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if save_path:
            cv2.imwrite(save_path, self.last_result)
            messagebox.showinfo("Success", f"Saved to {save_path}")

if __name__ == "__main__":
    # Performance Optimization: Ensure NumPy uses all available threads
    import os
    os.environ["OMP_NUM_THREADS"] = "4" 
    
    app = MainApp()
    app.mainloop()
