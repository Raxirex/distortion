import os
import sys
import math
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2

# Preview (recommended)
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def script_dir() -> Path:
    return Path(os.path.abspath(sys.argv[0])).parent


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def build_camera_matrix(w: int, h: int, focal_scale: float, focus_x: float = 0.0, focus_y: float = 0.0) -> np.ndarray:
    """
    Camera matrix with adjustable principal point.

    focus_x/focus_y are fractions of (w,h) added to the image center:
      cx = (w-1)/2 + focus_x*w
      cy = (h-1)/2 + focus_y*h
    """
    f = float(max(w, h)) * float(focal_scale)
    cx = (w - 1) * 0.5 + float(focus_x) * float(w)
    cy = (h - 1) * 0.5 + float(focus_y) * float(h)
    return np.array([[f, 0.0, cx],
                     [0.0, f, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def strength_to_coeffs(strength: float) -> np.ndarray:
    # strength in [-1, 1] : + = pincushion, - = barrel
    s = float(strength)
    sign = 1.0 if s >= 0 else -1.0
    # Tuned to feel good as a single knob
    k1 = 0.55 * s
    k2 = 0.25 * (abs(s) ** 2) * sign
    k3 = 0.10 * (s ** 3)
    p1 = 0.0
    p2 = 0.0
    return np.array([k1, k2, p1, p2, k3], dtype=np.float64)


def make_distort_maps_blocked(
    w_out: int,
    h_out: int,
    K_dist: np.ndarray,
    dist: np.ndarray,
    K_src: np.ndarray,
    pan_x_px: float = 0.0,
    pan_y_px: float = 0.0,
    block_rows: int = 128
):
    """
    Build maps for cv2.remap to APPLY lens distortion.
    destination (distorted) -> source (base) mapping.

    pts_dist are output pixel coords interpreted using (K_dist, dist).
    undistortPoints returns the corresponding undistorted pixel coords under P=K_src,
    which are used as sampling coordinates into the source image.
    """
    map_x = np.empty((h_out, w_out), dtype=np.float32)
    map_y = np.empty((h_out, w_out), dtype=np.float32)

    xs = np.arange(w_out, dtype=np.float32)

    for y0 in range(0, h_out, block_rows):
        y1 = min(h_out, y0 + block_rows)
        ys = np.arange(y0, y1, dtype=np.float32)

        X, Y = np.meshgrid(xs, ys)
        pts_dist = np.stack([X, Y], axis=-1).reshape(-1, 1, 2)  # Nx1x2 float32

        pts_ud = cv2.undistortPoints(pts_dist, K_dist, dist, R=None, P=K_src)  # Nx1x2 float64
        pts_ud = pts_ud.reshape((y1 - y0), w_out, 2).astype(np.float32)

        map_x[y0:y1, :] = pts_ud[:, :, 0] + float(pan_x_px)
        map_y[y0:y1, :] = pts_ud[:, :, 1] + float(pan_y_px)

    return map_x, map_y


def _inside_triangle_mask(X, Y, x0, y0, x1, y1, x2, y2):
    # sign-of-areas test (orientation independent)
    d1 = (X - x1) * (y0 - y1) - (x0 - x1) * (Y - y1)
    d2 = (X - x2) * (y1 - y2) - (x1 - x2) * (Y - y2)
    d3 = (X - x0) * (y2 - y0) - (x2 - x0) * (Y - y0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def make_mesh_maps_triangular(
    w_out: int,
    h_out: int,
    w_src: int,
    h_src: int,
    dest_norm: np.ndarray,
):
    """
    Custom mesh warp (dest->src) using a draggable grid of knots.

    - Source knots are a regular grid across the source image.
    - Destination knots are user-editable (dest_norm).

    We build dest->src maps by rasterizing 2 triangles per quad and using affine transforms.
    """
    ny, nx, _ = dest_norm.shape

    xs = np.linspace(0.0, float(w_src - 1), nx, dtype=np.float32)
    ys = np.linspace(0.0, float(h_src - 1), ny, dtype=np.float32)
    Sx, Sy = np.meshgrid(xs, ys)  # (ny,nx)

    Dx = (dest_norm[:, :, 0].astype(np.float32)) * float(w_out - 1)
    Dy = (dest_norm[:, :, 1].astype(np.float32)) * float(h_out - 1)

    map_x = np.full((h_out, w_out), -1.0, dtype=np.float32)
    map_y = np.full((h_out, w_out), -1.0, dtype=np.float32)

    for j in range(ny - 1):
        for i in range(nx - 1):
            # Two triangles per cell:
            # A: TL, TR, BL
            # B: BR, BL, TR
            tri_defs = [
                (i, j, i + 1, j, i, j + 1),
                (i + 1, j + 1, i, j + 1, i + 1, j),
            ]

            for (i0, j0, i1, j1, i2, j2) in tri_defs:
                xd0, yd0 = float(Dx[j0, i0]), float(Dy[j0, i0])
                xd1, yd1 = float(Dx[j1, i1]), float(Dy[j1, i1])
                xd2, yd2 = float(Dx[j2, i2]), float(Dy[j2, i2])

                xs0, ys0 = float(Sx[j0, i0]), float(Sy[j0, i0])
                xs1, ys1 = float(Sx[j1, i1]), float(Sy[j1, i1])
                xs2, ys2 = float(Sx[j2, i2]), float(Sy[j2, i2])

                minx = int(max(0, math.floor(min(xd0, xd1, xd2))))
                maxx = int(min(w_out - 1, math.ceil(max(xd0, xd1, xd2))))
                miny = int(max(0, math.floor(min(yd0, yd1, yd2))))
                maxy = int(min(h_out - 1, math.ceil(max(yd0, yd1, yd2))))
                if (maxx < minx) or (maxy < miny):
                    continue

                X, Y = np.meshgrid(
                    np.arange(minx, maxx + 1, dtype=np.float32),
                    np.arange(miny, maxy + 1, dtype=np.float32),
                )
                mask = _inside_triangle_mask(X, Y, xd0, yd0, xd1, yd1, xd2, yd2)
                if not np.any(mask):
                    continue

                M = cv2.getAffineTransform(
                    np.float32([[xd0, yd0], [xd1, yd1], [xd2, yd2]]),
                    np.float32([[xs0, ys0], [xs1, ys1], [xs2, ys2]]),
                )

                src_x = M[0, 0] * X + M[0, 1] * Y + M[0, 2]
                src_y = M[1, 0] * X + M[1, 1] * Y + M[1, 2]

                map_x[miny:maxy + 1, minx:maxx + 1][mask] = src_x[mask]
                map_y[miny:maxy + 1, minx:maxx + 1][mask] = src_y[mask]

    return map_x, map_y


BORDER_MAP = {
    "Black": cv2.BORDER_CONSTANT,
    "Reflect": cv2.BORDER_REFLECT,
    "Replicate": cv2.BORDER_REPLICATE,
}
INTERP_MAP = {
    "Linear (recommended)": cv2.INTER_LINEAR,
    "Cubic": cv2.INTER_CUBIC,
    "Nearest": cv2.INTER_NEAREST,
}


@dataclass
class DistortSettings:
    mode: str
    strength: float
    focal_scale: float

    # Framing
    output_scale: float
    focus_x: float
    focus_y: float
    pan_x: float
    pan_y: float

    # Pan behavior
    pan_stage: str

    # Coefficients
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float

    # Output
    overwrite: bool
    border: str
    interp: str

    # Preview overlay
    show_crosshair: bool

    # Mesh
    mesh_enabled: bool


class MeshEditor(tk.Toplevel):
    """
    Visual mesh editor:
    - Shows a large canvas with the CURRENT OUTPUT preview
    - Draws a draggable knot grid (destination space)
    - Dragging knots updates the mesh in the main App and triggers live preview

    The mesh is applied AFTER lens distortion (so your symmetric pincushion stays),
    then you can make it asymmetric / circular / crescent / wobbly by pulling knots.
    """
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Custom Distortion Mesh (Drag knots)")
        self.geometry("1024x768")
        self.configure(bg="#111")

        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        self.info = ttk.Label(
            top,
            text="Drag knots to reshape distortion. Mesh warp is applied AFTER lens distortion.",
            anchor="w"
        )
        self.info.pack(side="left", fill="x", expand=True)

        ttk.Button(top, text="Reset mesh (identity)", command=self._reset).pack(side="right")
        ttk.Button(top, text="Center all knots", command=self._center).pack(side="right", padx=6)

        row = ttk.Frame(self, padding=(8, 0, 8, 8))
        row.pack(fill="x")

        ttk.Label(row, text="Grid:").pack(side="left")
        self.nx_var = tk.IntVar(value=self.app.mesh_nx)
        self.ny_var = tk.IntVar(value=self.app.mesh_ny)
        nx_spin = ttk.Spinbox(row, from_=3, to=25, textvariable=self.nx_var, width=4, command=self._resize_grid)
        ny_spin = ttk.Spinbox(row, from_=3, to=25, textvariable=self.ny_var, width=4, command=self._resize_grid)
        nx_spin.pack(side="left", padx=(6, 0))
        ttk.Label(row, text="x").pack(side="left", padx=4)
        ny_spin.pack(side="left")

        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Show grid", variable=self.show_grid_var, command=self.redraw).pack(side="left", padx=14)

        
        # ---- Liquify / brush editing (new) ----
        toolbox = ttk.LabelFrame(self, text="Liquify (brush) tool options", padding=8)
        toolbox.pack(fill="x", padx=8, pady=(0, 8))

        # Mode selector: Knots vs Liquify
        mode_row = ttk.Frame(toolbox)
        mode_row.pack(fill="x")
        self.edit_mode_var = tk.StringVar(value="Knots")
        ttk.Label(mode_row, text="Edit mode:").pack(side="left")
        ttk.Radiobutton(mode_row, text="Knots", value="Knots", variable=self.edit_mode_var, command=self.redraw).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(mode_row, text="Liquify", value="Liquify", variable=self.edit_mode_var, command=self.redraw).pack(side="left", padx=(8, 0))

        # Liquify tool selector
        tool_row = ttk.Frame(toolbox)
        tool_row.pack(fill="x", pady=(6, 0))
        ttk.Label(tool_row, text="Tool:").pack(side="left")
        self.liq_tool_var = tk.StringVar(value="Push")
        ttk.Combobox(
            tool_row,
            textvariable=self.liq_tool_var,
            values=["Push", "Pull", "Bloat", "Pinch", "Twirl"],
            state="readonly",
            width=12,
        ).pack(side="left", padx=(8, 0))

        ttk.Label(tool_row, text="Shape:").pack(side="left", padx=(14, 0))
        self.brush_shape_var = tk.StringVar(value="Circle")
        ttk.Combobox(
            tool_row,
            textvariable=self.brush_shape_var,
            values=["Circle", "Rectangle", "Triangle"],
            state="readonly",
            width=12,
        ).pack(side="left", padx=(8, 0))

        self.pin_border_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tool_row, text="Pin border", variable=self.pin_border_var).pack(side="right")

        # Sliders
        sliders = ttk.Frame(toolbox)
        sliders.pack(fill="x", pady=(8, 0))

        # Size in pixels on the fitted preview in the MeshEditor
        self.brush_size_var = tk.DoubleVar(value=120.0)
        self.brush_pressure_var = tk.DoubleVar(value=35.0)
        self.brush_density_var = tk.DoubleVar(value=80.0)
        self.brush_rate_var = tk.DoubleVar(value=50.0)

        def _mk_slider(parent, label, var, frm, to, col):
            colf = ttk.Frame(parent)
            colf.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0))
            ttk.Label(colf, text=label).pack(anchor="w")
            s = ttk.Scale(colf, from_=frm, to=to, variable=var, command=lambda _=None: None)
            s.pack(fill="x")
            val = ttk.Label(colf, text=f"{var.get():.0f}")
            val.pack(anchor="e")
            var.trace_add("write", lambda *_: val.configure(text=f"{var.get():.0f}"))
            return s

        sliders.columnconfigure(0, weight=1)
        sliders.columnconfigure(1, weight=1)
        sliders.columnconfigure(2, weight=1)
        sliders.columnconfigure(3, weight=1)

        _mk_slider(sliders, "Size", self.brush_size_var, 10, 600, 0)
        _mk_slider(sliders, "Pressure", self.brush_pressure_var, 0, 100, 1)
        _mk_slider(sliders, "Density", self.brush_density_var, 0, 100, 2)
        _mk_slider(sliders, "Rate", self.brush_rate_var, 0, 100, 3)

        # Liquify hint
        ttk.Label(toolbox, text="Tip: In Liquify mode, drag to sculpt the mesh like a liquify filter.").pack(anchor="w", pady=(6, 0))

        ttk.Checkbutton(
            row,
            text="Enable mesh warp",
            variable=self.app.mesh_enabled_var,
            command=lambda: (self.app._schedule_preview(), self.redraw()),
        ).pack(side="right")

        self.canvas = tk.Canvas(self, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._dragging = None  # (j,i)
        self._imgtk = None
        self._fit_rect = None  # (x0,y0,w,h) fitted image rect in canvas


        self._liq_last = None  # (x,y) in canvas
        self._hover = None  # (x,y)

        self.canvas.bind("<Button-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        self.bind("<Configure>", lambda e: self.redraw())

        self.after(60, self.redraw)

        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Leave>", lambda e: self._set_hover(None))

    def _resize_grid(self):
        nx = int(self.nx_var.get())
        ny = int(self.ny_var.get())
        self.app.set_mesh_grid_size(nx, ny)
        self.app.mesh_enabled_var.set(True)
        self.app._schedule_preview()
        self.redraw()

    def _reset(self):
        self.app.reset_mesh_identity()
        self.app.mesh_enabled_var.set(True)
        self.app._schedule_preview()
        self.redraw()

    def _center(self):
        # Soft "re-center": moves knots toward their identity positions (helps recover)
        ident = self.app._make_identity_mesh(self.app.mesh_nx, self.app.mesh_ny)
        self.app.mesh_dest_norm[:] = 0.6 * self.app.mesh_dest_norm + 0.4 * ident
        self.app.mesh_enabled_var.set(True)
        self.app._schedule_preview()
        self.redraw()

    def _nearest_knot(self, x, y):
        if self._fit_rect is None:
            return None
        x0, y0, fw, fh = self._fit_rect
        if fw <= 2 or fh <= 2:
            return None

        dest = self.app.mesh_dest_norm
        ny, nx, _ = dest.shape

        best = None
        best_d2 = 1e18
        for j in range(ny):
            for i in range(nx):
                ku = float(dest[j, i, 0])
                kv = float(dest[j, i, 1])
                kx = x0 + ku * fw
                ky = y0 + kv * fh
                d2 = (kx - x) ** 2 + (ky - y) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = (j, i)
        if best is None:
            return None
        if best_d2 > (18.0 ** 2):
            return None
        return best

    def _set_knot_from_canvas(self, j, i, x, y):
        if self._fit_rect is None:
            return
        x0, y0, fw, fh = self._fit_rect
        u = (x - x0) / max(1.0, fw)
        v = (y - y0) / max(1.0, fh)

        # allow overshoot so you can form crescents / off-canvas pulls
        u = max(-0.50, min(1.50, u))
        v = max(-0.50, min(1.50, v))

        self.app.mesh_dest_norm[j, i, 0] = u
        self.app.mesh_dest_norm[j, i, 1] = v

    def _on_down(self, event):
        if str(self.edit_mode_var.get()) == "Liquify":
            # start brush stroke
            self._liq_last = (float(event.x), float(event.y))
            self.app.mesh_enabled_var.set(True)
            # Apply a tiny nudge at start (zero delta is okay)
            self._apply_liquify(event.x, event.y, 0.0, 0.0)
            self.app._schedule_preview()
            self.redraw()
            return

        knot = self._nearest_knot(event.x, event.y)
        if knot is None:
            return
        self._dragging = knot

    def _on_move(self, event):
        if str(self.edit_mode_var.get()) == "Liquify":
            if self._liq_last is None:
                self._liq_last = (float(event.x), float(event.y))
                return
            lx, ly = self._liq_last
            cx, cy = float(event.x), float(event.y)
            dx, dy = (cx - lx), (cy - ly)
            self._liq_last = (cx, cy)
            self._apply_liquify(cx, cy, dx, dy)
            self.app.mesh_enabled_var.set(True)
            self.app._schedule_preview()
            self.redraw()
            return

        if self._dragging is None:
            return
        j, i = self._dragging
        self._set_knot_from_canvas(j, i, event.x, event.y)
        self.app.mesh_enabled_var.set(True)
        self.app._schedule_preview()
        self.redraw()

    def _on_up(self, _event):
        self._dragging = None
        self._liq_last = None


    def _set_hover(self, xy):
        self._hover = xy
        # light redraw to show brush outline
        self.redraw()

    def _on_hover(self, event):
        self._set_hover((float(event.x), float(event.y)))

    def _canvas_to_uv(self, x, y):
        if self._fit_rect is None:
            return None
        x0, y0, fw, fh = self._fit_rect
        if fw <= 2 or fh <= 2:
            return None
        u = (x - x0) / fw
        v = (y - y0) / fh
        return u, v

    def _is_border_node(self, j, i, ny, nx):
        return (j == 0) or (i == 0) or (j == ny - 1) or (i == nx - 1)

    def _apply_liquify(self, x, y, dx, dy):
        # Applies a brush operation to the underlying mesh knots.
        if self._fit_rect is None:
            return
        x0, y0, fw, fh = self._fit_rect
        if fw <= 2 or fh <= 2:
            return

        # brush parameters
        size_px = float(self.brush_size_var.get())
        pressure = float(self.brush_pressure_var.get()) / 100.0
        density = float(self.brush_density_var.get()) / 100.0
        rate = float(self.brush_rate_var.get()) / 100.0

        if size_px <= 1 or pressure <= 0 or rate <= 0:
            return

        tool = str(self.liq_tool_var.get())
        shape = str(self.brush_shape_var.get())

        dest = self.app.mesh_dest_norm
        ny, nx, _ = dest.shape

        # Cursor in canvas px
        cx, cy = float(x), float(y)

        # Convert drag delta to normalized delta
        du = float(dx) / fw
        dv = float(dy) / fh

        # Triangle vertices (upright isosceles) for mask if needed
        tri = None
        if shape == "Triangle":
            tri = ((cx, cy - size_px), (cx - size_px, cy + size_px), (cx + size_px, cy + size_px))

        # iterate nodes and apply weighted displacement
        for j in range(ny):
            for i in range(nx):
                if self.pin_border_var.get() and self._is_border_node(j, i, ny, nx):
                    continue

                ku = float(dest[j, i, 0])
                kv = float(dest[j, i, 1])
                kx = x0 + ku * fw
                ky = y0 + kv * fh

                # mask + distance
                if shape == "Circle":
                    d = math.hypot(kx - cx, ky - cy)
                    if d > size_px:
                        continue
                    w = 1.0 - (d / size_px)
                elif shape == "Rectangle":
                    if abs(kx - cx) > size_px or abs(ky - cy) > size_px:
                        continue
                    # soft weight: max of normalized distances
                    w = 1.0 - max(abs(kx - cx), abs(ky - cy)) / size_px
                else:  # Triangle
                    # barycentric inside test
                    (xA, yA), (xB, yB), (xC, yC) = tri
                    # sign tests
                    d1 = (kx - xB) * (yA - yB) - (xA - xB) * (ky - yB)
                    d2 = (kx - xC) * (yB - yC) - (xB - xC) * (ky - yC)
                    d3 = (kx - xA) * (yC - yA) - (xC - xA) * (ky - yA)
                    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
                    if has_neg and has_pos:
                        continue
                    # Use distance to center for weight
                    d = math.hypot(kx - cx, ky - cy)
                    if d > size_px:
                        continue
                    w = 1.0 - (d / size_px)

                # Smooth weight curve and include density
                w = max(0.0, min(1.0, w))
                w = (w * w) * (0.15 + 0.85 * density)

                if tool == "Push":
                    ddu, ddv = du, dv
                elif tool == "Pull":
                    # pull toward cursor
                    uu, vv = self._canvas_to_uv(cx, cy) or (0.5, 0.5)
                    ddu = (uu - ku)
                    ddv = (vv - kv)
                elif tool == "Bloat":
                    uu, vv = self._canvas_to_uv(cx, cy) or (0.5, 0.5)
                    ddu = (ku - uu)
                    ddv = (kv - vv)
                elif tool == "Pinch":
                    uu, vv = self._canvas_to_uv(cx, cy) or (0.5, 0.5)
                    ddu = (uu - ku)
                    ddv = (vv - kv)
                else:  # Twirl
                    uu, vv = self._canvas_to_uv(cx, cy) or (0.5, 0.5)
                    vx = ku - uu
                    vy = kv - vv
                    # small angle per step
                    ang = (rate * pressure * w) * 0.35  # radians-ish small
                    ca = math.cos(ang)
                    sa = math.sin(ang)
                    rx = ca * vx - sa * vy
                    ry = sa * vx + ca * vy
                    ddu = (rx - vx)
                    ddv = (ry - vy)

                # Apply
                step = rate * pressure * w
                dest[j, i, 0] = max(-0.50, min(1.50, ku + ddu * step))
                dest[j, i, 1] = max(-0.50, min(1.50, kv + ddv * step))


    def redraw(self):
        if not PIL_OK:
            self.canvas.delete("all")
            self.canvas.create_text(10, 10, anchor="nw", fill="white", text="Install pillow: pip install pillow")
            return

        # show current output if available, else show source
        if self.app.last_preview_out is not None:
            img_show = self.app.last_preview_out
        elif self.app.preview_src_small is not None:
            img_show = self.app.preview_src_small
        else:
            self.canvas.delete("all")
            self.canvas.create_text(10, 10, anchor="nw", fill="white", text="Select an image/video in the main window first.")
            return

        if img_show.ndim == 2:
            rgb = cv2.cvtColor(img_show, cv2.COLOR_GRAY2RGB)
        elif img_show.shape[2] == 4:
            rgb = cv2.cvtColor(img_show[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(img_show[:, :, :3], cv2.COLOR_BGR2RGB)

        cw = int(self.canvas.winfo_width() or 1024)
        ch = int(self.canvas.winfo_height() or 700)
        ih, iw = rgb.shape[:2]
        scale = min((cw - 20) / max(1, iw), (ch - 20) / max(1, ih))
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        rgb2 = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(rgb2)
        self._imgtk = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        x0 = (cw - nw) // 2
        y0 = (ch - nh) // 2
        self.canvas.create_image(cw // 2, ch // 2, image=self._imgtk)
        self._fit_rect = (float(x0), float(y0), float(nw), float(nh))

        if not self.show_grid_var.get():
            return

        dest = self.app.mesh_dest_norm
        ny, nx, _ = dest.shape

        # lines
        for j in range(ny):
            pts = []
            for i in range(nx):
                u = float(dest[j, i, 0])
                v = float(dest[j, i, 1])
                pts.append((x0 + u * nw, y0 + v * nh))
            for i in range(nx - 1):
                self.canvas.create_line(*pts[i], *pts[i + 1], fill="#00ffcc", width=1)

        for i in range(nx):
            pts = []
            for j in range(ny):
                u = float(dest[j, i, 0])
                v = float(dest[j, i, 1])
                pts.append((x0 + u * nw, y0 + v * nh))
            for j in range(ny - 1):
                self.canvas.create_line(*pts[j], *pts[j + 1], fill="#00ffcc", width=1)

        # knots
        for j in range(ny):
            for i in range(nx):
                u = float(dest[j, i, 0])
                v = float(dest[j, i, 1])
                kx = x0 + u * nw
                ky = y0 + v * nh
                r = 5
                self.canvas.create_oval(kx - r - 1, ky - r - 1, kx + r + 1, ky + r + 1, fill="#000000", outline="")
                self.canvas.create_oval(kx - r, ky - r, kx + r, ky + r, fill="#ffffff", outline="")

        # Brush outline (Liquify mode)
        if str(self.edit_mode_var.get()) == "Liquify" and self._hover is not None:
            hx, hy = self._hover
            size_px = float(self.brush_size_var.get())
            shape = str(self.brush_shape_var.get())
            if shape == "Circle":
                self.canvas.create_oval(hx - size_px, hy - size_px, hx + size_px, hy + size_px, outline="#ffcc00", width=2)
            elif shape == "Rectangle":
                self.canvas.create_rectangle(hx - size_px, hy - size_px, hx + size_px, hy + size_px, outline="#ffcc00", width=2)
            else:
                self.canvas.create_polygon(
                    hx, hy - size_px,
                    hx - size_px, hy + size_px,
                    hx + size_px, hy + size_px,
                    outline="#ffcc00", fill="", width=2
                )


        # center mark
        self.canvas.create_line(cw / 2 - 14, ch / 2, cw / 2 + 14, ch / 2, fill="#ffffff", width=2)
        self.canvas.create_line(cw / 2, ch / 2 - 14, cw / 2, ch / 2 + 14, fill="#ffffff", width=2)


class FullscreenPreview(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Live Preview (Fullscreen)")
        self.configure(bg="#111")
        self.attributes("-fullscreen", True)

        self.canvas = tk.Canvas(self, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.info = ttk.Label(self, text="Esc: exit | F: toggle fullscreen | Drag: center | Shift/Right-drag: pan | Tools → Custom Mesh Warp…", anchor="w")
        self.info.pack(fill="x")

        self.imgtk = None

        self.bind("<Escape>", lambda e: self.close())
        self.bind("f", lambda e: self.toggle_fullscreen())
        self.bind("F", lambda e: self.toggle_fullscreen())
        self.bind("<Configure>", lambda e: master.rerender_last_preview())

    def toggle_fullscreen(self):
        cur = bool(self.attributes("-fullscreen"))
        self.attributes("-fullscreen", not cur)

    def close(self):
        try:
            self.destroy()
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image/Video Distorter")
        self.geometry("1180x720")

        self.files: list[Path] = []

        self.work_q = queue.Queue()
        self.preview_q = queue.Queue()

        # Preview state
        self.preview_token = 0
        self.preview_job = None
        self.preview_source_path: Path | None = None
        self.preview_src_small = None  # small BGR frame for preview

        # last preview cached for resize re-render
        self.last_preview_in = None
        self.last_preview_out = None
        self.last_preview_settings: DistortSettings | None = None
        self.last_preview_info_text = ""

        # Keep image refs
        self.preview_imgtk_main = None
        self.fs_win: FullscreenPreview | None = None

        # Resize debounce
        self._resize_job = None
        self.bind("<Configure>", self._on_window_resize)

        # Canvas layout for preview split rendering
        self._canvas_layouts: dict[int, dict] = {}

        # Drag state for preview interactions (focus + pan)
        self._drag_state = {}

        # -------- Mesh state --------
        self.mesh_enabled_var = tk.BooleanVar(value=False)
        self.mesh_nx = 9
        self.mesh_ny = 7
        self.mesh_dest_norm = self._make_identity_mesh(self.mesh_nx, self.mesh_ny)  # ny,nx,2
        self._mesh_editor: MeshEditor | None = None

        self._build_ui()
        self._build_menu()

        self.after(60, self._poll_work_queue)
        self.after(60, self._poll_preview_queue)

    # ---------------- Mesh helpers ----------------
    @staticmethod
    def _make_identity_mesh(nx: int, ny: int) -> np.ndarray:
        xs = np.linspace(0.0, 1.0, nx, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, ny, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        return np.stack([X, Y], axis=-1).astype(np.float32)

    def set_mesh_grid_size(self, nx: int, ny: int):
        nx = int(max(3, min(25, nx)))
        ny = int(max(3, min(25, ny)))
        self.mesh_nx = nx
        self.mesh_ny = ny
        self.mesh_dest_norm = self._make_identity_mesh(nx, ny)

    def reset_mesh_identity(self):
        self.mesh_dest_norm = self._make_identity_mesh(self.mesh_nx, self.mesh_ny)

    # ---------------- Menu ----------------
    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Add Files…", command=self.add_files)
        file_menu.add_command(label="Add Folder…", command=self.add_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Save As… (selected)", command=self.save_as_selected)
        file_menu.add_command(label="Process All → Save next to script", command=self.process_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)

        tools_menu = tk.Menu(menubar, tearoff=False)
        tools_menu.add_command(label="Custom Mesh Warp…", command=self.open_mesh_editor)
        tools_menu.add_checkbutton(label="Enable Mesh Warp", variable=self.mesh_enabled_var, command=self._schedule_preview)
        tools_menu.add_command(label="Reset Mesh (identity)", command=lambda: (self.reset_mesh_identity(), self._schedule_preview()))
        tools_menu.add_separator()
        tools_menu.add_command(label="Fullscreen Preview", command=self.toggle_fullscreen_preview)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(
            label="About",
            command=lambda: messagebox.showinfo(
                "About",
                "Lens distortion + custom mesh warp.\n\n"
                "Use Tools → Custom Mesh Warp… to drag knots and create asymmetric / wobbly distortions.\n"
                "Mesh is applied AFTER lens distortion."
            ),
        )

        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def open_mesh_editor(self):
        if not PIL_OK:
            self._log("Install pillow for mesh editor preview: pip install pillow")
            return
        if self._mesh_editor is not None and self._mesh_editor.winfo_exists():
            self._mesh_editor.lift()
            return
        self._mesh_editor = MeshEditor(self)

    # ---------------- UI ----------------
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # Left panel constrained, right expands for big preview
        left = ttk.Frame(root, width=320)
        left.pack(side="left", fill="y", expand=False)
        left.pack_propagate(False)

        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True)

        # ------- Left: Inputs + log -------
        ttk.Label(left, text="Inputs").pack(anchor="w")
        list_frame = ttk.Frame(left)
        list_frame.pack(fill="x")

        self.listbox = tk.Listbox(list_frame, height=8, selectmode=tk.EXTENDED)
        self.listbox.pack(side="left", fill="x", expand=True)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._on_select())

        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=sb.set)

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(8, 8))
        ttk.Button(btns, text="Add Files...", command=self.add_files).pack(side="left")
        ttk.Button(btns, text="Add Folder...", command=self.add_folder).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear", command=self.clear_files).pack(side="left")

        ttk.Label(left, text="Log").pack(anchor="w")
        self.log = tk.Text(left, height=8, wrap="word")
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

        prog = ttk.Frame(left)
        prog.pack(fill="x", pady=(8, 0))
        self.pb = ttk.Progressbar(prog, mode="determinate")
        self.pb.pack(side="left", fill="x", expand=True)
        self.pb_label = ttk.Label(prog, text="0/0")
        self.pb_label.pack(side="right", padx=(8, 0))

        # ------- Right: Controls (scrollable, height-limited) -------
        controls_outer = ttk.LabelFrame(right, text="Controls", padding=0)
        controls_outer.pack(fill="x", expand=False)
        controls_outer.configure(height=360)
        controls_outer.pack_propagate(False)

        self._controls_canvas = tk.Canvas(controls_outer, highlightthickness=0)
        c_scroll = ttk.Scrollbar(controls_outer, orient="vertical", command=self._controls_canvas.yview)
        self._controls_canvas.configure(yscrollcommand=c_scroll.set)

        c_scroll.pack(side="right", fill="y")
        self._controls_canvas.pack(side="left", fill="both", expand=True)

        controls = ttk.Frame(self._controls_canvas, padding=10)
        self._controls_window = self._controls_canvas.create_window((0, 0), window=controls, anchor="nw")

        def _controls_on_configure(_e=None):
            self._controls_canvas.configure(scrollregion=self._controls_canvas.bbox("all"))

        def _controls_on_canvas_configure(e):
            self._controls_canvas.itemconfigure(self._controls_window, width=e.width)

        controls.bind("<Configure>", _controls_on_configure)
        self._controls_canvas.bind("<Configure>", _controls_on_canvas_configure)

        # Mode
        self.mode_var = tk.StringVar(value="easy")
        row = ttk.Frame(controls)
        row.pack(fill="x")
        ttk.Radiobutton(row, text="Easy", variable=self.mode_var, value="easy", command=self._sync_mode).pack(side="left")
        ttk.Radiobutton(row, text="Advanced", variable=self.mode_var, value="advanced", command=self._sync_mode).pack(side="left", padx=10)

        ttk.Separator(controls).pack(fill="x", pady=10)

        # Strength
        ttk.Label(controls, text="Strength (-1 barrel … +1 pincushion)").pack(anchor="w")
        self.strength_var = tk.DoubleVar(value=0.35)
        self.strength = ttk.Scale(controls, from_=-1.0, to=1.0, variable=self.strength_var,
                                  command=lambda _=None: self._schedule_preview())
        self.strength.pack(fill="x")
        self.str_read = ttk.Label(controls, text="0.35")
        self.str_read.pack(anchor="e")
        self.strength_var.trace_add("write", lambda *_: self._on_strength())

        # Focal
        ttk.Label(controls, text="Focal scale (feel / zoom)").pack(anchor="w", pady=(10, 0))
        self.focal_var = tk.DoubleVar(value=1.0)
        self.focal = ttk.Scale(controls, from_=0.5, to=2.0, variable=self.focal_var,
                               command=lambda _=None: self._schedule_preview())
        self.focal.pack(fill="x")
        self.focal_read = ttk.Label(controls, text="1.00")
        self.focal_read.pack(anchor="e")
        self.focal_var.trace_add("write", lambda *_: self._on_focal())

        # Framing
        framing = ttk.LabelFrame(controls, text="Framing (prevent clipping / move center)", padding=10)
        framing.pack(fill="x", pady=(10, 0))

        ttk.Label(framing, text="Output scale (expand canvas)").pack(anchor="w")
        self.outscale_var = tk.DoubleVar(value=1.0)
        self.outscale = ttk.Scale(framing, from_=1.0, to=2.0, variable=self.outscale_var,
                                  command=lambda _=None: self._schedule_preview())
        self.outscale.pack(fill="x")
        self.outscale_read = ttk.Label(framing, text="1.00")
        self.outscale_read.pack(anchor="e")
        self.outscale_var.trace_add("write", lambda *_: self._on_outscale())

        ttk.Label(framing, text="Focus X").pack(anchor="w", pady=(8, 0))
        self.focusx_var = tk.DoubleVar(value=0.0)
        self.focusx = ttk.Scale(framing, from_=-0.45, to=0.45, variable=self.focusx_var,
                                command=lambda _=None: self._schedule_preview())
        self.focusx.pack(fill="x")
        self.focusx_read = ttk.Label(framing, text="0.00")
        self.focusx_read.pack(anchor="e")
        self.focusx_var.trace_add("write", lambda *_: self._on_focusx())

        ttk.Label(framing, text="Focus Y").pack(anchor="w", pady=(8, 0))
        self.focusy_var = tk.DoubleVar(value=0.0)
        self.focusy = ttk.Scale(framing, from_=-0.45, to=0.45, variable=self.focusy_var,
                                command=lambda _=None: self._schedule_preview())
        self.focusy.pack(fill="x")
        self.focusy_read = ttk.Label(framing, text="0.00")
        self.focusy_read.pack(anchor="e")
        self.focusy_var.trace_add("write", lambda *_: self._on_focusy())

        ttk.Label(framing, text="Pan X").pack(anchor="w", pady=(8, 0))
        self.panx_var = tk.DoubleVar(value=0.0)
        self.panx = ttk.Scale(framing, from_=-0.45, to=0.45, variable=self.panx_var,
                              command=lambda _=None: self._schedule_preview())
        self.panx.pack(fill="x")
        self.panx_read = ttk.Label(framing, text="0.00")
        self.panx_read.pack(anchor="e")
        self.panx_var.trace_add("write", lambda *_: self._on_panx())

        ttk.Label(framing, text="Pan Y").pack(anchor="w", pady=(8, 0))
        self.pany_var = tk.DoubleVar(value=0.0)
        self.pany = ttk.Scale(framing, from_=-0.45, to=0.45, variable=self.pany_var,
                              command=lambda _=None: self._schedule_preview())
        self.pany.pack(fill="x")
        self.pany_read = ttk.Label(framing, text="0.00")
        self.pany_read.pack(anchor="e")
        self.pany_var.trace_add("write", lambda *_: self._on_pany())

        # Pan stage: whether Pan is applied before distortion (affects distortion) or after (reframe)
        ttk.Label(framing, text="Pan behavior").pack(anchor="w", pady=(10, 0))
        self.pan_stage_var = tk.StringVar(value="After distortion (reframe)")
        ttk.Combobox(
            framing,
            textvariable=self.pan_stage_var,
            values=["After distortion (reframe)", "Before distortion (affects distortion)"],
            state="readonly",
        ).pack(fill="x")
        self.pan_stage_var.trace_add("write", lambda *_: self._schedule_preview())

        ttk.Button(framing, text="Reset framing", command=self._reset_framing).pack(anchor="e", pady=(6, 0))

        # Advanced coeffs
        self.adv = ttk.LabelFrame(controls, text="Advanced Coefficients", padding=10)
        self.adv.pack(fill="x", pady=(10, 0))

        self.k1_var = tk.DoubleVar(value=0.2)
        self.k2_var = tk.DoubleVar(value=0.0)
        self.k3_var = tk.DoubleVar(value=0.0)
        self.p1_var = tk.DoubleVar(value=0.0)
        self.p2_var = tk.DoubleVar(value=0.0)

        self._add_entry(self.adv, "k1", self.k1_var)
        self._add_entry(self.adv, "k2", self.k2_var)
        self._add_entry(self.adv, "k3", self.k3_var)
        self._add_entry(self.adv, "p1", self.p1_var)
        self._add_entry(self.adv, "p2", self.p2_var)

        ttk.Separator(controls).pack(fill="x", pady=10)

        out = ttk.LabelFrame(controls, text="Output / Preview", padding=10)
        out.pack(fill="x")

        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(out, text="Overwrite if exists (batch default)", variable=self.overwrite_var).pack(anchor="w")

        self.live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(out, text="Live preview (selected image/video)", variable=self.live_var,
                        command=self._schedule_preview).pack(anchor="w", pady=(6, 0))

        self.cross_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(out, text="Show center crosshair (lens)", variable=self.cross_var,
                        command=self.rerender_last_preview).pack(anchor="w", pady=(6, 0))

        ttk.Checkbutton(out, text="Enable custom mesh warp (Tools → Custom Mesh Warp…)",
                        variable=self.mesh_enabled_var, command=self._schedule_preview).pack(anchor="w", pady=(6, 0))
        ttk.Button(out, text="Open Mesh Editor…", command=self.open_mesh_editor).pack(anchor="w", pady=(4, 0))

        border_row = ttk.Frame(out)
        border_row.pack(fill="x", pady=(8, 0))
        ttk.Label(border_row, text="Border:").pack(side="left")
        self.border_var = tk.StringVar(value="Black")
        ttk.Combobox(border_row, textvariable=self.border_var, values=list(BORDER_MAP.keys()),
                     state="readonly", width=18).pack(side="right")

        interp_row = ttk.Frame(out)
        interp_row.pack(fill="x", pady=(8, 0))
        ttk.Label(interp_row, text="Interpolation:").pack(side="left")
        self.interp_var = tk.StringVar(value="Linear (recommended)")
        ttk.Combobox(interp_row, textvariable=self.interp_var, values=list(INTERP_MAP.keys()),
                     state="readonly", width=18).pack(side="right")

        # Action bar pinned
        act = ttk.Frame(right)
        act.pack(fill="x", pady=(10, 0))
        ttk.Button(act, text="Save As… (selected)", command=self.save_as_selected).pack(side="left")
        ttk.Button(act, text="Fullscreen Preview", command=self.toggle_fullscreen_preview).pack(side="left", padx=8)
        ttk.Button(act, text="Process All → Save next to script", command=self.process_all).pack(side="right")

        # Live preview (big)
        prev = ttk.LabelFrame(right, text="Live Preview", padding=10)
        prev.pack(fill="both", expand=True, pady=(10, 0))

        if PIL_OK:
            self.canvas = tk.Canvas(prev, bg="#111", highlightthickness=0)
            self.canvas.pack(fill="both", expand=True)
            self.prev_info = ttk.Label(prev, text="Drag: move center | Shift+drag or Right-drag: pan | Tools → Custom Mesh Warp…")
            self.prev_info.pack(anchor="w", pady=(6, 0))

            # Drag on preview:
            # - Left-drag: move lens center (focus)
            # - Shift+drag or Right-drag: pan content
            self.canvas.bind("<Button-1>", lambda e: self._on_preview_drag_start(e, canvas=self.canvas, mode="focus"))
            self.canvas.bind("<B1-Motion>", lambda e: self._on_preview_drag_move(e, canvas=self.canvas))
            self.canvas.bind("<ButtonRelease-1>", lambda e: self._on_preview_drag_end(e, canvas=self.canvas))
            self.canvas.bind("<Button-3>", lambda e: self._on_preview_drag_start(e, canvas=self.canvas, mode="pan"))
            self.canvas.bind("<B3-Motion>", lambda e: self._on_preview_drag_move(e, canvas=self.canvas))
            self.canvas.bind("<ButtonRelease-3>", lambda e: self._on_preview_drag_end(e, canvas=self.canvas))
        else:
            ttk.Label(prev, text="Install pillow for preview:\n  pip install pillow").pack(anchor="w")

        self._sync_mode()

        try:
            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")
        except Exception:
            pass

    # ------------- Helpers -------------
    def _add_entry(self, parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label).pack(side="left")
        e = ttk.Entry(row, textvariable=var, width=10)
        e.pack(side="right")
        var.trace_add("write", lambda *_: self._schedule_preview())

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _sync_mode(self):
        mode = self.mode_var.get()
        state = "normal" if mode == "advanced" else "disabled"
        for child in self.adv.winfo_children():
            for w in child.winfo_children():
                if isinstance(w, ttk.Entry):
                    w.configure(state=state)
        self._schedule_preview()

    # readouts
    def _on_strength(self): self.str_read.configure(text=f"{self.strength_var.get():.2f}")
    def _on_focal(self): self.focal_read.configure(text=f"{self.focal_var.get():.2f}")
    def _on_outscale(self): self.outscale_read.configure(text=f"{self.outscale_var.get():.2f}")
    def _on_focusx(self): self.focusx_read.configure(text=f"{self.focusx_var.get():.2f}")
    def _on_focusy(self): self.focusy_read.configure(text=f"{self.focusy_var.get():.2f}")
    def _on_panx(self): self.panx_read.configure(text=f"{self.panx_var.get():.2f}")
    def _on_pany(self): self.pany_read.configure(text=f"{self.pany_var.get():.2f}")

    def _reset_framing(self):
        self.outscale_var.set(1.0)
        self.focusx_var.set(0.0)
        self.focusy_var.set(0.0)
        self.panx_var.set(0.0)
        self.pany_var.set(0.0)
        self._schedule_preview()

    # ---------- Left pane actions ----------
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select images/videos",
            filetypes=[("Media", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
                       ("All files", "*.*")]
        )
        if not paths:
            return
        for p in paths:
            path = Path(p)
            if is_image(path) or is_video(path):
                self.files.append(path)
        self._refresh_list()

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select a folder")
        if not folder:
            return
        folder = Path(folder)
        for p in folder.rglob("*"):
            if p.is_file() and (is_image(p) or is_video(p)):
                self.files.append(p)
        self._refresh_list()

    def clear_files(self):
        self.files.clear()
        self._refresh_list()

    def _refresh_list(self):
        self.listbox.delete(0, "end")
        for f in self.files:
            self.listbox.insert("end", str(f))
        self.pb.configure(value=0, maximum=max(1, len(self.files)))
        self.pb_label.configure(text=f"0/{len(self.files)}")

    # ---------- Settings ----------
    def _get_settings(self) -> DistortSettings:
        mode = self.mode_var.get()
        strength = float(self.strength_var.get())
        focal_scale = float(self.focal_var.get())

        output_scale = float(self.outscale_var.get())
        focus_x = float(self.focusx_var.get())
        focus_y = float(self.focusy_var.get())
        pan_x = float(self.panx_var.get())
        pan_y = float(self.pany_var.get())

        if mode == "easy":
            d = strength_to_coeffs(strength)
            k1, k2, p1, p2, k3 = d.tolist()
        else:
            k1 = float(self.k1_var.get())
            k2 = float(self.k2_var.get())
            k3 = float(self.k3_var.get())
            p1 = float(self.p1_var.get())
            p2 = float(self.p2_var.get())

        return DistortSettings(
            mode=mode,
            strength=strength,
            focal_scale=focal_scale,

            output_scale=output_scale,
            focus_x=focus_x,
            focus_y=focus_y,
            pan_x=pan_x,
            pan_y=pan_y,

            pan_stage=str(self.pan_stage_var.get()),

            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            overwrite=bool(self.overwrite_var.get()),
            border=str(self.border_var.get()),
            interp=str(self.interp_var.get()),
            show_crosshair=bool(self.cross_var.get()),
            mesh_enabled=bool(self.mesh_enabled_var.get()),
        )

    # ---------- Resize: re-render preview to scale with window ----------
    def _on_window_resize(self, _event):
        if not PIL_OK:
            return
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.after(80, self.rerender_last_preview)

    def rerender_last_preview(self):
        if not PIL_OK:
            return
        if self.last_preview_in is None or self.last_preview_out is None or self.last_preview_settings is None:
            return
        self._render_to_canvas(self.canvas, self.last_preview_in, self.last_preview_out, is_fullscreen=False)
        if self.fs_win is not None and self.fs_win.winfo_exists():
            self._render_to_canvas(self.fs_win.canvas, self.last_preview_in, self.last_preview_out, is_fullscreen=True)
            self.fs_win.info.configure(text=self.last_preview_info_text)

        if self._mesh_editor is not None and self._mesh_editor.winfo_exists():
            self._mesh_editor.redraw()

    # ------------------ Live preview ------------------
    def _on_select(self):
        if not PIL_OK:
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        path = self.files[int(sel[0])]
        self.preview_source_path = path

        img = None
        if is_image(path):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        elif is_video(path):
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                img = frame

        if img is None:
            self.preview_src_small = None
            if hasattr(self, "prev_info"):
                self.prev_info.configure(text="Could not read selection.")
            return

        h, w = img.shape[:2]
        max_dim = max(w, h)
        target = 1100
        scale = 1.0 if max_dim <= target else (target / max_dim)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        self.preview_src_small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        self._schedule_preview()

    def _schedule_preview(self):
        if not PIL_OK or not self.live_var.get() or self.preview_src_small is None:
            return
        if self.preview_job is not None:
            try:
                self.after_cancel(self.preview_job)
            except Exception:
                pass
        self.preview_job = self.after(60, self._start_preview_worker)

    def _start_preview_worker(self):
        if self.preview_src_small is None:
            return
        settings = self._get_settings()
        self.preview_token += 1
        token = self.preview_token
        src = self.preview_src_small.copy()
        threading.Thread(target=self._preview_worker, args=(token, src, settings), daemon=True).start()

    def _preview_worker(self, token: int, img: np.ndarray, settings: DistortSettings):
        try:
            h, w = img.shape[:2]
            w_out = max(1, int(round(w * settings.output_scale)))
            h_out = max(1, int(round(h * settings.output_scale)))

            border_mode = BORDER_MAP.get(settings.border, cv2.BORDER_CONSTANT)
            interp = INTERP_MAP.get(settings.interp, cv2.INTER_LINEAR)

            # 1) Lens distortion

            # Optional: apply pan BEFORE distortion (affects distortion)
            if settings.pan_stage == "Before distortion (affects distortion)":
                img_work = self._apply_pre_pan(img, settings.pan_x, settings.pan_y, interp, border_mode)
            else:
                img_work = img

            # 1) Lens distortion
            K_src = build_camera_matrix(w, h, settings.focal_scale, 0.0, 0.0)
            K_dist = build_camera_matrix(w_out, h_out, settings.focal_scale, settings.focus_x, settings.focus_y)
            dist = np.array([settings.k1, settings.k2, settings.p1, settings.p2, settings.k3], dtype=np.float64)
            pan_x_px = 0.0
            pan_y_px = 0.0

            map_x_l, map_y_l = make_distort_maps_blocked(
                w_out, h_out, K_dist, dist, K_src,
                pan_x_px=pan_x_px, pan_y_px=pan_y_px,
                block_rows=256
            )
            lens_out = cv2.remap(img_work, map_x_l, map_y_l, interpolation=interp, borderMode=border_mode)

            # 2) Mesh warp (optional), applied AFTER lens
            if settings.mesh_enabled:
                map_x_m, map_y_m = make_mesh_maps_triangular(
                    w_out, h_out, w_out, h_out, self.mesh_dest_norm
                )
                out = cv2.remap(lens_out, map_x_m, map_y_m, interpolation=interp, borderMode=border_mode)
            else:
                out = lens_out

            # 3) Post-pan translation (does NOT change distortion)
            if settings.pan_stage == "After distortion (reframe)":
                out = self._apply_post_pan(out, settings.pan_x, settings.pan_y, interp, border_mode)

            self.preview_q.put(("preview", token, img, out, settings))
        except Exception as e:
            self.preview_q.put(("preview_err", token, str(e)))

    def _poll_preview_queue(self):
        try:
            while True:
                item = self.preview_q.get_nowait()
                kind = item[0]
                if kind == "preview":
                    _, token, img_in, img_out, settings = item
                    if token != self.preview_token:
                        continue

                    self.last_preview_in = img_in
                    self.last_preview_out = img_out
                    self.last_preview_settings = settings

                    model = "LENS+MESH" if settings.mesh_enabled else "LENS"
                    info = (
                        f"{self.preview_source_path.name if self.preview_source_path else ''} | {model} | "
                        f"out={settings.output_scale:.2f} | "
                        f"strength={settings.strength:.2f} | focal={settings.focal_scale:.2f}"
                    )
                    self.last_preview_info_text = info
                    if hasattr(self, "prev_info"):
                        self.prev_info.configure(text=info)

                    self.rerender_last_preview()

                elif kind == "preview_err":
                    _, token, msg = item
                    if token == self.preview_token and hasattr(self, "prev_info"):
                        self.prev_info.configure(text=f"Preview error: {msg}")
        except queue.Empty:
            pass

        self.after(60, self._poll_preview_queue)

    def _render_to_canvas(self, canvas: tk.Canvas, img_in: np.ndarray, img_out: np.ndarray, is_fullscreen: bool):
        if not PIL_OK:
            return

        canvas_w = int(canvas.winfo_width() or (1400 if is_fullscreen else 800))
        canvas_h = int(canvas.winfo_height() or (800 if is_fullscreen else 520))
        half_w = max(1, canvas_w // 2)

        def to_rgb(im):
            if im.ndim == 2:
                return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            if im.shape[2] == 4:
                bgr = im[:, :, :3]
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB)

        a = to_rgb(img_in)
        b = to_rgb(img_out)

        def fit(im, target_w, target_h):
            ih, iw = im.shape[:2]
            scale = min((target_w - 12) / iw, (target_h - 12) / ih)
            nw = max(1, int(iw * scale))
            nh = max(1, int(ih * scale))
            return cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)

        a2 = fit(a, half_w, canvas_h)
        b2 = fit(b, half_w, canvas_h)

        frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        y0 = (canvas_h - a2.shape[0]) // 2
        x0 = (half_w - a2.shape[1]) // 2
        frame[y0:y0 + a2.shape[0], x0:x0 + a2.shape[1]] = a2

        y1 = (canvas_h - b2.shape[0]) // 2
        x1 = half_w + (half_w - b2.shape[1]) // 2
        frame[y1:y1 + b2.shape[0], x1:x1 + b2.shape[1]] = b2

        pil = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(pil)

        if canvas is getattr(self, "canvas", None):
            self.preview_imgtk_main = imgtk
        else:
            if self.fs_win is not None:
                self.fs_win.imgtk = imgtk

        canvas.delete("all")
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=imgtk)

        self._canvas_layouts[id(canvas)] = {
            "left": (float(x0), float(y0), float(a2.shape[1]), float(a2.shape[0])),
            "right": (float(x1), float(y1), float(b2.shape[1]), float(b2.shape[0])),
        }

        if self.last_preview_settings is not None and self.last_preview_settings.show_crosshair:
            self._draw_crosshair(canvas)

    def _draw_crosshair(self, canvas: tk.Canvas):
        layout = self._canvas_layouts.get(id(canvas))
        if not layout or self.last_preview_settings is None:
            return
        s = self.last_preview_settings
        rx0, ry0, rw, rh = layout["right"]

        cx = rx0 + (0.5 + float(s.focus_x)) * rw
        cy = ry0 + (0.5 + float(s.focus_y)) * rh
        cx = max(rx0, min(rx0 + rw, cx))
        cy = max(ry0, min(ry0 + rh, cy))

        size = max(12.0, min(rw, rh) * 0.06)
        canvas.create_line(cx - size, cy, cx + size, cy, width=3, fill="#000000")
        canvas.create_line(cx, cy - size, cx, cy + size, width=3, fill="#000000")
        canvas.create_line(cx - size, cy, cx + size, cy, width=1, fill="#00ffcc")


    # ------------------ Preview drag: set center / pan ------------------
    def _get_output_rect(self, canvas: tk.Canvas):
        # Returns (x0, y0, w, h) of the OUTPUT pane (right side) in the split preview.
        layout = self._canvas_layouts.get(id(canvas))
        if not layout:
            return None
        return layout.get("right")

    def _focus_from_event(self, canvas: tk.Canvas, x: float, y: float):
        rect = self._get_output_rect(canvas)
        if rect is None:
            return None
        rx0, ry0, rw, rh = rect
        if rw <= 2 or rh <= 2:
            return None
        x = max(rx0, min(rx0 + rw, x))
        y = max(ry0, min(ry0 + rh, y))
        fx = (x - rx0) / rw - 0.5
        fy = (y - ry0) / rh - 0.5
        fx = max(-0.45, min(0.45, fx))
        fy = max(-0.45, min(0.45, fy))
        return fx, fy

    def _on_preview_drag_start(self, event, canvas: tk.Canvas, mode: str):
        rect = self._get_output_rect(canvas)
        if rect is None:
            return
        # Shift = pan mode even on left click
        if (getattr(event, "state", 0) & 0x0001) != 0:
            mode = "pan"

        self._drag_state[id(canvas)] = {
            "mode": mode,
            "start_x": float(event.x),
            "start_y": float(event.y),
            "start_panx": float(self.panx_var.get()),
            "start_pany": float(self.pany_var.get()),
        }

        if mode == "focus":
            out = self._focus_from_event(canvas, float(event.x), float(event.y))
            if out is None:
                return
            fx, fy = out
            self.focusx_var.set(fx)
            self.focusy_var.set(fy)
            self._schedule_preview()

    def _on_preview_drag_move(self, event, canvas: tk.Canvas):
        st = self._drag_state.get(id(canvas))
        if not st:
            return

        mode = st.get("mode", "focus")
        if (getattr(event, "state", 0) & 0x0001) != 0:
            mode = "pan"

        rect = self._get_output_rect(canvas)
        if rect is None:
            return
        rx0, ry0, rw, rh = rect
        if rw <= 2 or rh <= 2:
            return

        if mode == "focus":
            out = self._focus_from_event(canvas, float(event.x), float(event.y))
            if out is None:
                return
            fx, fy = out
            self.focusx_var.set(fx)
            self.focusy_var.set(fy)
            self._schedule_preview()
            return

        # Pan mode: drag moves content; invert direction so drag-right moves content right
        dx = (float(event.x) - float(st["start_x"])) / rw
        dy = (float(event.y) - float(st["start_y"])) / rh

        panx = float(st["start_panx"]) + dx
        pany = float(st["start_pany"]) + dy
        panx = max(-0.45, min(0.45, panx))
        pany = max(-0.45, min(0.45, pany))

        self.panx_var.set(panx)
        self.pany_var.set(pany)
        self._schedule_preview()

    def _on_preview_drag_end(self, _event, canvas: tk.Canvas):
        self._drag_state.pop(id(canvas), None)


    def _apply_pre_pan(self, img: np.ndarray, pan_x: float, pan_y: float, interp_flag: int, border_mode: int):
        """Translate the INPUT frame before distortion (so pan changes the distortion outcome)."""
        h, w = img.shape[:2]
        tx = float(pan_x) * float(w)
        ty = float(pan_y) * float(h)
        if abs(tx) < 1e-3 and abs(ty) < 1e-3:
            return img
        M = np.array([[1.0, 0.0, tx],
                      [0.0, 1.0, ty]], dtype=np.float32)

        if img.ndim == 2:
            border_value = 0
        else:
            ch = img.shape[2]
            border_value = (0,) * ch

        return cv2.warpAffine(img, M, (w, h), flags=interp_flag, borderMode=border_mode, borderValue=border_value)

    def _apply_post_pan(self, img: np.ndarray, pan_x: float, pan_y: float, interp_flag: int, border_mode: int):
        """Translate the already-distorted output to reposition it inside the visible frame."""
        h, w = img.shape[:2]
        tx = float(pan_x) * float(w)
        ty = float(pan_y) * float(h)
        if abs(tx) < 1e-3 and abs(ty) < 1e-3:
            return img
        M = np.array([[1.0, 0.0, tx],
                      [0.0, 1.0, ty]], dtype=np.float32)

        # borderValue must match channels
        if img.ndim == 2:
            border_value = 0
        else:
            ch = img.shape[2]
            border_value = (0,) * ch

        return cv2.warpAffine(img, M, (w, h), flags=interp_flag, borderMode=border_mode, borderValue=border_value)
        canvas.create_line(cx, cy - size, cx, cy + size, width=1, fill="#00ffcc")

    # ------------------ Fullscreen preview ------------------
    def toggle_fullscreen_preview(self):
        if not PIL_OK:
            self._log("Install pillow for fullscreen preview: pip install pillow")
            return
        if self.fs_win is not None and self.fs_win.winfo_exists():
            self.fs_win.close()
            self.fs_win = None
            return
        self.fs_win = FullscreenPreview(self)
        self.rerender_last_preview()

    # ------------------ Save As (selected) ------------------
    def save_as_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            self._log("Select a file first.")
            return

        path = self.files[int(sel[0])]
        settings = self._get_settings()

        if is_image(path):
            default_name = f"{path.stem}_distorted{path.suffix.lower()}"
            out = filedialog.asksaveasfilename(
                title="Save distorted image as...",
                initialdir=str(script_dir()),
                initialfile=default_name,
                defaultextension=path.suffix.lower(),
                filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]
            )
            if not out:
                return
            out_path = Path(out)
            threading.Thread(target=self._save_as_image_worker, args=(path, out_path, settings), daemon=True).start()

        elif is_video(path):
            default_name = f"{path.stem}_distorted.mp4"
            out = filedialog.asksaveasfilename(
                title="Save distorted video as...",
                initialdir=str(script_dir()),
                initialfile=default_name,
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All files", "*.*")]
            )
            if not out:
                return
            out_path = Path(out)
            threading.Thread(target=self._save_as_video_worker, args=(path, out_path, settings), daemon=True).start()
        else:
            self._log("Unsupported file type.")

    def _save_as_image_worker(self, in_path: Path, out_path: Path, settings: DistortSettings):
        try:
            self.work_q.put(("log", f"Save As (image): {in_path.name} → {out_path.name}"))
            self._process_image_to_path(in_path, out_path, settings)
            self.work_q.put(("log", f"Saved: {out_path}"))
        except Exception as e:
            self.work_q.put(("log", f"Save As error: {e}"))

    def _save_as_video_worker(self, in_path: Path, out_path: Path, settings: DistortSettings):
        try:
            self.work_q.put(("log", f"Save As (video): {in_path.name} → {out_path.name}"))
            self._process_video_to_path(in_path, out_path, settings)
            self.work_q.put(("log", f"Saved: {out_path}"))
        except Exception as e:
            self.work_q.put(("log", f"Save As error: {e}"))

    # ------------------ Batch processing ------------------
    def process_all(self):
        if not self.files:
            self._log("Add files first.")
            return
        settings = self._get_settings()
        threading.Thread(target=self._worker_process, args=(list(self.files), settings), daemon=True).start()

    def _worker_process(self, files: list[Path], settings: DistortSettings):
        out_dir = script_dir()
        self.work_q.put(("log", f"Output folder: {out_dir}"))
        self.work_q.put(("progress_max", len(files)))
        done = 0

        for path in files:
            try:
                if is_image(path):
                    out_path = out_dir / f"{path.stem}_distorted{path.suffix.lower()}"
                    if (not settings.overwrite) and out_path.exists():
                        i = 1
                        while True:
                            cand = out_dir / f"{path.stem}_distorted_{i}{path.suffix.lower()}"
                            if not cand.exists():
                                out_path = cand
                                break
                            i += 1
                    self._process_image_to_path(path, out_path, settings)
                    self.work_q.put(("log", f"Saved: {out_path.name}"))

                elif is_video(path):
                    out_path = out_dir / f"{path.stem}_distorted.mp4"
                    if (not settings.overwrite) and out_path.exists():
                        i = 1
                        while True:
                            cand = out_dir / f"{path.stem}_distorted_{i}.mp4"
                            if not cand.exists():
                                out_path = cand
                                break
                            i += 1
                    self._process_video_to_path(path, out_path, settings)
                    self.work_q.put(("log", f"Saved: {out_path.name}"))

                else:
                    self.work_q.put(("log", f"Skipped: {path.name}"))
            except Exception as e:
                self.work_q.put(("log", f"Error on {path.name}: {e}"))

            done += 1
            self.work_q.put(("progress", done))

        self.work_q.put(("log", "Done."))

    def _process_image_to_path(self, in_path: Path, out_path: Path, settings: DistortSettings):
        img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("Could not read image")

        h, w = img.shape[:2]
        w_out = max(1, int(round(w * settings.output_scale)))
        h_out = max(1, int(round(h * settings.output_scale)))

        border_mode = BORDER_MAP.get(settings.border, cv2.BORDER_CONSTANT)
        interp = INTERP_MAP.get(settings.interp, cv2.INTER_LINEAR)

        # Lens maps

        # Optional: apply pan BEFORE distortion (affects distortion)
        if settings.pan_stage == "Before distortion (affects distortion)":
            img = self._apply_pre_pan(img, settings.pan_x, settings.pan_y, interp, border_mode)

        # Lens maps
        K_src = build_camera_matrix(w, h, settings.focal_scale, 0.0, 0.0)
        K_dist = build_camera_matrix(w_out, h_out, settings.focal_scale, settings.focus_x, settings.focus_y)
        dist = np.array([settings.k1, settings.k2, settings.p1, settings.p2, settings.k3], dtype=np.float64)
        pan_x_px = 0.0
        pan_y_px = 0.0
        map_x_l, map_y_l = make_distort_maps_blocked(w_out, h_out, K_dist, dist, K_src, pan_x_px, pan_y_px, 128)
        lens_out = cv2.remap(img, map_x_l, map_y_l, interpolation=interp, borderMode=border_mode)

        if settings.mesh_enabled:
            map_x_m, map_y_m = make_mesh_maps_triangular(w_out, h_out, w_out, h_out, self.mesh_dest_norm)
            out_img = cv2.remap(lens_out, map_x_m, map_y_m, interpolation=interp, borderMode=border_mode)
        else:
            out_img = lens_out

        # Post-pan translation (does NOT change distortion)
        if settings.pan_stage == "After distortion (reframe)":
            out_img = self._apply_post_pan(out_img, settings.pan_x, settings.pan_y, interp, border_mode)

        if not cv2.imwrite(str(out_path), out_img):
            raise RuntimeError("Could not write image")

    def _process_video_to_path(self, in_path: Path, out_path: Path, settings: DistortSettings):
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            cap.release()
            raise RuntimeError("Invalid video size")

        w_out = max(1, int(round(w * settings.output_scale)))
        h_out = max(1, int(round(h * settings.output_scale)))

        ext = out_path.suffix.lower()
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if ext == ".avi" else "mp4v"))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_out, h_out))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not open VideoWriter (codec/container issue)")

        border_mode = BORDER_MAP.get(settings.border, cv2.BORDER_CONSTANT)
        interp = INTERP_MAP.get(settings.interp, cv2.INTER_LINEAR)

        # Precompute lens maps once
        K_src = build_camera_matrix(w, h, settings.focal_scale, 0.0, 0.0)
        K_dist = build_camera_matrix(w_out, h_out, settings.focal_scale, settings.focus_x, settings.focus_y)
        dist = np.array([settings.k1, settings.k2, settings.p1, settings.p2, settings.k3], dtype=np.float64)
        pan_x_px = 0.0
        pan_y_px = 0.0
        map_x_l, map_y_l = make_distort_maps_blocked(w_out, h_out, K_dist, dist, K_src, pan_x_px, pan_y_px, 128)

        # Precompute mesh maps once (optional)
        if settings.mesh_enabled:
            map_x_m, map_y_m = make_mesh_maps_triangular(w_out, h_out, w_out, h_out, self.mesh_dest_norm)
        else:
            map_x_m = map_y_m = None

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if settings.pan_stage == "Before distortion (affects distortion)":
                frame = self._apply_pre_pan(frame, settings.pan_x, settings.pan_y, interp, border_mode)

            lens_out = cv2.remap(frame, map_x_l, map_y_l, interpolation=interp, borderMode=border_mode)
            if map_x_m is not None:
                out_frame = cv2.remap(lens_out, map_x_m, map_y_m, interpolation=interp, borderMode=border_mode)
            else:
                out_frame = lens_out
            if settings.pan_stage == "After distortion (reframe)":
                out_frame = self._apply_post_pan(out_frame, settings.pan_x, settings.pan_y, interp, border_mode)
            writer.write(out_frame)

        writer.release()
        cap.release()

    # ------------------ Work queue polling ------------------
    def _poll_work_queue(self):
        try:
            while True:
                item = self.work_q.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._log(item[1])
                elif kind == "progress_max":
                    mx = int(item[1])
                    self.pb.configure(maximum=max(1, mx), value=0)
                    self.pb_label.configure(text=f"0/{mx}")
                elif kind == "progress":
                    v = int(item[1])
                    mx = int(self.pb.cget("maximum"))
                    self.pb.configure(value=v)
                    self.pb_label.configure(text=f"{v}/{mx}")
        except queue.Empty:
            pass

        self.after(60, self._poll_work_queue)


if __name__ == "__main__":
    App().mainloop()
