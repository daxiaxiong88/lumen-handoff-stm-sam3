"""RGB codecs for the Vision Banana segmentation reproduction.

Vision Banana (*Image Generators are Generalist Vision Learners*,
arXiv:2604.20329) parameterises segmentation outputs as **RGB images**:
the model is prompted to colour each class (or each instance) with a
specific colour, and the resulting image is decoded back into masks by
matching pixel colours to the prompt-specified palette.

This module holds the model-independent, fully testable core of that
scheme — prompt construction and colour-based mask decoding — so the
heavy generative model (FLUX.2-klein-4B) only enters through
:mod:`lumen.models.vision_banana.segmenter`.

Colours are expressed as ``0–255`` integer RGB tuples everywhere, to
match the prompt examples in the paper (e.g. ``(255, 255, 0)``) and the
``uint8`` arrays used by :mod:`supervision`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from PIL import Image

ColorRGB = tuple[int, int, int]
ColorMap = dict[str, ColorRGB]

# Euclidean colour-ball radius (in 0–255 RGB space) within which a pixel
# is considered to belong to a prompt-specified colour. Generative outputs
# are rarely pixel-perfect, so this is intentionally lenient.
DEFAULT_SEMANTIC_TOLERANCE = 48.0
DEFAULT_INSTANCE_TOLERANCE = 64.0
DEFAULT_BACKGROUND_TOLERANCE = 60.0
DEFAULT_MIN_INSTANCE_AREA = 16
DEFAULT_CONNECTIVITY = 2  # 8-neighbourhood

# Names treated as the background class and therefore excluded from the
# returned semantic masks.
_BACKGROUND_ALIASES = frozenset({"background", "bg", "__background__", ""})


# ---------------------------------------------------------------------------
# Colour parsing / normalisation
# ---------------------------------------------------------------------------


def parse_color(value: object) -> ColorRGB:
    """Coerce a colour spec into an ``(r, g, b)`` tuple of ``0–255`` ints.

    Accepts:
    * ``(r, g, b)`` / ``[r, g, b]`` ints (0–255) or floats (0–1 if max ≤ 1).
    * hex strings ``"#rrggbb"`` / ``"rrggbb"``.
    * CSS colour names via :mod:`PIL.ImageColor` (lazy import).
    """
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"Colour must have 3 channels, got {len(value)}: {value!r}")
        channels = [float(c) for c in value]
        if max(channels) <= 1.0:  # treat as 0–1 floats
            channels = [c * 255.0 for c in channels]
        return tuple(int(round(c)) for c in channels)  # type: ignore[return-value]

    if isinstance(value, str):
        from PIL import ImageColor

        rgb = ImageColor.getrgb(value)  # accepts "#rrggbb", "rgb(...)", names
        if len(rgb) >= 3:
            return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    raise ValueError(f"Unrecognised colour spec: {value!r}")


def normalize_color_map(
    mapping: Mapping[str, object] | None,
) -> ColorMap:
    """Return a copy of *mapping* with every value coerced via :func:`parse_color`."""
    if not mapping:
        return {}
    return {str(k): parse_color(v) for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_segmentation_prompt(
    color_map: Mapping[str, object] | None,
    *,
    instance: bool = False,
    background: object | None = None,
) -> str:
    """Build a Vision-Banana-style instruction prompt.

    Semantic (``instance=False``): emit a JSON ``class -> <r,g,b>`` mapping
    embedded in the canonical Vision Banana instruction, which is the most
    reliably decodable of the prompt styles the paper demonstrates::

        Generate a visualization image of semantic segmentation, using this
        color mapping: {"cell": <0, 255, 0>, "background": <0, 0, 0>}.

    Instance (``instance=True``): instruct the model to render each instance
    of the named class in a *different* colour (the model chooses the
    per-instance colours; :func:`decode_instances` recovers them by
    connected components)::

        Generate an instance segmentation visualization of this image. Each
        cell is colored differently. Background is RGB(0, 0, 0).
    """
    cmap = normalize_color_map(color_map)

    if instance:
        targets = [k for k in cmap if k.lower() not in _BACKGROUND_ALIASES]
        bg = parse_color(background) if background is not None else cmap.get(
            "background", (0, 0, 0)
        )
        what = ", ".join(targets) if targets else "object"
        bg_str = f" Background is RGB{bg}."
        return (
            "Generate an instance segmentation visualization of this image. "
            f"Each {what} is colored differently.{bg_str}"
        )

    body = ", ".join(f'"{name}": <{r}, {g}, {b}>' for name, (r, g, b) in cmap.items())
    return (
        "Generate a visualization image of semantic segmentation, using this "
        f"color mapping: {{{body}}}."
    )


# ---------------------------------------------------------------------------
# Mask decoding
# ---------------------------------------------------------------------------


def color_distance(rgb: np.ndarray, target: ColorRGB) -> np.ndarray:
    """Per-pixel Euclidean distance from *rgb* (``HxWx3``) to *target*."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 image, got shape {rgb.shape}")
    target_arr = np.asarray(target, dtype=np.float32)
    diff = rgb.astype(np.float32) - target_arr
    return cast(np.ndarray, np.sqrt((diff * diff).sum(axis=-1)))


def decode_semantic(
    rgb: np.ndarray,
    color_map: Mapping[str, object] | None,
    *,
    tolerance: float = DEFAULT_SEMANTIC_TOLERANCE,
    exclude: Iterable[str] = _BACKGROUND_ALIASES,
) -> list[tuple[str, np.ndarray]]:
    """Decode a semantic-segmentation RGB image into per-class masks.

    Per the paper, each pixel is assigned to the class whose target colour is
    **closest** (winner-take-all argmin in RGB space); pixels whose nearest
    class is farther than *tolerance* are left unassigned. Background-named
    classes (in *exclude*) are not returned, but still compete for pixels, so a
    ``background`` entry cleanly claims non-object regions. Classes are returned
    in ``color_map`` order; empty classes are skipped.

    Returns:
        ``[(class_name, bool mask HxW), ...]``.
    """
    cmap = normalize_color_map(color_map)
    if not cmap:
        return []
    excluded = {str(e).lower() for e in exclude}
    names = list(cmap)
    dists = np.stack([color_distance(rgb, cmap[n]) for n in names], axis=0)
    nearest = dists.argmin(axis=0)
    assigned = dists.min(axis=0) <= tolerance
    out: list[tuple[str, np.ndarray]] = []
    for i, name in enumerate(names):
        if name.lower() in excluded:
            continue
        mask = (nearest == i) & assigned
        if not mask.any():
            continue
        out.append((name, mask))
    return out


def decode_instances(
    rgb: np.ndarray,
    background: object = (0, 0, 0),
    *,
    tau: float = 14.0,
    min_area_frac: float = 2e-4,
    theta_erosion: float = 0.1,
    gamma: float = 5.0,
) -> list[np.ndarray]:
    """Decode an instance-segmentation RGB image into per-instance masks.

    Faithful implementation of the paper's multi-stage clustering (Appendix A).
    The model colours each instance with a distinct colour against *background*;
    we recover individual masks via:

    1. **background removal** — pixels within ``tau`` of the background colour,
    2. **colour-seeded flood-fill** — a region grows from a seed pixel, taking
       8-neighbours whose colour is within ``tau`` of the *seed's* colour, so
       adjacent instances of different colours stay separate,
    3. **noise pruning** — drop components smaller than ``min_area_frac``·H·W,
    4. **boundary-halo elimination** — drop components that survive a 3×3
       erosion by less than ``theta_erosion`` (thin generative halos),
    5. **spatially-constrained merging** — merge disjoint components with
       similar mean colour (≤ ``tau``) and bbox expansion ≤ ``gamma``.

    Returns ``[bool mask HxW, ...]`` ordered largest-area first.
    """
    h, w = rgb.shape[:2]
    min_area = max(1, int(min_area_frac * h * w))
    conn8 = cast(np.ndarray, ndimage.generate_binary_structure(2, 2))
    sq3 = np.ones((3, 3), dtype=bool)

    # 1. background
    is_bg = color_distance(rgb, parse_color(background)) <= tau
    labeled = np.zeros((h, w), dtype=np.int32)
    comps: list[np.ndarray] = []

    # 2. colour-seeded flood-fill (one region per distinct colour cluster)
    while True:
        pending = (~is_bg) & (labeled == 0)
        if not pending.any():
            break
        ys, xs = np.where(pending)
        y0, x0 = int(ys[0]), int(xs[0])
        seed = rgb[y0, x0]
        within = (
            color_distance(rgb, (int(seed[0]), int(seed[1]), int(seed[2]))) <= tau
        ) & (~is_bg) & (labeled == 0)
        cc, _ = ndimage.label(within, structure=conn8)
        comp = (cc == cc[y0, x0]) & (cc != 0)
        labeled[comp] = len(comps) + 1
        comps.append(comp)

    # 3. noise pruning
    comps = [c for c in comps if int(c.sum()) >= min_area]

    # 4. boundary-halo erosion pruning
    kept: list[np.ndarray] = []
    for c in comps:
        area = int(c.sum())
        if area == 0:
            continue
        eroded = cast(np.ndarray, ndimage.binary_erosion(c, structure=sq3))
        if eroded.sum() / area >= theta_erosion:
            kept.append(c)
    comps = kept

    # 5. spatially-constrained merging (union-find over mean colour + bbox)
    n = len(comps)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def bbox_area(mask: np.ndarray) -> int:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return 0
        return int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))

    means = [rgb[c].astype(np.float64).mean(axis=0) for c in comps]
    for a in range(n):
        for b in range(a + 1, n):
            if find(a) == find(b):
                continue
            if np.linalg.norm(means[a] - means[b]) > tau:
                continue
            union_bbox = bbox_area(comps[a] | comps[b])
            if union_bbox <= gamma * (comps[a].sum() + comps[b].sum()):
                parent[find(b)] = find(a)

    groups: dict[int, np.ndarray] = {}
    for i, c in enumerate(comps):
        r = find(i)
        groups[r] = groups.get(r, np.zeros_like(c)) | c
    masks = list(groups.values())
    masks.sort(key=lambda m: int(m.sum()), reverse=True)
    return masks


def mask_to_xyxy(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Tight bounding box of a boolean mask as ``(x0, y0, x1, y1)``."""
    if not mask.any():
        return (0.0, 0.0, 0.0, 0.0)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return (float(x0), float(y0), float(x1 + 1), float(y1 + 1))


def encode_segmentation(
    label_map: np.ndarray,
    palette: Sequence[ColorRGB],
    *,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Render an integer label map as a flat RGB segmentation image.

    This is the inverse of :func:`decode_semantic` and produces the **target**
    images used to instruction-tune the generator (Vision Banana's training
    signal): every pixel is filled with its class's colour, yielding a flat,
    unambiguous colour map that decodes back to the original labels.

    Args:
        label_map: ``(H, W)`` integer array; value ``i`` maps to ``palette[i]``.
            Out-of-range values and ``ignore_index`` are rendered black.
        palette: Ordered RGB colours — ``palette[i]`` is the colour for class ``i``.
        ignore_index: Optional label value to render as black (e.g. background).

    Returns:
        ``(H, W, 3)`` uint8 RGB image.
    """
    if label_map.ndim != 2:
        raise ValueError(f"Expected a 2-D label map, got shape {label_map.shape}")
    pal = np.asarray([parse_color(c) for c in palette], dtype=np.uint8)
    if pal.ndim != 2 or pal.shape[1] != 3:
        raise ValueError("palette must be a sequence of RGB colour specs")
    safe = np.clip(label_map.astype(np.int64), 0, pal.shape[0] - 1)
    rgb = pal[safe]  # (H, W, 3)
    valid = (label_map >= 0) & (label_map < pal.shape[0])
    if ignore_index is not None:
        valid = valid & (label_map != ignore_index)
    rgb = np.where(valid[..., None], rgb, 0)
    return rgb.astype(np.uint8)


def to_pil_uint8(rgb: np.ndarray) -> Image.Image:
    """Coerce an ``HxWx3`` array (any dtype) to a ``uint8`` PIL RGB image."""
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3, got {arr.shape}")
    if arr.dtype != np.uint8:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            arr = np.zeros(arr.shape, dtype=np.uint8)
        else:
            arr = ((arr - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def decode_segmentation(
    rgb: np.ndarray,
    color_map: Mapping[str, object] | None,
    *,
    instance: bool = False,
    background: object = (0, 0, 0),
    tolerance: float | None = None,
    exclude: Iterable[str] = _BACKGROUND_ALIASES,
) -> list[tuple[str, np.ndarray]]:
    """Unified decode returning ``[(label, mask), ...]``.

    * ``instance=False`` → :func:`decode_semantic` (one mask per named class,
      winner-take-all by nearest colour).
    * ``instance=True`` → :func:`decode_instances` (the paper's Appendix-A
      multi-stage clustering, one mask per instance), labelled
      ``"<class>_0"``, ``"<class>_1"``, … where the class is the single
      non-background entry of *color_map* (else ``"instance"``).
    """
    if not instance:
        tol = DEFAULT_SEMANTIC_TOLERANCE if tolerance is None else tolerance
        return decode_semantic(rgb, color_map, tolerance=tol, exclude=exclude)

    cmap = normalize_color_map(color_map)
    names = [k for k in cmap if k.lower() not in _BACKGROUND_ALIASES]
    base = names[0] if names else "instance"
    bg = background if background is not None else cmap.get("background", (0, 0, 0))
    masks = decode_instances(rgb, background=bg)
    return [(f"{base}_{i}", m) for i, m in enumerate(masks)]


# ---------------------------------------------------------------------------
# Monocular metric depth codec
# ---------------------------------------------------------------------------
#
# Vision Banana visualises metric depth (metres, [0, ∞)) as an RGB image via a
# *bijection*: a Barron power transform (arXiv:2502.10647) curves depth into
# [0, 1), which is then mapped along the edges of the RGB cube (a Hamiltonian
# path from black to white, like the first iteration of a 3-D Hilbert curve).
# Both steps are strictly invertible, so a generated RGB image decodes back to
# metric depth.  Parameters: λ = −3, c = 10/3  (λc = −10).

BARRON_LAMBDA = -3.0
BARRON_C = 10.0 / 3.0

# 8 cube corners forming a 7-edge Hamiltonian path black → … → white along the
# edges of the RGB cube (a 3D-Hilbert-style traversal). Order matches the
# paper's Fig 5 tube: black(0m)→blue(1m)→cyan(2m)→green(5m)→yellow(10m)→
# red(50m)→magenta(100m)→white(∞). Each edge changes exactly one channel, so the
# inverse projects cleanly onto one axis.
_CUBE_PATH = np.array(
    [
        [0, 0, 0],  # black   ≈ 0m   (near)
        [0, 0, 1],  # blue    ≈ 1m
        [0, 1, 1],  # cyan    ≈ 2m
        [0, 1, 0],  # green   ≈ 5m
        [1, 1, 0],  # yellow  ≈ 10m
        [1, 0, 0],  # red     ≈ 50m
        [1, 0, 1],  # magenta ≈ 100m
        [1, 1, 1],  # white   → ∞    (far)
    ],
    dtype=np.float64,
)


def power_transform_depth(
    depth: np.ndarray, lam: float = BARRON_LAMBDA, c: float = BARRON_C
) -> np.ndarray:
    """Barron power transform mapping depth ``d ≥ 0`` (metres) → ``[0, 1)``.

    ``f(d) = 1 − (1 − d/(λc))**(λ+1)``; with λ=−3, c=10/3 this is
    ``1 − (1 + d/10)**(−2)``.
    """
    d = np.asarray(depth, dtype=np.float64)
    base = 1.0 - d / (lam * c)
    return 1.0 - np.power(base, lam + 1.0)


def inverse_power_transform_depth(
    f: np.ndarray, lam: float = BARRON_LAMBDA, c: float = BARRON_C
) -> np.ndarray:
    """Inverse of :func:`power_transform_depth`; ``f ∈ [0,1)`` → depth (metres)."""
    f = np.clip(np.asarray(f, dtype=np.float64), 0.0, 1.0 - 1e-7)
    base = np.power(1.0 - f, 1.0 / (lam + 1.0))
    return (lam * c) * (1.0 - base)


def curve_to_rgb(t: np.ndarray) -> np.ndarray:
    """Map ``t ∈ [0,1]`` to an RGB point on the cube-edge path (``[0,1]³``)."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    pos = t * 7.0
    idx = np.clip(np.floor(pos).astype(int), 0, 6)
    local = pos - idx
    a: np.ndarray = _CUBE_PATH[idx]
    b: np.ndarray = _CUBE_PATH[idx + 1]
    rgb: np.ndarray = a + local[..., None] * (b - a)
    return rgb[0] if scalar else rgb


def rgb_to_curve(rgb: np.ndarray) -> np.ndarray:
    """Project an RGB point (``[0,1]³``) onto the cube-edge path → ``t ∈ [0,1]``.

    For each of the 7 edges, the nearest point clamps the edge's varying
    channel and fixes the other two; we keep the edge with the smallest
    squared distance. Accepts ``(..., 3)`` arrays (batch over leading dims).
    """
    arr = np.asarray(rgb, dtype=np.float64)
    scalar = arr.ndim == 1
    flat = arr.reshape(-1, 3) if arr.ndim >= 2 else arr[None, :]
    best_t: np.ndarray = np.zeros(len(flat))
    best_d: np.ndarray = np.full(len(flat), np.inf)
    for i in range(7):
        a = _CUBE_PATH[i]
        b = _CUBE_PATH[i + 1]
        axis = int(np.argmax(np.abs(b - a)))
        cand = np.repeat(a[None, :], len(flat), axis=0)
        v = np.clip(flat[:, axis], 0.0, 1.0)
        cand[:, axis] = v
        dist = np.sum((cand - flat) ** 2, axis=1)
        # fraction along this edge — accounts for edges where the channel
        # decreases (b[axis] < a[axis]); each edge changes one channel by 1.
        local = (v - a[axis]) / (b[axis] - a[axis])
        better = dist < best_d
        best_d = np.where(better, dist, best_d)
        best_t = np.where(better, (i + local) / 7.0, best_t)
    out = best_t if not scalar else best_t[0]
    return out.reshape(arr.shape[:-1]) if arr.ndim >= 2 else out


# Alternative depth colormaps the paper augments its training targets with, for
# robustness to diverse colour representations ("cube" is the canonical,
# invertible one and the only one ``decode_depth`` reverses).
_DEPTH_COLORMAPS = ("cube", "plasma", "inferno", "viridis", "grayscale")


def _colormap_rgb(t: np.ndarray, name: str) -> np.ndarray:
    """Map ``t ∈ [0,1]`` → RGB in ``[0,1]³`` via matplotlib colormap *name*."""
    import matplotlib as mpl

    # The paper augments with "grayscale"; matplotlib's colormap is registered
    # as "gray".
    name = {"grayscale": "gray"}.get(name, name)
    try:  # matplotlib >= 3.5
        cmap = mpl.colormaps[name]
    except (AttributeError, KeyError):  # older matplotlib
        from matplotlib import cm

        cmap = cm.get_cmap(name)
    rgba = cmap(np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0))
    return cast(np.ndarray, np.asarray(rgba)[..., :3])


def encode_depth(
    depth: np.ndarray,
    *,
    lam: float = BARRON_LAMBDA,
    c: float = BARRON_C,
    colormap: str = "cube",
) -> np.ndarray:
    """Render a metric depth map (HxW, metres ≥ 0) as an RGB image (HxWx3 uint8).

    *colormap* selects the false-color scheme: ``"cube"`` (the paper's
    invertible cube-edge tube, used for decoding) or one of the training
    augmentations ``"plasma"``/``"inferno"``/``"viridis"``/``"grayscale"``.
    """
    if colormap not in _DEPTH_COLORMAPS:
        raise ValueError(f"Unknown colormap {colormap!r}; choose from {_DEPTH_COLORMAPS}")
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"Expected an HxW depth map, got shape {d.shape}")
    f = power_transform_depth(d, lam, c)
    rgb = curve_to_rgb(f) if colormap == "cube" else _colormap_rgb(f, colormap)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def decode_depth(
    rgb: np.ndarray,
    *,
    lam: float = BARRON_LAMBDA,
    c: float = BARRON_C,
) -> np.ndarray:
    """Decode an RGB depth visualization (HxWx3) back to metric depth (HxW, metres).

    Always inverts the canonical cube-edge colormap (the model is prompted with
    the rainbow/cube scheme at inference).
    """
    rgb01 = np.asarray(rgb, dtype=np.float64) / 255.0
    f = rgb_to_curve(rgb01)
    return inverse_power_transform_depth(f, lam, c).astype(np.float32)


def unproject_depth(
    depth: np.ndarray,
    fx: float,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """Unproject a metric depth map (HxW) to a 3D point cloud ``(N, 3)``.

    Pinhole back-projection (paper Fig 6): ``X=(u-cx)·d/fx``,
    ``Y=(v-cy)·d/fy``, ``Z=d``. ``fx``/``fy`` are focal lengths (px); ``cx``,
    ``cy`` default to the image centre. Depth (the prediction itself needs no
    intrinsics; this reconstruction step does).
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"Expected an HxW depth map, got shape {d.shape}")
    h, w = d.shape
    fy = fx if fy is None else fy
    cx = w / 2.0 if cx is None else cx
    cy = h / 2.0 if cy is None else cy
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = d > 0
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    pts = np.stack([x, y, d], axis=-1)[valid]
    return pts.astype(np.float32)


def build_depth_prompt() -> str:
    """Vision-Banana-style instruction for metric-depth generation."""
    return (
        "Predict the metric depth of this scene as an image. Visualized in the "
        "rainbow (black-red-yellow-green-cyan-blue-violet-white) color palette."
    )


# ---------------------------------------------------------------------------
# Surface normal codec
# ---------------------------------------------------------------------------
#
# Camera-space unit normals (+x right, +y up, +z toward camera) map directly to
# RGB per the paper (camera-space, +x right, +y up, +z toward camera):
#   R = (1 − x)/2 ,  G = (1 + y)/2 ,  B = (1 + z)/2   (×255, truncated)
# So facing-left (−x) → pinkish red, up (+y) → green, toward camera (+z) → blue.
# Decoding inverts and renormalises (generative output drifts off the unit sphere).


def encode_normal(normal: np.ndarray) -> np.ndarray:
    """Render a camera-space normal map (HxWx3, components in [−1,1]) as RGB."""
    n = np.asarray(normal, dtype=np.float64)
    if n.ndim != 3 or n.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 normal map, got shape {n.shape}")
    rgb = np.stack(
        [
            (1.0 - n[..., 0]) / 2.0,  # R from x (sign-flipped)
            (1.0 + n[..., 1]) / 2.0,  # G from y
            (1.0 + n[..., 2]) / 2.0,  # B from z
        ],
        axis=-1,
    )
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def decode_normal(rgb: np.ndarray) -> np.ndarray:
    """Decode an RGB normal map (HxWx3) to camera-space unit normals (HxWx3)."""
    rgb01 = np.asarray(rgb, dtype=np.float64) / 255.0
    n = np.stack(
        [
            1.0 - 2.0 * rgb01[..., 0],  # x from R
            2.0 * rgb01[..., 1] - 1.0,  # y from G
            2.0 * rgb01[..., 2] - 1.0,  # z from B
        ],
        axis=-1,
    )
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-6, 1.0, norm)
    return cast(np.ndarray, (n / norm).astype(np.float32))


def build_normal_prompt() -> str:
    """Vision-Banana-style instruction for surface-normal generation."""
    return "Generate a surface normal map of the input image."


__all__ = [
    "BARRON_C",
    "BARRON_LAMBDA",
    "ColorMap",
    "ColorRGB",
    "build_depth_prompt",
    "build_normal_prompt",
    "build_segmentation_prompt",
    "color_distance",
    "curve_to_rgb",
    "decode_depth",
    "decode_instances",
    "decode_normal",
    "decode_segmentation",
    "decode_semantic",
    "encode_depth",
    "encode_normal",
    "encode_segmentation",
    "inverse_power_transform_depth",
    "mask_to_xyxy",
    "normalize_color_map",
    "parse_color",
    "power_transform_depth",
    "rgb_to_curve",
    "to_pil_uint8",
    "unproject_depth",
]
