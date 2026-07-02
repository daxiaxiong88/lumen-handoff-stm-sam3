"""Unit tests for the Vision Banana RGB segmentation codecs.

These cover the model-independent core (prompt building + colour-based mask
decoding) without touching FLUX.2-klein-4B or ``diffusers`` — they are the
scientific heart of the reproduction and must be exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumen.models.vision_banana.codecs import (
    DEFAULT_SEMANTIC_TOLERANCE,
    build_depth_prompt,
    build_normal_prompt,
    build_segmentation_prompt,
    color_distance,
    curve_to_rgb,
    decode_depth,
    decode_instances,
    decode_normal,
    decode_segmentation,
    decode_semantic,
    encode_depth,
    encode_normal,
    encode_segmentation,
    inverse_power_transform_depth,
    mask_to_xyxy,
    normalize_color_map,
    parse_color,
    power_transform_depth,
    rgb_to_curve,
    unproject_depth,
)

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


# ---------------------------------------------------------------------------
# Colour parsing / normalisation
# ---------------------------------------------------------------------------


class TestParseColor:
    def test_int_tuple(self) -> None:
        assert parse_color((10, 20, 30)) == (10, 20, 30)

    def test_float_tuple_0_to_1(self) -> None:
        assert parse_color((1.0, 0.0, 0.0)) == (255, 0, 0)

    def test_list(self) -> None:
        assert parse_color([1, 2, 3]) == (1, 2, 3)

    def test_hex(self) -> None:
        assert parse_color("#ff8000") == (255, 128, 0)

    def test_named(self) -> None:
        # "red" parses via PIL.ImageColor
        assert parse_color("red") == (255, 0, 0)

    def test_rejects_bad_length(self) -> None:
        with pytest.raises(ValueError, match="3 channels"):
            parse_color((1, 2))

    def test_rejects_bad_spec(self) -> None:
        with pytest.raises(ValueError):
            parse_color(object())


class TestNormalizeColorMap:
    def test_coerces_values(self) -> None:
        out = normalize_color_map({"a": (1, 2, 3), "b": "#ffffff"})
        assert out == {"a": (1, 2, 3), "b": (255, 255, 255)}

    def test_empty(self) -> None:
        assert normalize_color_map(None) == {}
        assert normalize_color_map({}) == {}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestBuildSegmentationPrompt:
    def test_semantic_json_mapping(self) -> None:
        prompt = build_segmentation_prompt({"cell": GREEN, "background": BLACK})
        assert "semantic segmentation" in prompt
        assert '"cell": <0, 255, 0>' in prompt
        assert '"background": <0, 0, 0>' in prompt

    def test_instance_prompt(self) -> None:
        prompt = build_segmentation_prompt({"cell": GREEN}, instance=True)
        assert "instance segmentation visualization" in prompt
        assert "cell is colored differently" in prompt
        assert "Background is RGB(0, 0, 0)" in prompt

    def test_instance_explicit_background(self) -> None:
        prompt = build_segmentation_prompt(
            {"cell": GREEN}, instance=True, background=BLUE
        )
        assert "Background is RGB(0, 0, 255)" in prompt


# ---------------------------------------------------------------------------
# colour distance
# ---------------------------------------------------------------------------


class TestColorDistance:
    def test_zero_distance(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        assert color_distance(rgb, BLACK).shape == (4, 4)
        assert np.allclose(color_distance(rgb, BLACK), 0.0)

    def test_pure_red_to_black(self) -> None:
        rgb = np.full((2, 2, 3), 255, dtype=np.uint8)
        rgb[..., 1:] = 0
        assert np.allclose(color_distance(rgb, BLACK), 255.0)


# ---------------------------------------------------------------------------
# Semantic decode
# ---------------------------------------------------------------------------


def _halves_image(h: int = 16, w: int = 16) -> np.ndarray:
    """Left half green, right half red, uint8 HxWx3."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2] = GREEN
    img[:, w // 2 :] = RED
    return img


class TestDecodeSemantic:
    def test_recovers_two_classes(self) -> None:
        rgb = _halves_image()
        decoded = dict(
            decode_semantic(rgb, {"green": GREEN, "red": RED, "background": BLACK})
        )
        assert set(decoded) == {"green", "red"}
        assert decoded["green"].sum() == 16 * 8
        assert decoded["red"].sum() == 16 * 8
        # masks are mutually exclusive and cover the whole image
        assert not (decoded["green"] & decoded["red"]).any()

    def test_background_is_excluded(self) -> None:
        rgb = _halves_image()
        decoded = decode_semantic(rgb, {"background": BLACK})
        assert decoded == []

    def test_empty_class_is_skipped(self) -> None:
        rgb = _halves_image()
        decoded = decode_semantic(
            rgb, {"green": GREEN, "blue": BLUE, "background": BLACK}
        )
        assert [name for name, _ in decoded] == ["green"]

    def test_tolerance_controls_strictness(self) -> None:
        rgb = np.full((4, 4, 3), 0, dtype=np.uint8)
        rgb[..., 0] = 240  # dark red, Euclidean distance 15 from pure red (255,0,0)
        loose = decode_semantic(rgb, {"red": RED}, tolerance=DEFAULT_SEMANTIC_TOLERANCE)
        strict = decode_semantic(rgb, {"red": RED}, tolerance=10.0)
        assert len(loose) == 1 and loose[0][1].all()
        assert strict == []

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="HxWx3"):
            color_distance(np.zeros((4, 4), dtype=np.uint8), BLACK)


# ---------------------------------------------------------------------------
# Instance decode
# ---------------------------------------------------------------------------


def _blobs_image(h: int = 32, w: int = 32) -> np.ndarray:
    """Two disjoint squares of DIFFERENT colours on a black background.

    The paper's instance algorithm merges same-colour regions, so realistic
    instances must have distinct colours (as the model assigns them).
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[4:12, 4:12] = GREEN          # instance A
    img[20:28, 20:28] = RED          # instance B (different colour)
    return img


class TestDecodeInstances:
    def test_finds_two_instances(self) -> None:
        masks = decode_instances(_blobs_image(), background=BLACK)
        assert len(masks) == 2
        assert not (masks[0] & masks[1]).any()  # disjoint
        total = masks[0].sum() + masks[1].sum()
        assert total == 8 * 8 * 2

    def test_different_colours_not_merged(self) -> None:
        # adjacent different-colour blocks must stay separate (the bug the
        # paper's colour-seeded flood-fill fixes vs. plain connected-components)
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[:, :8] = GREEN
        img[:, 8:] = RED
        masks = decode_instances(img, background=BLACK)
        assert len(masks) == 2  # would be 1 with naive foreground CC

    def test_same_colour_separate_regions_can_merge(self) -> None:
        # two disjoint same-colour blobs within the bbox-expansion criterion are
        # merged by the paper's spatially-constrained merging step
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[2:6, 2:6] = GREEN
        img[2:6, 8:12] = GREEN
        masks = decode_instances(img, background=BLACK)
        assert len(masks) == 1

    def test_ordered_largest_first(self) -> None:
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[2:6, 2:6] = GREEN        # 16 px
        img[10:30, 10:30] = RED      # 400 px (different colour)
        masks = decode_instances(img, background=BLACK)
        assert masks[0].sum() > masks[1].sum()

    def test_min_area_filters_small(self) -> None:
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[2:5, 2:5] = GREEN         # 9 px
        img[10:30, 10:30] = RED       # 400 px
        # default min_area_frac=2e-4 -> ~0 px threshold: both survive
        assert len(decode_instances(img, background=BLACK)) == 2
        # raise the floor to prune the small one (10% of 1600 = 160)
        assert len(decode_instances(img, background=BLACK, min_area_frac=0.1)) == 1

    def test_no_foreground_returns_empty(self) -> None:
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        assert decode_instances(img, background=BLACK) == []


# ---------------------------------------------------------------------------
# Unified decode + mask_to_xyxy
# ---------------------------------------------------------------------------


class TestDecodeSegmentation:
    def test_routes_to_semantic(self) -> None:
        rgb = _halves_image()
        decoded = decode_segmentation(
            rgb, {"green": GREEN, "red": RED, "background": BLACK}
        )
        assert [n for n, _ in decoded] == ["green", "red"]

    def test_routes_to_instances(self) -> None:
        rgb = _blobs_image()
        decoded = decode_segmentation(
            rgb, {"cell": GREEN}, instance=True, background=BLACK
        )
        assert len(decoded) == 2
        assert all(name.startswith("cell_") for name, _ in decoded)


class TestMaskToXYXY:
    def test_tight_box(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 3:7] = True
        x0, y0, x1, y1 = mask_to_xyxy(mask)
        assert (x0, y0, x1, y1) == (3.0, 2.0, 7.0, 5.0)

    def test_empty_returns_zeros(self) -> None:
        assert mask_to_xyxy(np.zeros((5, 5), dtype=bool)) == (0.0, 0.0, 0.0, 0.0)


class TestEncodeSegmentation:
    def test_renders_flat_colours(self) -> None:
        label_map = np.zeros((8, 8), dtype=int)
        label_map[:4] = 1  # green
        label_map[4:] = 2  # red
        rgb = encode_segmentation(label_map, [BLACK, GREEN, RED])
        assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
        assert np.all(rgb[:4] == GREEN)
        assert np.all(rgb[4:] == RED)

    def test_round_trips_with_decode(self) -> None:
        label_map = np.zeros((16, 16), dtype=int)
        label_map[:, :8] = 1  # green
        label_map[:, 8:] = 2  # red
        palette = [BLACK, GREEN, RED]
        rgb = encode_segmentation(label_map, palette)
        decoded = dict(
            decode_semantic(
                rgb, {"background": BLACK, "green": GREEN, "red": RED}
            )
        )
        # decoded masks recover the original label regions
        assert set(decoded) == {"green", "red"}
        assert np.array_equal(decoded["green"], label_map == 1)
        assert np.array_equal(decoded["red"], label_map == 2)

    def test_out_of_range_and_ignore_render_black(self) -> None:
        label_map = np.array([[0, 1], [2, 255]])  # 255 out of range
        rgb = encode_segmentation(label_map, [BLACK, GREEN, RED])
        assert np.all(rgb[1, 1] == 0)  # out-of-range -> black
        rgb_ign = encode_segmentation(label_map, [BLACK, GREEN, RED], ignore_index=1)
        assert np.all(rgb_ign[0, 1] == 0)  # ignored class -> black

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D label map"):
            encode_segmentation(np.zeros((2, 2, 2), dtype=int), [BLACK, GREEN])


# ---------------------------------------------------------------------------
# Depth codec (power transform + cube-edge bijection)
# ---------------------------------------------------------------------------


class TestPowerTransform:
    def test_monotonic_and_bounded(self) -> None:
        d = np.array([0.0, 0.1, 1.0, 10.0, 100.0, 1000.0])
        f = power_transform_depth(d)
        assert f[0] == 0.0
        assert np.all(np.diff(f) > 0)  # monotonically increasing
        assert np.all(f < 1.0) and np.all(f >= 0.0)

    def test_round_trip(self) -> None:
        d = np.array([0.0, 0.5, 1.0, 2.5, 7.3, 15.0, 42.0, 200.0])
        rec = inverse_power_transform_depth(power_transform_depth(d))
        assert np.allclose(rec, d, atol=1e-6)


class TestCubeEdge:
    def test_endpoints(self) -> None:
        assert np.allclose(curve_to_rgb(0.0), [0, 0, 0])
        assert np.allclose(curve_to_rgb(1.0), [1, 1, 1])

    def test_round_trip_dense(self) -> None:
        t = np.linspace(0.0, 1.0, 1000)
        rec = rgb_to_curve(curve_to_rgb(t))
        assert np.allclose(rec, t, atol=1e-9)

    def test_round_trip_image(self) -> None:
        t = np.random.RandomState(0).rand(16, 16)
        rec = rgb_to_curve(curve_to_rgb(t))
        assert rec.shape == (16, 16)
        assert np.allclose(rec, t, atol=1e-9)


class TestDepthCodec:
    def test_encode_decode_round_trip(self) -> None:
        rng = np.random.RandomState(0)
        depth = rng.uniform(0.2, 30.0, (32, 32))
        rgb = encode_depth(depth)
        assert rgb.shape == (32, 32, 3) and rgb.dtype == np.uint8
        rec = decode_depth(rgb)
        # quantisation to 8-bit + cube-edge projection; allow a few % error
        rel_err = np.abs(rec - depth) / np.maximum(depth, 1e-3)
        assert np.mean(rel_err) < 0.05, f"mean rel err {np.mean(rel_err):.3f}"

    def test_near_far_ordering(self) -> None:
        # a near surface should decode smaller than a far one
        depth = np.array([[1.0, 10.0]])
        rec = decode_depth(encode_depth(depth))
        assert rec[0, 0] < rec[0, 1]

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="HxW depth map"):
            encode_depth(np.zeros((2, 2, 2)))

    def test_augmentation_colormaps(self) -> None:
        depth = np.random.RandomState(0).uniform(0.5, 10.0, (8, 8))
        cube = encode_depth(depth)
        for cm in ("plasma", "inferno", "viridis", "grayscale"):
            alt = encode_depth(depth, colormap=cm)
            assert alt.shape == (8, 8, 3) and alt.dtype == np.uint8
        assert not np.array_equal(cube, encode_depth(depth, colormap="plasma"))

    def test_prompt(self) -> None:
        assert "metric depth" in build_depth_prompt()


class TestUnprojectDepth:
    def test_plane_shape(self) -> None:
        depth = np.full((10, 20), 5.0)
        pts = unproject_depth(depth, fx=100.0)
        assert pts.shape == (200, 3) and pts.dtype == np.float32
        assert np.allclose(pts[:, 2], 5.0)  # Z = depth

    def test_center_points_forward(self) -> None:
        depth = np.full((4, 4), 2.0)
        pts = unproject_depth(depth, fx=50.0, fy=50.0)
        # the principal point (image centre) should map to (0, 0, d)
        center = pts[np.argmin(np.linalg.norm(pts[:, :2], axis=1))]
        assert np.allclose(center[:2], 0.0, atol=0.3)
        assert np.isclose(center[2], 2.0)


# ---------------------------------------------------------------------------
# Normal codec
# ---------------------------------------------------------------------------


class TestNormalCodec:
    def _unit(self, n):
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        return n

    def test_encode_decode_round_trip(self) -> None:
        rng = np.random.RandomState(0)
        raw = rng.uniform(-1.0, 1.0, (32, 32, 3))
        normals = self._unit(raw)
        rgb = encode_normal(normals)
        assert rgb.shape == (32, 32, 3) and rgb.dtype == np.uint8
        rec = decode_normal(rgb)
        # unit length
        assert np.allclose(np.linalg.norm(rec, axis=-1), 1.0, atol=1e-3)
        # angular error small (8-bit quantisation)
        dot = np.sum(rec * normals, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        deg = np.degrees(np.arccos(dot))
        assert np.mean(deg) < 1.5, f"mean angle err {np.mean(deg):.2f}°"

    def test_axis_colours(self) -> None:
        # Paper convention: R=(1-x)/2, G=(1+y)/2, B=(1+z)/2.
        # +x right -> R=0; +y up -> G=255; +z toward camera -> B=255.
        n = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        rgb = encode_normal(n)[0]
        assert np.allclose(rgb[0], [0, 127, 127], atol=1)     # +x right  -> R low
        assert np.allclose(rgb[1], [127, 255, 127], atol=1)   # +y up     -> G high
        assert np.allclose(rgb[2], [127, 127, 255], atol=1)   # +z camera -> B high
        # facing left (-x) -> pinkish red (paper: "Facing Left = Pinkish Red")
        n_left = np.array([[[-1.0, 0.0, 0.0]]])  # (1,1,3)
        assert encode_normal(n_left)[0, 0, 0] > 200

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="HxWx3 normal map"):
            encode_normal(np.zeros((4, 4)))

    def test_prompt(self) -> None:
        assert "surface normal" in build_normal_prompt()
