"""Composable tracking presets for different video scenarios.

This module provides a layered preset system where multiple concerns can be
combined. Presets are organized into categories (platform, scene, duration,
modality) and can be stacked to create custom configurations.

Usage:
    from tracking_presets import compose_presets, apply_presets

    # Single preset
    preset = compose_presets(["uav"])

    # Composable: combine multiple concerns
    preset = compose_presets(["uav", "fast_motion", "long_video"])

    # Apply to environment via CLI
    # python cli.py --preset uav+fast_motion+long_video
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# LAYER SYSTEM
# =============================================================================

@dataclass
class TrackingLayer:
    """A single layer of tracking configuration adjustments.

    Each field can be None (no change) or a value/delta to apply.
    Layers are applied in order, with later layers overriding earlier ones.

    For numeric fields:
        - Absolute values (float/int) replace the current value
        - Strings like "+0.1" or "-50" apply relative adjustments
        - Strings like "*1.5" apply multiplicative adjustments
    """

    name: str
    description: str
    category: str  # platform, scene, duration, modality, quality

    # Detection thresholds (Grounding DINO)
    box_threshold: Optional[Union[float, str]] = None
    text_threshold: Optional[Union[float, str]] = None
    model_score_threshold: Optional[Union[float, str]] = None

    # Tracker parameters (ByteTrack)
    track_activation_threshold: Optional[Union[float, str]] = None
    lost_track_buffer: Optional[Union[int, str]] = None
    minimum_matching_threshold: Optional[Union[float, str]] = None
    minimum_consecutive_frames: Optional[Union[int, str]] = None


@dataclass
class TrackingPreset:
    """Final composed tracking configuration.

    This is the result of composing multiple layers together.
    """

    name: str
    description: str
    layers: List[str] = field(default_factory=list)

    # Detection thresholds
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    model_score_threshold: float = 0.50

    # Tracker parameters
    track_activation_threshold: float = 0.30
    lost_track_buffer: int = 100
    minimum_matching_threshold: float = 0.40
    minimum_consecutive_frames: int = 8


# Default baseline values
BASELINE = TrackingPreset(
    name="baseline",
    description="Default balanced settings",
    box_threshold=0.25,
    text_threshold=0.25,
    model_score_threshold=0.50,
    track_activation_threshold=0.30,
    lost_track_buffer=100,
    minimum_matching_threshold=0.40,
    minimum_consecutive_frames=8,
)


# =============================================================================
# LAYER DEFINITIONS - Organized by category
# =============================================================================

TRACKING_LAYERS: Dict[str, TrackingLayer] = {
    # =========================================================================
    # PLATFORM LAYERS - Camera/vehicle type
    # =========================================================================
    "uav": TrackingLayer(
        name="uav",
        description="Aerial/drone: small subjects, fast relative motion",
        category="platform",
        # Lower thresholds to catch small/distant objects
        box_threshold=0.20,
        text_threshold=0.20,
        model_score_threshold=0.40,
        # Lower activation threshold - small objects have lower confidence
        track_activation_threshold=0.25,
        # Longer buffer for temporary occlusions during maneuvers
        lost_track_buffer=150,
        # Lower IoU threshold - objects move fast between frames
        minimum_matching_threshold=0.20,
        # Fewer consecutive frames - don't wait too long to confirm
        minimum_consecutive_frames=5,
    ),

    "ugv": TrackingLayer(
        name="ugv",
        description="Ground vehicle: larger subjects, stable tracking",
        category="platform",
        box_threshold=0.25,
        text_threshold=0.25,
        model_score_threshold=0.50,
        track_activation_threshold=0.30,
        lost_track_buffer=150,
        minimum_matching_threshold=0.40,
        minimum_consecutive_frames=8,
    ),

    "handheld": TrackingLayer(
        name="handheld",
        description="Handheld camera: moderate shake, variable framing",
        category="platform",
        box_threshold=0.25,
        text_threshold=0.25,
        minimum_matching_threshold=0.35,
        lost_track_buffer=120,
    ),

    "fixed": TrackingLayer(
        name="fixed",
        description="Fixed/stationary camera: stable background",
        category="platform",
        minimum_matching_threshold=0.50,
        lost_track_buffer=80,
        minimum_consecutive_frames=6,
    ),

    # =========================================================================
    # SCENE LAYERS - Scene characteristics
    # =========================================================================
    "crowded": TrackingLayer(
        name="crowded",
        description="Many subjects, frequent occlusions",
        category="scene",
        box_threshold="+0.10",  # Increase to reduce overlapping detections
        model_score_threshold="+0.10",
        track_activation_threshold="+0.15",
        lost_track_buffer="*1.5",  # 50% longer buffer for occlusions
        minimum_consecutive_frames="+7",
    ),

    "sparse": TrackingLayer(
        name="sparse",
        description="Few subjects, minimal occlusion",
        category="scene",
        track_activation_threshold="-0.05",
        lost_track_buffer="*0.7",
        minimum_consecutive_frames="-2",
    ),

    "cluttered": TrackingLayer(
        name="cluttered",
        description="Complex background, potential false positives",
        category="scene",
        box_threshold="+0.05",
        model_score_threshold="+0.05",
        track_activation_threshold="+0.10",
        minimum_consecutive_frames="+5",
    ),

    # =========================================================================
    # MOTION LAYERS - Subject motion characteristics
    # =========================================================================
    "fast_motion": TrackingLayer(
        name="fast_motion",
        description="High-speed subjects, large frame-to-frame displacement",
        category="motion",
        box_threshold="-0.05",  # Lower to catch motion blur
        text_threshold="-0.05",
        minimum_matching_threshold=0.15,  # Very permissive IoU
        minimum_consecutive_frames=5,
    ),

    "slow_motion": TrackingLayer(
        name="slow_motion",
        description="Slow-moving subjects, predictable trajectories",
        category="motion",
        minimum_matching_threshold=0.55,  # Strict IoU
        lost_track_buffer="*0.6",
    ),

    "erratic": TrackingLayer(
        name="erratic",
        description="Unpredictable motion patterns",
        category="motion",
        minimum_matching_threshold="-0.10",
        lost_track_buffer="+50",
    ),

    # =========================================================================
    # DURATION LAYERS - Video length considerations
    # =========================================================================
    "long_video": TrackingLayer(
        name="long_video",
        description="Long videos (>10 min), minimize fragmentation",
        category="duration",
        # DON'T raise detection thresholds - causes missed detections
        # Instead, focus on tracker parameters to reduce fragmentation
        lost_track_buffer="*2.0",  # Double the buffer for re-identification
        minimum_consecutive_frames="+3",  # Slightly more frames to confirm (not +12)
        # Lower matching threshold to allow re-association after gaps
        minimum_matching_threshold="-0.05",
    ),

    "short_clip": TrackingLayer(
        name="short_clip",
        description="Short clips (<1 min), prioritize completeness",
        category="duration",
        track_activation_threshold="-0.05",
        lost_track_buffer="*0.5",
        minimum_consecutive_frames="-3",
    ),

    # =========================================================================
    # MODALITY LAYERS - Sensor/image type
    # =========================================================================
    "thermal": TrackingLayer(
        name="thermal",
        description="Thermal/IR imagery, lower contrast",
        category="modality",
        box_threshold=0.20,
        text_threshold=0.20,
        model_score_threshold=0.45,
        minimum_matching_threshold=0.30,
        lost_track_buffer=200,
    ),

    "lowlight": TrackingLayer(
        name="lowlight",
        description="Low-light conditions, noisy detections",
        category="modality",
        box_threshold="-0.05",
        track_activation_threshold="+0.10",
        minimum_consecutive_frames="+5",
    ),

    "hdr": TrackingLayer(
        name="hdr",
        description="High dynamic range, good visibility",
        category="modality",
        box_threshold="-0.03",
        minimum_consecutive_frames="-2",
    ),

    # =========================================================================
    # QUALITY LAYERS - Detection quality tuning
    # =========================================================================
    "high_precision": TrackingLayer(
        name="high_precision",
        description="Prioritize precision over recall",
        category="quality",
        box_threshold="+0.15",
        model_score_threshold="+0.15",
        track_activation_threshold="+0.15",
        minimum_consecutive_frames="+10",
    ),

    "high_recall": TrackingLayer(
        name="high_recall",
        description="Prioritize recall over precision",
        category="quality",
        box_threshold="-0.10",
        model_score_threshold="-0.10",
        track_activation_threshold="-0.10",
        minimum_consecutive_frames="-3",
    ),
}

# Legacy compatibility: full presets as single-layer compositions
TRACKING_PRESETS: Dict[str, TrackingPreset] = {}  # Populated by compose_presets


@dataclass
class ParameterBounds:
    """Valid bounds for a tracking parameter."""
    min_val: float
    max_val: float
    description: str


# Parameter validation bounds
PARAMETER_BOUNDS: Dict[str, ParameterBounds] = {
    "box_threshold": ParameterBounds(0.05, 0.95, "detection box confidence"),
    "text_threshold": ParameterBounds(0.05, 0.95, "text-to-region matching"),
    "model_score_threshold": ParameterBounds(0.05, 0.95, "final prediction score"),
    "track_activation_threshold": ParameterBounds(0.05, 0.95, "track activation confidence"),
    "lost_track_buffer": ParameterBounds(1, 1800, "frames to keep lost track (1-60s at 30fps)"),
    "minimum_matching_threshold": ParameterBounds(0.05, 0.95, "IoU for track matching"),
    "minimum_consecutive_frames": ParameterBounds(1, 100, "frames before track confirmation"),
}


def _apply_adjustment(
    current: Union[float, int],
    adjustment: Union[float, int, str, None],
    is_int: bool = False,
) -> Union[float, int]:
    """Apply an adjustment to a current value.

    Args:
        current: Current value
        adjustment: Absolute value, or string like "+0.1", "-50", "*1.5"
        is_int: Whether the result should be an integer

    Returns:
        Adjusted value (not yet validated against bounds)
    """
    if adjustment is None:
        return current

    if isinstance(adjustment, str):
        adjustment = adjustment.strip()
        if adjustment.startswith("+"):
            result = current + float(adjustment[1:])
        elif adjustment.startswith("-"):
            result = current - float(adjustment[1:])
        elif adjustment.startswith("*"):
            result = current * float(adjustment[1:])
        else:
            result = float(adjustment)
    else:
        result = adjustment

    if is_int:
        result = int(round(result))
    else:
        result = float(result)

    return result


def validate_preset(preset: TrackingPreset) -> List[str]:
    """Validate a preset's parameters against defined bounds.

    Args:
        preset: TrackingPreset to validate.

    Returns:
        List of warning messages for out-of-bounds values.
    """
    warnings = []

    param_values = {
        "box_threshold": preset.box_threshold,
        "text_threshold": preset.text_threshold,
        "model_score_threshold": preset.model_score_threshold,
        "track_activation_threshold": preset.track_activation_threshold,
        "lost_track_buffer": preset.lost_track_buffer,
        "minimum_matching_threshold": preset.minimum_matching_threshold,
        "minimum_consecutive_frames": preset.minimum_consecutive_frames,
    }

    for param_name, value in param_values.items():
        bounds = PARAMETER_BOUNDS.get(param_name)
        if bounds is None:
            continue

        if value < bounds.min_val:
            warnings.append(
                f"{param_name}={value} is below minimum {bounds.min_val} "
                f"({bounds.description})"
            )
        elif value > bounds.max_val:
            warnings.append(
                f"{param_name}={value} is above maximum {bounds.max_val} "
                f"({bounds.description})"
            )

    return warnings


def clamp_preset(preset: TrackingPreset) -> TrackingPreset:
    """Clamp a preset's parameters to valid bounds.

    Args:
        preset: TrackingPreset to clamp.

    Returns:
        New TrackingPreset with clamped values.
    """
    def clamp(value: Union[float, int], param_name: str, is_int: bool = False) -> Union[float, int]:
        bounds = PARAMETER_BOUNDS.get(param_name)
        if bounds is None:
            return value
        clamped = max(bounds.min_val, min(bounds.max_val, value))
        return int(round(clamped)) if is_int else clamped

    return TrackingPreset(
        name=preset.name,
        description=preset.description,
        layers=preset.layers,
        box_threshold=clamp(preset.box_threshold, "box_threshold"),
        text_threshold=clamp(preset.text_threshold, "text_threshold"),
        model_score_threshold=clamp(preset.model_score_threshold, "model_score_threshold"),
        track_activation_threshold=clamp(preset.track_activation_threshold, "track_activation_threshold"),
        lost_track_buffer=clamp(preset.lost_track_buffer, "lost_track_buffer", is_int=True),
        minimum_matching_threshold=clamp(preset.minimum_matching_threshold, "minimum_matching_threshold"),
        minimum_consecutive_frames=clamp(preset.minimum_consecutive_frames, "minimum_consecutive_frames", is_int=True),
    )


def compose_presets(layer_names: List[str]) -> TrackingPreset:
    """Compose multiple layers into a single TrackingPreset.

    Layers are applied in order. Later layers override or adjust values
    from earlier layers.

    Args:
        layer_names: List of layer names to compose (e.g., ["uav", "fast_motion", "long_video"])

    Returns:
        Composed TrackingPreset

    Raises:
        ValueError: If any layer name is unknown
    """
    # Start with baseline
    result = TrackingPreset(
        name="+".join(layer_names),
        description="",
        layers=list(layer_names),
        box_threshold=BASELINE.box_threshold,
        text_threshold=BASELINE.text_threshold,
        model_score_threshold=BASELINE.model_score_threshold,
        track_activation_threshold=BASELINE.track_activation_threshold,
        lost_track_buffer=BASELINE.lost_track_buffer,
        minimum_matching_threshold=BASELINE.minimum_matching_threshold,
        minimum_consecutive_frames=BASELINE.minimum_consecutive_frames,
    )

    descriptions = []

    for name in layer_names:
        key = name.lower().strip()
        if key not in TRACKING_LAYERS:
            available = ", ".join(sorted(TRACKING_LAYERS.keys()))
            raise ValueError(f"Unknown layer '{name}'. Available: {available}")

        layer = TRACKING_LAYERS[key]
        descriptions.append(layer.description)

        # Apply each field from the layer
        result.box_threshold = _apply_adjustment(
            result.box_threshold, layer.box_threshold
        )
        result.text_threshold = _apply_adjustment(
            result.text_threshold, layer.text_threshold
        )
        result.model_score_threshold = _apply_adjustment(
            result.model_score_threshold, layer.model_score_threshold
        )
        result.track_activation_threshold = _apply_adjustment(
            result.track_activation_threshold, layer.track_activation_threshold
        )
        result.lost_track_buffer = _apply_adjustment(
            result.lost_track_buffer, layer.lost_track_buffer, is_int=True
        )
        result.minimum_matching_threshold = _apply_adjustment(
            result.minimum_matching_threshold, layer.minimum_matching_threshold
        )
        result.minimum_consecutive_frames = _apply_adjustment(
            result.minimum_consecutive_frames, layer.minimum_consecutive_frames, is_int=True
        )

    result.description = " + ".join(descriptions)

    # Validate and warn about out-of-bounds values
    warnings = validate_preset(result)
    for warning in warnings:
        logger.warning("Preset '%s': %s (will be clamped)", result.name, warning)

    # Clamp to valid bounds
    result = clamp_preset(result)

    return result


def parse_preset_string(preset_str: str) -> List[str]:
    """Parse a preset string into layer names.

    Supports formats:
        - "uav" -> ["uav"]
        - "uav+fast_motion" -> ["uav", "fast_motion"]
        - "uav,thermal,long_video" -> ["uav", "thermal", "long_video"]

    Args:
        preset_str: Preset specification string

    Returns:
        List of layer names
    """
    # Support both + and , as separators
    if "+" in preset_str:
        parts = preset_str.split("+")
    elif "," in preset_str:
        parts = preset_str.split(",")
    else:
        parts = [preset_str]

    return [p.strip().lower() for p in parts if p.strip()]


def get_preset(name: str) -> TrackingPreset:
    """Get a tracking preset by name or composition.

    Supports both single layers and composed presets:
        - "uav" -> single layer
        - "uav+fast_motion+long_video" -> composed layers

    Args:
        name: Preset name or composition string (case-insensitive).

    Returns:
        TrackingPreset configuration.

    Raises:
        ValueError: If any layer name is not found.
    """
    layer_names = parse_preset_string(name)
    return compose_presets(layer_names)


def apply_preset(name: str, override_env: bool = True) -> TrackingPreset:
    """Apply a tracking preset by setting environment variables.

    Supports composable presets like "uav+fast_motion+long_video".

    Args:
        name: Preset name or composition (e.g., "uav+thermal+long_video").
        override_env: If True, override existing environment variables.
            If False, only set variables that are not already set.

    Returns:
        The applied TrackingPreset.
    """
    preset = get_preset(name)

    env_mapping = {
        "GROUNDING_DINO_BOX_THRESHOLD": str(preset.box_threshold),
        "GROUNDING_DINO_TEXT_THRESHOLD": str(preset.text_threshold),
        "MODEL_SCORE_THRESHOLD": str(preset.model_score_threshold),
        "track_activation_threshold": str(preset.track_activation_threshold),
        "lost_track_buffer": str(preset.lost_track_buffer),
        "minimum_matching_threshold": str(preset.minimum_matching_threshold),
        "minimum_consecutive_frames": str(preset.minimum_consecutive_frames),
    }

    for env_name, value in env_mapping.items():
        if override_env or os.getenv(env_name) is None:
            os.environ[env_name] = value
            logger.debug("Set %s=%s", env_name, value)

    logger.info(
        "Applied tracking preset '%s': %s",
        preset.name,
        preset.description,
    )
    return preset


def get_tracker_kwargs_from_preset(preset: TrackingPreset) -> Dict[str, Union[float, int]]:
    """Convert a preset to tracker kwargs dict for direct use with ByteTrack.

    Args:
        preset: TrackingPreset instance.

    Returns:
        Dict suitable for passing to sv.ByteTrack(**kwargs).
    """
    return {
        "track_activation_threshold": preset.track_activation_threshold,
        "lost_track_buffer": preset.lost_track_buffer,
        "minimum_matching_threshold": preset.minimum_matching_threshold,
        "minimum_consecutive_frames": preset.minimum_consecutive_frames,
    }


def list_presets() -> str:
    """Return a formatted string listing all available layers by category."""
    categories = {}
    for name, layer in TRACKING_LAYERS.items():
        cat = layer.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, layer.description))

    lines = [
        "Composable Tracking Layers",
        "=" * 60,
        "",
        "Layers can be combined with '+' separator:",
        "  --preset uav+fast_motion+long_video",
        "",
    ]

    category_order = ["platform", "scene", "motion", "duration", "modality", "quality"]
    for cat in category_order:
        if cat not in categories:
            continue
        lines.append(f"{cat.upper()}:")
        for name, desc in sorted(categories[cat]):
            lines.append(f"  {name:15} - {desc}")
        lines.append("")

    lines.extend([
        "Example Compositions:",
        "  uav+long_video           - Aerial footage, long video",
        "  uav+thermal+fast_motion  - Thermal drone, fast targets",
        "  ugv+crowded+high_precision - Ground vehicle, crowded scene",
        "  fixed+slow_motion        - Surveillance camera",
    ])

    return "\n".join(lines)


def describe_preset(name: str) -> str:
    """Return a detailed description of a preset or composition.

    Args:
        name: Preset name or composition string.

    Returns:
        Formatted string with all parameter values and validation status.
    """
    preset = get_preset(name)

    def format_param(param_name: str, value: Union[float, int], is_int: bool = False) -> str:
        bounds = PARAMETER_BOUNDS.get(param_name)
        if bounds:
            bounds_str = f"[{bounds.min_val}-{bounds.max_val}]"
        else:
            bounds_str = ""
        if is_int:
            return f"{value:4d} {bounds_str}"
        else:
            return f"{value:.2f} {bounds_str}"

    lines = [
        f"Preset: {preset.name}",
        f"Description: {preset.description}",
        f"Layers: {', '.join(preset.layers)}",
        "",
        "Detection Parameters:                    Value   Valid Range",
        f"  box_threshold:                         {format_param('box_threshold', preset.box_threshold)}",
        f"  text_threshold:                        {format_param('text_threshold', preset.text_threshold)}",
        f"  model_score_threshold:                 {format_param('model_score_threshold', preset.model_score_threshold)}",
        "",
        "Tracking Parameters:",
        f"  track_activation_threshold:            {format_param('track_activation_threshold', preset.track_activation_threshold)}",
        f"  lost_track_buffer:                     {format_param('lost_track_buffer', preset.lost_track_buffer, is_int=True)} frames",
        f"  minimum_matching_threshold:            {format_param('minimum_matching_threshold', preset.minimum_matching_threshold)}",
        f"  minimum_consecutive_frames:            {format_param('minimum_consecutive_frames', preset.minimum_consecutive_frames, is_int=True)}",
        "",
        "All values are validated and clamped to valid ranges automatically.",
    ]

    return "\n".join(lines)


# Populate TRACKING_PRESETS for backward compatibility
for _layer_name in TRACKING_LAYERS:
    TRACKING_PRESETS[_layer_name] = compose_presets([_layer_name])


__all__ = [
    "TrackingPreset",
    "TrackingLayer",
    "ParameterBounds",
    "TRACKING_PRESETS",
    "TRACKING_LAYERS",
    "PARAMETER_BOUNDS",
    "BASELINE",
    "get_preset",
    "apply_preset",
    "compose_presets",
    "parse_preset_string",
    "validate_preset",
    "clamp_preset",
    "get_tracker_kwargs_from_preset",
    "list_presets",
    "describe_preset",
]
