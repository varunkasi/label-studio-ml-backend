# Improved Video Frame Extraction for Label Studio Sync

## Overview

This enhanced version of `convert_ls2yolo.py` includes a new time-based frame extraction method that addresses synchronization issues between Label Studio's video processing and YOLO training data.

## The Problem

Label Studio performs its own video reencoding which can cause frame misalignment when:
- Videos have non-standard frame rates (e.g., 29.97 FPS instead of 30 FPS)
- Variable frame rate (VFR) videos are used
- There are slight timing discrepancies between annotation timestamps and actual frame positions

## The Solution

### New `video_to_images()` Function

The `video_to_images()` function uses time-based frame extraction instead of sequential frame reading:

```python
def video_to_images(video_path: str, images_dir_path: str, target_frame_rate: float):
    """
    Extract frames from video at a target frame rate, handling non-standard FPS.
    This function addresses frame synchronization issues with Label Studio by using
    time-based frame extraction instead of sequential frame reading.
    """
```

**Key Features:**
- **Time-based sampling**: Extracts frames based on timestamp intervals rather than sequential reading
- **Frame rate normalization**: Converts any input FPS to a target FPS for consistency
- **Label Studio compatibility**: Matches the frame extraction behavior that Label Studio uses internally
- **Accurate frame counting**: Provides precise frame counts that match annotation expectations

### Enhanced `convert_labelstudio_to_yolo()` Function

The main conversion function now includes a `use_time_based_extraction` parameter:

```python
convert_labelstudio_to_yolo(
    labelstudio_json=annotations,
    output_labels_dir='labels/',
    output_frames_dir='frames/',
    video_path='video.mp4',
    use_time_based_extraction=True  # New parameter (default: True)
)
```

## Usage Examples

### 1. Standard Conversion (Time-based extraction - Recommended)

```python
from convert_ls2yolo import convert_labelstudio_to_yolo

# Load your Label Studio annotations
with open('annotations.json', 'r') as f:
    annotations = json.load(f)

# Convert with improved frame extraction
convert_labelstudio_to_yolo(
    labelstudio_json=annotations,
    output_labels_dir='./yolo_labels',
    output_frames_dir='./yolo_frames',
    video_path='./source_video.mp4',
    use_time_based_extraction=True  # Default behavior
)
```

### 2. Legacy Conversion (Sequential extraction)

```python
# Use the original method if needed
convert_labelstudio_to_yolo(
    labelstudio_json=annotations,
    output_labels_dir='./yolo_labels',
    output_frames_dir='./yolo_frames',
    video_path='./source_video.mp4',
    use_time_based_extraction=False
)
```

### 3. Standalone Frame Extraction

```python
from convert_ls2yolo import video_to_images

# Extract frames at specific FPS for Label Studio compatibility
frame_count = video_to_images(
    video_path='./input_video.mp4',
    images_dir_path='./extracted_frames',
    target_frame_rate=30.0
)
```

## When to Use Time-based Extraction

**Always use time-based extraction (default) when:**
- Working with Label Studio annotations
- Videos have non-standard frame rates (29.97, 23.976, etc.)
- You need precise synchronization between annotations and frames
- Working with variable frame rate videos

**Use sequential extraction only when:**
- Working with legacy workflows that expect exact sequential frame extraction
- Debugging frame extraction issues
- You have a specific requirement for sequential frame reading

## Technical Details

### Time-based Algorithm

1. **Calculate target frame count**: `target_frames = round(original_frames / original_fps * target_fps)`
2. **Read frames by timestamp**: Instead of reading every frame sequentially, read frames at specific time intervals
3. **Time interval calculation**: `interval = 1 / target_fps` seconds
4. **Frame selection**: Only save frames when `current_time - last_saved_time >= interval`

### Benefits

- **Consistent output**: Always produces the expected number of frames regardless of input FPS quirks
- **Better sync**: Matches Label Studio's internal frame extraction behavior
- **Handles edge cases**: Works correctly with VFR videos and unusual frame rates
- **Maintains quality**: No loss in frame quality, just better temporal alignment

## Testing

Use the provided test script to compare extraction methods:

```bash
python test_improved_extraction.py
```

This will show you the difference between sequential and time-based extraction for your specific videos.

## Compatibility

- **Backward compatible**: Existing code will work unchanged (time-based is now default)
- **Optional**: Can be disabled with `use_time_based_extraction=False`
- **Performance**: Slightly slower than sequential due to timestamp calculations, but more accurate
- **Dependencies**: No new dependencies required

## Migration Guide

### Updating Existing Code

**Before:**
```python
convert_labelstudio_to_yolo(annotations, labels_dir, frames_dir, video_path)
```

**After (no changes needed, but recommended to be explicit):**
```python
convert_labelstudio_to_yolo(
    annotations, labels_dir, frames_dir, video_path,
    use_time_based_extraction=True
)
```

### Troubleshooting Frame Count Mismatches

If you see warnings like:
```
⚠️  Frame count mismatch: expected 1500, extracted 1498
🔄 Updating YOLO annotations for correct frame count...
```

This is normal and indicates that time-based extraction has corrected a frame count discrepancy. The YOLO annotations are automatically updated to match the actual extracted frame count.