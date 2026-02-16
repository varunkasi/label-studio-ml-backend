# Label Studio Video Annotation Skills Design

Date: 2026-02-14

## Problem

The video object tracking workflow involves many CLI tools with complex flags. Running them manually via SSH is error-prone (wrong flags, forgotten env vars, 0-based vs 1-based frame confusion, argparse issues with dash-prefixed IDs).

## Solution

4 Claude Code skills that map natural-language requests to the correct CLI commands:

| Skill | Commands | Runs via |
|-------|----------|----------|
| `label-studio-track-tools` | `video_tools.py` (swap-ids, sparsify, trim-tail, smooth, pad) | SSH |
| `label-studio-seeding` | `initial_seeding_video_boxes_manual_merge.py` | Docker exec (GPU) |
| `label-studio-export` | `process_annotation.py`, `export_interpolated_annotation.sh` | Docker exec or SSH |
| `label-studio-annotation-ops` | `delete_annotation_or_prediction.py`, `mergevideoregions.py`, `validate_prediction.py` | SSH |

## Shared Conventions

- Accept any input format (prose, lists, tables, shorthand)
- Ask for missing required fields before executing
- Confirm before executing destructive or long-running operations
- `source ~/.env_keys &&` prefix for env vars
- `python3` not `python`
- Track IDs starting with `-` use `=` syntax
- Never restart containers

## Skill-Specific Design Decisions

### label-studio-seeding
- Converts 1-based user frame numbers to 0-based `--global-start/end`
- Asks user for region ID when `--track-id` is needed
- Warns if `--enable-oracle` set without `--oracle-stride`
- Long timeout (600s) for seeding runs

### label-studio-export
- Always asks: (1) do you need masks? (2) all persons or specific?
- For specific person, asks for the number in the track's metadata box
- Routes to full pipeline (Docker exec) vs lightweight (SSH) based on mask need

### label-studio-annotation-ops
- Delete always requires explicit confirmation (irreversible)
- Asks annotation vs prediction if ambiguous
- Merge requires `id:N` in track meta.text

## Files Created

- `~/.claude/skills/label-studio-track-tools/SKILL.md`
- `~/.claude/skills/label-studio-seeding/SKILL.md`
- `~/.claude/skills/label-studio-export/SKILL.md`
- `~/.claude/skills/label-studio-annotation-ops/SKILL.md`
