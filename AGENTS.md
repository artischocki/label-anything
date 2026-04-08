# AGENTS.md

## Storage Overview

This project does not currently use a database.

There is no SQLite file, no SQL schema, no ORM, no migrations, and no DB client
dependency in the repository. Persistence is file-based and centered around a
single task directory that contains source images, in-progress masks, and a
`results/` subdirectory for completed work.

## Effective Data Model

The app behaves like a tiny filesystem-backed datastore with three record types:

1. Source image
   - A labelable input image in the selected `tasks_dir`
   - Supported extensions: `bmp`, `png`, `jpg`, `jpeg`
   - Discovery is non-recursive and only scans the top level of `tasks_dir`

2. Working mask
   - A sidecar file next to the source image
   - Canonical filename: `<image_stem_without_optional__raw>_label.png`
   - Example:
     - `sample.png` -> `sample_label.png`
     - `sample_raw.png` -> `sample_label.png`

3. Finalized result
   - When the user marks a job done, the source image is moved into
     `tasks_dir/results/`
   - The final mask is written into that same `results/` directory

## Persistence Rules

### Image discovery

- The app lists only top-level image files in `tasks_dir`
- Files ending with `_label.png` are excluded from the work queue
- The `results/` directory is not scanned because discovery is not recursive

### Existing mask lookup

When opening an image, the app searches for an existing mask in this order:

1. Canonical sidecar name derived from the current image filename
2. Legacy fallback match against any `*_label.png` file whose stem matches:
   - the full image stem
   - the image stem with an exact trailing `_raw` removed

This means old masks can still be reused after the naming cleanup.

### Save behavior

There are two save modes:

1. In-progress save
   - Triggered by `Next Job`, `Prev Job`, and `Quit`
   - Writes the current mask next to the source image as `*_label.png`

2. Final save
   - Triggered by `Mark as Done`
   - Ensures `results/` exists
   - Moves the source image into `results/`
   - Writes the mask into `results/` using the canonical sidecar name

## Mask Format

- Masks are stored as PNG images, not as serialized arrays or DB rows
- The saved mask is an RGB image
- Masked pixels are white: `(255, 255, 255)`
- Unmasked pixels are black: `(0, 0, 0)`
- On load, masks are converted to grayscale, resized to the working image size,
  and thresholded with `> 128`

## In-Memory vs On-Disk Representation

The app uses two sizes for the same logical record:

- In memory:
  - Images are resized to a target height of `1024` before interactive work
  - Masks are edited as boolean NumPy arrays aligned to that resized image

- On disk:
  - When saving, the boolean mask is rendered back to the original image size
  - The original image file is preserved until finalization moves it into
    `results/`

## Important Modules

- `src/label_anything/files.py`
  - Defines file naming, discovery, existing-mask lookup, and `results/` paths

- `src/label_anything/ui/image_labeler.py`
  - Owns session lifecycle and save semantics
  - `save_mask(final=False)` writes the in-progress sidecar file
  - `save_mask(final=True)` moves the source image and writes the final mask

- `src/label_anything/masks.py`
  - Defines mask load/save conversions and boolean mask utilities

## Current Storage Invariants

- A source image is the unit of work
- At most one canonical mask is expected per source image
- `_raw` is treated as a removable suffix only when it appears exactly at the
  end of the stem
- Finalized jobs leave the active queue because the image is moved out of the
  top-level task directory

## Debugging Notes

If a user reports a "DB bug", the likely issue is actually in one of these
areas:

- image discovery in `tasks_dir`
- sidecar mask naming
- legacy mask matching
- save timing on `Quit` / `Next` / `Prev`
- move behavior when finalizing into `results/`
- mask resize and threshold behavior during load/save

## Repo Evidence

The repository currently contains:

- no `.db`, `.sqlite`, `.sqlite3`, or `.sql` files
- no migration directory
- no ORM or SQL dependency in `pyproject.toml`
- tests that validate the file naming and legacy mask lookup behavior
