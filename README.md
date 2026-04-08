# label_anything

## Prerequisites
You will need WSL installed to run this!
I did not test in Windows and I dont know how you would need to setup Docker in Windows for this.

## Quick Start

```bash
./run /path/to/tasks_dir
```

Useful variants:

```bash
./run --rebuild /path/to/images
./run /path/to/images --sam-model large # or: tiny, small, base_plus (<- default)
./run /path/to/images --rgb-to-bgr (if colors inverted by numpy)
./run --cpu /path/to/images
```

The `run` wrapper builds the image if needed, installs SAM 2 inside the image,
downloads the SAM 2.1 checkpoints, mounts your image directory, and forwards
X11 so the Tk window opens on your desktop.

If your network uses a custom corporate root CA, `./run` also bundles host
certs from `/usr/local/share/ca-certificates` into the Docker build
automatically. You can override that with `LABEL_ANYTHING_EXTRA_CA_FILE`.


## Local Install (not recommended)

Docker is the intended path now. If you still want to run it locally:

```bash
pip install -e .
export LABEL_ANYTHING_SAM2_DIR=/path/to/sam2
label-anything /path/to/tasks_dir
```

## Behavior

- Masks are saved next to the source image as `*_label.png`.
- `Mark as Done` moves the source image into `results/` and writes the mask
  there too.
- You can just move results back to the tasks dir if you want to re-edit them.
- Panning with: L-Mouse + CTRL or M-Mouse, you'll figure out the rest i believe
