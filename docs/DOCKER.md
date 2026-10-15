# Running the demo in Docker

The image holds the code and the CPU builds of TensorFlow (tensorflow-cpu 2.21.0) and PyTorch (2.14.0+cpu) on
Python 3.11.9. The model weights are not in the image: they are mounted read-only, or downloaded when the
container starts.

## Build

From the repository root:

```bash
docker build -f app/Dockerfile -t drgrade:1.0.0 .
```

## Run with local weights

Download the weights once on the host (`python scripts/download_weights.py --inference-only`), then:

```bash
docker run --rm --name drgrade -p 8501:8501 --memory 5g \
  -v "$(pwd)/weights:/weights:ro" \
  drgrade:1.0.0
```

Open http://localhost:8501 once the log prints `open http://localhost:8501` (loading the models took 80 s in our
test). One image with Grad-CAM took about 2-2.5 minutes on four CPU cores (the first image is slower: the graphs are
compiled on first use), with a peak of 3.6 GB of RAM; Integrated Gradients and Occlusion add several minutes each.

Settings are environment variables, so nothing has to be rebuilt:

| Variable | Default in the image | Meaning |
|---|---|---|
| `DRGRADE_WEIGHTS_DIR` | `/weights` | folder with the model files |
| `DRGRADE_OUTPUT_DIR` | `/tmp/drgrade` | where command-line tools write |
| `DRGRADE_DEMO_IMAGES` | `/app/data/demo_images` | demo images listed on the page (`labels.csv`) |
| `DRGRADE_ARM` | `crop` (from config.yaml) | grader arm loaded at start: `crop` or `direct` |
| `PORT` / `HOST` | `8501` / `0.0.0.0` | where the server listens |
| `OMP_NUM_THREADS` | `4` | CPU threads for TensorFlow and PyTorch; set it to the cores the container may use |
| `DRGRADE_DOWNLOAD_WEIGHTS` | `0` | `1`: download the weights into `DRGRADE_WEIGHTS_DIR` at start (it must be writable) |

An uploaded image is written to a temporary file for its run and deleted when the run ends.

The command-line tools work in the same image, e.g. the parity check against a reference file:

```bash
docker run --rm --memory 5g -v "$(pwd)/weights:/weights:ro" -v "$(pwd)/my_images:/images:ro" \
  -v "$(pwd)/data/parity:/parity:ro" drgrade:1.0.0 \
  python scripts/parity_check.py --images /images --reference /parity/reference.npz
```

## Hugging Face Docker Space

A Space builds the image itself and cannot mount a folder, so the container downloads the weights at start.

1. Create a new Space with the **Docker** SDK (CPU basic or larger; the models need about 4 GB of RAM).
2. Push this repository to the Space and put this header at the top of the Space's README.md:

   ```yaml
   ---
   title: DR grading pipeline
   sdk: docker
   app_port: 7860
   ---
   ```

3. A Space builds from `Dockerfile` in the repository root: copy `app/Dockerfile` there (`cp app/Dockerfile
   Dockerfile`; its paths are relative to the repository root, so it works unchanged).
4. In the Space settings, set the variables `PORT=7860`, `DRGRADE_DOWNLOAD_WEIGHTS=1`,
   `DRGRADE_WEIGHTS_DIR=/home/app/weights` and `OMP_NUM_THREADS=2` (CPU basic has two cores).

The weights download needs the published weights release (`weights_manifest.json`, `release.base_url`).
