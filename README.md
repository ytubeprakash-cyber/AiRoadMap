# Modelcraft Lab

Week 1 exercises for the AI Roadmap. This branch covers Python data utilities, matrix multiplication, and the math behind a training step.

## Run from a fresh checkout

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). The project uses Python 3.12 and pins dependency versions in `uv.lock`.

```bash
git clone --branch branchx https://github.com/ytubeprakash-cyber/AiRoadMap.git
cd AiRoadMap
uv sync --locked
uv run --locked pytest
uv run --locked jupyter lab
```

In JupyterLab, open [the Week 1 notebook](notebooks/week01_numpy.ipynb), select the Python kernel from the project environment, and choose **Kernel → Restart Kernel and Run All Cells**.

## What to check

- The notebook compares dot products, shows which matrix shapes can be multiplied, and times the Python loop and NumPy versions. Benchmark timings vary by machine.
- The statistics example reports a mean of 200 ms, a population variance of 5,000 ms², and 40% of the observed launches above 200 ms.
- The gradient descent example reduces loss from 4.00 to 2.56 at a learning rate of 0.1. At 1.1, the update overshoots and loss rises to 5.76.
- The tests check the linear algebra and JSONL helpers. [Sample JSONL data](data/sample.jsonl) is included.
