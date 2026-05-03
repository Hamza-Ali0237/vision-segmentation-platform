# Contributing

Thank you for your interest in contributing to the Vision Segmentation Platform!

## Getting Started

```bash
git clone https://github.com/Hamza-Ali0237/vision-segmentation-platform
cd vision-segmentation-platform

conda create -n vsp python=3.10 -y
conda activate vsp

pip install -e .
pip install -r requirements.txt
pip install "numpy<2"
```

## Running Tests

All tests are CPU-only and use synthetic data — no GPU, real dataset, or live AWS endpoint needed.

```bash
python -m pytest                   # run all tests
python -m pytest -v --tb=long      # verbose output
```

Make sure tests pass before opening a pull request.

## Reporting Bugs

Open an issue and include:
- Python version and OS
- Steps to reproduce the problem
- The full error traceback

## Submitting Changes

1. Fork the repository and create a branch from `main`.
2. Make your changes — keep commits focused and the diff readable.
3. Add or update tests for any changed behaviour.
4. Ensure `python -m pytest` passes.
5. Open a pull request with a clear description of what you changed and why.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Keep functions small and focused.
- Avoid leaving debug `print` statements in committed code.
- Add docstrings to public functions and classes.

## Sensitive Data

Never commit real AWS credentials, account IDs, or S3 bucket names.  
`training/configs/base.yaml` is git-ignored for this reason — fill it in locally.
