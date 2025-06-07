# Tutorial 3: GitHub Actions CI/CD + ONNX + Releases

**Scope:**
- Build on Tutorial 2 codebase (MLflow + DVC)
- Add GitHub Actions CI/CD pipeline
- ONNX model conversion for production
- Automated testing and deployment
- GitHub releases with model artifacts

## Setup

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install mlflow pytest tqdm numpy dvc onnx onnxruntime
```

## Quick Start

```bash
# Initialize repository
git init
dvc init --no-scm

# Setup data and train
python prepare_expanded_data.py
python train.py

# Convert to ONNX
python export_onnx.py

# Run tests
pytest tests/ -v

# Push to GitHub (triggers CI/CD)
git remote add origin https://github.com/your-username/tutorial-3-cicd.git
git push -u origin main
```

## GitHub Actions Pipeline

The CI/CD pipeline (`.github/workflows/ci-cd.yml`) includes:

1. **CI (Continuous Integration)**:
   - Download and process data (40 samples)
   - Train model for 1 epoch
   - Convert to ONNX format
   - Run comprehensive tests
   - Validate model accuracy

2. **CD (Continuous Deployment)**:
   - Create GitHub release
   - Upload model artifacts (PyTorch + ONNX)
   - Tag with version number

## What You'll Learn

1. **GitHub Actions**: Create CI/CD workflows for ML projects
2. **ONNX Conversion**: Convert PyTorch models for production deployment
3. **Automated Testing**: Test ML pipelines in CI environment
4. **Artifact Management**: Handle model files in CI/CD
5. **Release Automation**: Automatically create releases with model versions

## Files

- `train.py` - Training script (retained from Tutorial 2)
- `export_onnx.py` - ONNX conversion script
- `model.py` - SimpleCNN architecture (unchanged)
- `prepare_expanded_data.py` - Data preparation (retained)
- `.github/workflows/ci-cd.yml` - GitHub Actions workflow
- `tests/` - Comprehensive test suite
- `requirements.txt` - All dependencies including ONNX

## CI/CD Features Demonstrated

- Automated data pipeline testing
- Model training in CI environment
- ONNX conversion and validation
- Automated releases with artifacts
- Test-driven ML development

## Verification Steps

1. **Local Testing**: All tests pass locally
2. **ONNX Conversion**: PyTorch model converts to ONNX successfully
3. **GitHub Actions**: CI/CD pipeline runs without errors
4. **Model Artifacts**: ONNX and PyTorch models available in releases
5. **Integration**: MLflow, DVC, and CI/CD work together