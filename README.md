# FloodDiffusion: Tailored Diffusion Forcing for Streaming Motion Generation

We present **FloodDiffusion**, a new framework for text-driven, streaming human motion generation. Given time-varying text prompts, FloodDiffusion generates text-aligned, seamless motion sequences with real-time latency.

## Features

-   🔄 **Streaming Generation**: Support for continuous motion generation with text condition changes
-   🚀 **Latent Diffusion Forcing**: Efficient generation using compressed latent space with diffusion
-   ⚡ **Real-time Capable**: Optimized for streaming inference with ~50 FPS model output

## Installation

### Environment Setup

```bash
# Create conda environment
conda create -n motion_gen python=3.10
conda activate motion_gen

# Install PyTorch
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention
conda install -c nvidia cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX
pip install flash-attn --no-build-isolation
```

## Quick Inference (No Data Required)

If you only need to generate motions and don't plan to train or evaluate models, you can use our standalone model on Hugging Face:

🤗 **[ShandaAI/FloodDiffusion](https://huggingface.co/ShandaAI/FloodDiffusion)**

This version requires **no dataset downloads** and works out-of-the-box for inference:

```python
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained(
    "ShandaAI/FloodDiffusion",
    trust_remote_code=True
)

# Generate motion from text
motion = model("a person walking forward", length=60)
print(f"Generated motion: {motion.shape}")  # (~240, 263)

# Generate as joint coordinates for visualization
motion_joints = model("a person walking forward", length=60, output_joints=True)
print(f"Generated joints: {motion_joints.shape}")  # (~240, 22, 3)

# Multi-text transitions
motion = model(
    text=[["walk forward", "turn around", "run back"]],
    length=[120],
    text_end=[[40, 80, 120]]
)
```

For detailed API documentation, see the [model card](https://huggingface.co/ShandaAI/FloodDiffusion).

> **Note**: For training, evaluation, or using the scripts in this repository, continue with the Data Preparation section below.

## Data Preparation

### Prepare Data from Original Sources

To reproduce our results from scratch, follow the original data preparation pipelines:

**HumanML3D**:
- Follow the instructions in the [HumanML3D repository](https://github.com/EricGuo5513/HumanML3D)
- Extract 263D motion features using their processing pipeline
- Place the processed data in `raw_data/HumanML3D/`

**BABEL**:
- Download from the [BABEL website](https://babel.is.tue.mpg.de/)
- Process the motion sequences to extract 263D features
- For streaming generation, segment and process according to the frame-level annotations
- Place the processed data in `raw_data/BABEL_streamed/`

**Dependencies**:
- Download T5 encoder weights from Hugging Face
- Download T2M evaluation models from the [text-to-motion repository](https://github.com/EricGuo5513/text-to-motion)
- Download GloVe embeddings

### Quick Start: Download Preprocessed Data (Recommended)

We provide all necessary data (datasets, dependencies, and pretrained models) on Hugging Face: 🤗 **[ShandaAI/FloodDiffusionDownloads](https://huggingface.co/ShandaAI/FloodDiffusionDownloads)**

Simply run:

```bash
pip install huggingface_hub
python download_assets.py
```

This will automatically download and extract everything into the correct directories (`deps/`, `raw_data/`, `outputs/`).

### Directory Structure

After downloading or preparing the data, your project should have the following structure:

**Dependencies Directory**:

```
deps/
├── t2m/                     # Text-to-Motion evaluation models
│   ├── humanml3d/           # HumanML3D evaluator
│   ├── kit/                 # KIT-ML evaluator
│   └── meta/                # Statistics (mean.npy, std.npy)
├── glove/                   # GloVe word embeddings
│   ├── our_vab_data.npy
│   ├── our_vab_idx.pkl
│   └── our_vab_words.pkl
└── t5_umt5-xxl-enc-bf16/    # T5 text encoder
```

**Dataset Directory**:

```
raw_data/
├── HumanML3D/
│   ├── new_joint_vecs/      # 263D motion features (required)
│   ├── texts/               # Text annotations
│   ├── train.txt            # Training split
│   ├── val.txt              # Validation split
│   ├── test.txt             # Test split
│   ├── all.txt              # All samples
│   ├── Mean.npy             # Dataset mean
│   ├── Std.npy              # Dataset std
│   ├── TOKENS_*/            # Pretokenized features (auto-generated)
│   └── animations/          # Rendered videos (optional)
│
└── BABEL_streamed/
    ├── motions/             # 263D motion features (required)
    ├── texts/               # Text annotations
    ├── frames/              # Frame-level annotations
    ├── train_processed.txt  # Training split
    ├── val_processed.txt    # Validation split
    ├── test_processed.txt   # Test split
    ├── TOKENS_*/            # Pretokenized features (auto-generated)
    └── animations/          # Rendered videos (optional)
```

**Pretrained Models Directory**:

```
outputs/                     # Pretrained model checkpoints
├── vae_1d_z4_step=300000.ckpt          # VAE model (1D, z_dim=4)
├── 20251106_063218_ldf/
│   └── step_step=50000.ckpt            # LDF model checkpoint (HumanML3D)
└── 20251107_021814_ldf_stream/
    └── step_step=240000.ckpt           # LDF streaming model checkpoint (BABEL)
```

> **Note**: If you downloaded the models using the script above, the paths are already correctly configured. Otherwise, update `test_ckpt` and `test_vae_ckpt` in your config files to point to your checkpoint locations.

## Configuration

Create `configs/paths.yaml` from the example:

```bash
cp configs/paths_default.yaml configs/paths.yaml
# Edit paths.yaml to point to your data directories
```

### Available Configs

-   `vae_wan_1d.yaml` - VAE training configuration
-   `ldf.yaml` - LDF training on HumanML3D
-   `ldf_babel.yaml` - LDF training on BABEL
-   `stream.yaml` - Streaming generation config
-   `ldf_generate.yaml` - Generation-only config

## Training

### 1. Train VAE (Motion Encoder)

```bash
# Train VAE
python train_vae.py --config configs/vae_wan_1d.yaml --override train=True

# Test VAE
python train_vae.py --config configs/vae_wan_1d.yaml
```

### 2. Pretokenize Dataset

Precompute VAE tokens for diffusion training:

```bash
python pretokenize_vae.py --config configs/vae_wan_1d.yaml
```

### 3. Train Latent Diffusion Forcing (Flood Diffusion)

```bash
# Train on HumanML3D
python train_ldf.py --config configs/ldf.yaml --override train=True

# Train on BABEL (streaming)
python train_ldf.py --config configs/ldf_babel.yaml --override train=True

# Test/Evaluate
python train_ldf.py --config configs/ldf.yaml
```

## Generation

### Interactive Generation

```bash
python generate_ldf.py --config configs/stream.yaml
```

## Visualization

Render motion files to videos:

```bash
python visualize_motion.py
```

This script:

-   Reads 263D motion features from disk
-   Renders to MP4 videos with skeleton visualization
-   Supports batch processing of directories

## Web Real-time Demo

For real-time interactive demo with streaming generation, see [`web_demo/README.md`](web_demo/README.md).

## Model Architecture

### VAE (Variational Autoencoder)

-   **Input**: T × 263 motion features
-   **Latent**: (T/4) × 4 tokens
-   **Architecture**: Causal encoder and decoder based on WAN2.2

### LDF (Latent Diffusion Forcing)

-   **Backbone**: DiT based on WAN2.2
-   **Text Encoder**: T5
-   **Diffusion Schedule**: Triangular noise schedule
-   **Streaming**: Autoregressive latent generation

## Project Structure

```
pl_train/
├── configs/                        # Configuration files
│   ├── vae_wan_1d.yaml             # VAE training config
│   ├── ldf.yaml                    # LDF training (HumanML3D)
│   ├── ldf_babel.yaml              # LDF training (BABEL)
│   ├── stream.yaml                 # Streaming generation
│   └── paths.yaml                  # Data paths (create from .example)
│
├── datasets/                       # Dataset loaders
│   ├── humanml3d.py                # HumanML3D dataset
│   └── babel.py                    # BABEL dataset
│
├── models/                         # Model implementations
│   ├── vae_wan_1d.py               # VAE encoder-decoder
│   └── diffusion_forcing_wan.py    # LDF diffusion model
│
├── metrics/                        # Evaluation metrics
│   ├── t2m.py                      # Text-to-Motion metrics
│   └── mr.py                       # Motion reconstruction metrics
│
├── utils/                          # Utilities
│   ├── initialize.py               # Config & model loading
│   ├── motion_process.py           # Motion data processing
│   └── visualize.py                # Rendering utilities
│
├── train_vae.py                    # VAE training script
├── train_ldf.py                    # LDF training script
├── pretokenize_vae.py              # Dataset pretokenization
├── generate_ldf.py                 # Motion generation
├── visualize_motion.py             # Batch visualization
├── requirements.txt                # Python dependencies
└── web_demo/                       # Real-time web demo (separate)
```

**External Dependencies**:

```
<project_root>/
├── deps/                           # Model dependencies
└── raw_data/                       # Motion datasets
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{flood2025,
  title={FloodDiffusion: Tailored Diffusion Forcing for Streaming Motion Generation},
  author={YIYI CAI, Yuhan Wu, Kunhang Li, YOU ZHOU, Bo Zheng, Haiyang Liu},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Shanda AI Research Tokyo

**Note**: This project includes code from third-party sources with separate licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details.

## Acknowledgments

-   [HumanML3D](https://github.com/EricGuo5513/HumanML3D) - Dataset
-   [text-to-motion](https://github.com/EricGuo5513/text-to-motion) - Evaluation metrics
-   [BABEL](https://babel.is.tue.mpg.de/) - Dataset for streaming motion generation
-   [AMASS](https://amass.is.tue.mpg.de/) - Source motion capture data
-   [PyTorch Lightning](https://lightning.ai/) - Training framework
-   [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) - Quaternion operations code
-   [Hugging Face Transformers](https://github.com/huggingface/transformers) - T5 model implementation
-   [Alibaba Wan Team](https://github.com/Wan-Video/Wan2.2) - WAN model architecture and components

### Data License Notice

The preprocessed datasets we provide contain **extracted motion features (263-dim) and text annotations** derived from HumanML3D and BABEL, which are built upon AMASS and HumanAct12. We **do not distribute raw AMASS data** (SMPL parameters/meshes). This follows standard practice in the motion generation research community. If you require raw motion data or plan to use it for commercial purposes, please register and agree to the licenses on the [AMASS website](https://amass.is.tue.mpg.de/).
