# samAudio 🎵

A lightweight and optimized implementation of **SAM-Audio** (Segment Anything Model for Audio) tailored for Linux environments with NVIDIA GPUs. This project provides a robust wrapper for audio separation based on textual prompts, optimized for memory efficiency and stability.

## 🚀 Key Features

- **Lite Model Weights**: Automatically strips unused heavy components (like the vision encoder) to maximize VRAM availability and stability.
- **Dynamic Chunking**: Processes long audio files in configurable chunks to avoid OOM (Out Of Memory) errors and ensure consistent performance.
- **NVIDIA Optimized**: Uses `bfloat16` precision and peak normalization for high-quality, distortion-free audio output.
- **Robust Pre-processing**: Handles automatic resampling to 44.1kHz (or model default) and mono-mixing to prevent common audio processing artifacts.

## 🛠️ Installation

### 1. Prerequisites
Ensure you have a Python 3.10+ environment.

### 2. Install PyTorch (NVIDIA Users)
For optimal performance on Linux with NVIDIA GPUs, it is recommended to install the stable PyTorch version:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
```

### 3. Install Dependencies
Install the required packages, including specialized wheels for audio/video decoding:

```bash
pip install -r requirements.txt
```

*Note: The requirements include custom wheels for `decord` and `torchcodec` optimized for specific architectures.*

## 📖 Usage

You can run the audio separation directly from the command line using `samaudio.py`.

### Basic Command
```bash
python samaudio.py -i input_file.wav -p "a description of the sound to extract" -o output_separated.wav
```

### Arguments
| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Path to the input audio file (Required) | - |
| `--prompt` | `-p` | Textual description of the sound to separate (Required) | - |
| `--output` | `-o` | Path to the output saved audio | `output_base.wav` |
| `--chunk_sec` | | Duration of each processing chunk in seconds | `20.0` |

### Examples
- **Extracting vocals:** `python samaudio.py -i song.mp3 -p "human singing voice" -o vocals.wav`
- **Extracting background noise:** `python samaudio.py -i street.wav -p "siren sound" -o siren_only.wav`

## ⚙️ Optimization Details

This implementation applies several tricks to improve stability on Linux:
- **Component Pruning**: Deletes `visual_ranker`, `text_ranker`, and `span_predictor` when not in use.
- **Inference Mode**: Uses `torch.inference_mode()` and `torch.cuda.amp.autocast` for faster execution.
- **Memory Management**: Explicitly triggers `gc.collect()` and `torch.cuda.empty_cache()` between processing steps.
- **Peak Normalization**: Automatically normalizes output to 0.9 peak to ensure clear audibility.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Based on the [SAM-Audio](https://github.com/facebookresearch/sam-audio) project by Facebook Research.
- Prebuilt wheels provided by the [Nvidia-DGX-prebuild-wheels](https://github.com/mamorett/Nvidia-DGX-prebuild-wheels) repository.
