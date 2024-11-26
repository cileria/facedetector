# YOLO Object Detection with Video Processing

This project uses a YOLO model for real-time object detection on video streams, specifically detecting persons and performing face recognition. The project is implemented in Python with `ultralytics` and `OpenCV`.

## Features
- Real-time object detection using YOLO
- Face recognition using DeepFace
- Detection results are shown in a video stream
- Detected frames are saved to `output` directory
- Output video with detected objects is saved in `.mp4` format

## Prerequisites

### Required Software

1. **UV Package Manager**
   - Install using the following command:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - After installation, restart your terminal or run:
     ```bash
     source ~/.bashrc  # Linux
     source ~/.zshrc   # macOS
     ```
   - UV will automatically download and manage Python for us - no separate Python installation needed!

2. **Make** (usually pre-installed on macOS/Linux)
   - macOS: Comes with Xcode Command Line Tools
   - Linux: Install using your package manager
     ```bash
     sudo apt install make  # Ubuntu/Debian
     sudo yum install make  # CentOS/RHEL
     ```

### System Requirements
- At least 8GB RAM (16GB recommended)
- Modern CPU (Intel i5/AMD Ryzen 5 or better)
- GPU is optional but recommended for better performance
- At least 5GB free disk space

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Prepare Training Data**
   - Create a directory for your training images:
     ```bash
     mkdir -p videos/trainingsdata/jan
     ```
   - Copy your training images (photos of people you want to recognize) to:
     ```
     videos/trainingsdata/jan/
     ```
   - Supported formats: `.jpg`, `.png`
   - Best results with clear, front-facing photos
   - Name your images meaningfully as they will be used as labels

3. **Setup the Environment**
   ```bash
   make setup-all
   ```
   This will:
   - Download and install Python 3.10 (handled automatically by UV)
   - Create a virtual environment
   - Install all dependencies
   - Download required ML models
   - Verify your training data
   - Create necessary directories

## Usage

### Basic Usage
Run the example with included test videos:
```bash
make run
```

### Custom Video Processing
Process your own video:
```bash
make run-custom VIDEO_PATH=/path/to/your/video.mp4
```

### Development Commands
- Format code:
  ```bash
  make format
  ```
- Run linting:
  ```bash
  make lint
  ```
- Run tests:
  ```bash
  make test
  ```

### Clean Up
Remove all generated files and virtual environment:
```bash
make clean-all
```

## Project Structure
```
customer_js/
├── model/              # ML model files
│   └── yolo11n.pt     # YOLO model (downloaded automatically)
├── videos/             # Video and training data
│   ├── jan/           # Example videos
│   └── trainingsdata/ # Training images for face recognition
├── output/            # Output directory for processed frames
├── src/               # Source code
├── tests/             # Test files
├── pyproject.toml     # Project configuration
├── README.md          # This file
└── Makefile          # Build and run commands
```

## Common Issues and Solutions

### No Module Found Errors
If you see "No module found" errors, try reinstalling everything:
```bash
make reinstall
```

### GPU/CUDA Issues
By default, the project runs on CPU. For GPU support, ensure you have:
- NVIDIA GPU with recent drivers
- CUDA toolkit installed
The project will automatically use GPU if available.

### Training Data Issues
If you see "No training images found":
1. Check that your images are in the correct directory
2. Verify image formats are `.jpg` or `.png`
3. Run `make setup-training-data` to verify

## Available Make Commands
Run `make help` to see all available commands. Common ones include:
- `make setup-all`: Full installation with all dependencies
- `make run`: Run with example videos
- `make run-custom`: Run with your own video
- `make clean`: Clean temporary files
- `make clean-all`: Complete cleanup including downloaded models

## Requirements for Videos
- Supported formats: MP4
- Recommended resolution: 720p or 1080p
- Good lighting conditions for better detection
- Clear, unobstructed views of faces for recognition

## License
This project is licensed under the MIT License.

## Support
If you encounter any issues:
1. Check the Common Issues section above
2. Run `make clean-all` followed by `make setup-all`
3. Verify your training data and video files
4. Create an issue in the repository with:
   - Command you ran
   - Complete error message
   - Your system information