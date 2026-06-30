# Football Player Detection

![Demo](output_videos/output-video.mp4)

A computer vision project for detecting football players in images and video. This repository includes training and inference support for a player detection model, along with utilities for data preparation, evaluation, and visualization.

## Features

- Detect football players in images and video frames
- Support for training on custom data
- Inference pipeline for real-time or batch processing
- Visualization of detection results

## Installation

1. Clone the repository

   ```bash
   git clone https://github.com/yourfzz/football-player-detection.git
   cd football-player-detection
   ```

2. Create a Python virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Prepare your dataset in the expected format, then run the training script:

```bash
python train.py --data data/football_dataset.yaml --epochs 50 --batch-size 16
```

### Inference

Run detection on a single image:

```bash
python detect.py --source assets/sample.jpg --output results/
```

Run detection on a video:

```bash
python detect.py --source assets/sample.mp4 --output results/ --view-img
```

## Dataset

The project expects a dataset with labeled images for player detection. Common dataset structure:

- `data/images/train/`
- `data/images/val/`
- `data/labels/train/`
- `data/labels/val/`

Labels should match the model format used by the training pipeline.

## Evaluation

Evaluate the model using the provided evaluation script:

```bash
python evaluate.py --weights runs/train/exp/weights/best.pt --data data/football_dataset.yaml
```

## Contributing

Contributions are welcome. Please open issues or pull requests for bug fixes, feature requests, and improvements.

## License

This project is provided under the MIT License.
