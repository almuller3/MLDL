# Video Feature Extraction Model

This repository contains a Python script designed to extract significant features from a video using a combination of OpenCV, NumPy, and a pre-trained CLIP model from Hugging Face's Transformers library.

## Overview

The script processes a video file by performing the following steps:

1. **Frame Extraction**: Extracts all frames from the input video using OpenCV.
2. **Delta Calculation**: Computes the difference (delta) between consecutive frames to identify changes or motion within the video.
3. **Significant Change Detection**: Filters out frames with significant changes based on a specified threshold, retaining only the most relevant frames.
4. **Feature Extraction using CLIP**: Uses a pre-trained CLIP model to extract image features from the significant frames. These features can be used for further analysis or downstream tasks.

## Dependencies

The script requires the following Python libraries:

- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- Hugging Face Transformers (`transformers`)
- PyTorch (`torch`)

You can install these dependencies using the following command:

```bash
pip install opencv-python numpy matplotlib transformers torch
