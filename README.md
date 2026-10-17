# Emotion Analysis

Real-time facial emotion analysis using DeepFace and OpenCV. The program reads
your webcam, estimates emotion scores every few frames, and at the end saves
the emotion history as a CSV file and a graph.

![Emotion graph](emotion_graph.png)

## Features

- Live webcam face detection with the dominant emotion shown on screen
- Scores for 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
- Emotion history saved to `emotions.csv` (pandas)
- Smoothed time-series graph and average score chart saved as `emotion_graph.png`

## Installation

```bash
git clone https://github.com/KULLANICI_ADIN/emotion-analysis.git
cd emotion-analysis
pip install -r requirements.txt
```

Python 3.9-3.11 is recommended.

## Usage

```bash
python main.py
```

Press `q` to quit. After quitting, the graph window opens and the results are
saved in the project folder.

## Built with

DeepFace, OpenCV, pandas, Matplotlib

## Limitations

- The model classifies **facial expressions**, not actual feelings. Posed
  expressions, lighting, camera angle and cultural differences affect results.
- Accuracy of public emotion datasets is limited, so scores are estimates.
- Only the first detected face is analyzed.

## Future improvements

- Dominant emotion duration analysis
- Multiple face support
- Refactoring into functions