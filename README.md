# Audio Conversation Analyzer - Setup and Usage Guide

This guide will help you set up and use the Audio Conversation Analyzer application that can process `.mp3` and `.wav` files to:
- Transcribe audio using open-source models (no commercial APIs)
- Detect and report the number of participants in the recording
- Generate a concise text summary of the conversation
- Analyze sentiment or tone (interested, frustrated, etc.)

## Setup Instructions

### Prerequisites

- Python 3.8 or newer
- pip (Python package installer)
- Virtual environment (recommended)
- FFmpeg (for audio processing - optional, as we've added fallback methods)

### Installation Steps

1. **Create and activate a virtual environment** (recommended)

```bash
# On Windows
python3.10 -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

2. **Install the required dependencies**

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- numpy
- librosa (for audio processing)
- speechbrain
- torch
- matplotlib (for visualizations)
- transformers (for sentiment analysis and summarization)
- whisper (for transcription)
- pydub (for audio format conversion)
- soundfile
- protobuf

3. **FFmpeg Installation (recommended but optional)**

FFmpeg is recommended for better audio processing, but the application now includes fallback methods if FFmpeg is not available.

To install FFmpeg, you can:

- **Use our FFmpeg Installer Helper**:
  ```bash
  python ffmpeg_installer.py
  ```
  This utility will help you install FFmpeg on your system.

- **Manual installation options**:
  - On Windows:
    - Download from [FFmpeg website](https://ffmpeg.org/download.html)
    - Add to system PATH
  - On macOS:
    ```bash
    brew install ffmpeg
    ```
  - On Linux:
    ```bash
    sudo apt-get install ffmpeg  # Debian/Ubuntu
    ```

## Running the Application

1. **Launch the application**

```bash
python audio_conversation_analyzer.py
```

2. **Using the application**

- Click the "Browse" button to select an audio file (`.mp3` or `.wav`)
- Click "Analyze" to start processing the audio
- The application will:
  - Display the audio waveform in the Visualization tab
  - Transcribe the audio
  - Detect the number of speakers
  - Analyze sentiment
  - Generate a summary
- Results will appear in the respective tabs once processing is complete

## Understanding the Results

### Waveform Visualization
- Displays the audio amplitude over time
- Useful for identifying quiet and loud sections

### Transcription
- Full text transcription of the audio file

### Analysis
- **Number of participants**: Estimated number of speakers in the conversation
- **Sentiment Analysis**: Overall tone of the conversation (Positive/Interested, Negative/Frustrated, or Neutral)
- **Conversation Summary**: Concise summary of the main points discussed

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - The application will now use an alternative method for audio processing if FFmpeg is not available
   - For best results, consider installing FFmpeg using our installer helper: `python ffmpeg_installer.py`

2. **Memory errors**
   - The application uses ML models that require significant memory
   - Try closing other applications or use a computer with more RAM
   - Use the simplified version for systems with limited resources

3. **Slow processing**
   - Transcription and analysis may take time, especially for longer audio files
   - The progress bar indicates current status

4. **Inaccurate speaker detection**
   - Speaker detection is based on a simple energy-based approach and may not be perfect
   - For better results, ensure good audio quality with minimal background noise

## Technical Notes

- Transcription uses OpenAI's Whisper model (turbo version)
- The application now includes fallback methods if FFmpeg is not available
- Speaker detection uses a simple energy-based approach
- Sentiment analysis and summarization use Hugging Face Transformers pre-trained models
- All processing is done locally without using any commercial APIs

## License

This application is provided for educational purposes. The underlying models and libraries have their own licenses:
- Whisper: MIT License
- Transformers: Apache 2.0
- Other libraries: Various open-source licenses