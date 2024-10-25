# Prerequisites

ffmpeg must be installed on your system.

Download: https://www.ffmpeg.org/download.html

If you don't have it installed, install it and restart the terminal.

# Running the CLI

```bash
python -m tool-suite.video-info-extraction.cli -h
```

## Running an Audio Transcription Task

```bash
python -m tool-suite.video-info-extraction.cli transcribe --audio_files tool-suite/video-info-extraction/audio_files/news.mp3
```
