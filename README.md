# Codex Audio Transcriber

## Overview
Codex Audio Transcriber is a desktop application for transcribing audio and video files into text. It utilizes `faster-whisper` speech recognition models to provide accurate transcriptions. The application features a user-friendly graphical interface (GUI) built with Tkinter, including a dark theme, audio visualization, and batch processing capabilities.

## Features
*   **Accurate Transcription:** Leverages various sizes of Whisper models for speech-to-text.
*   **Audio & Video Support:** Transcribes a wide range of audio and video file formats.
*   **User-Friendly GUI:**
    *   Intuitive interface for selecting files, models, and processing options.
    *   Dark theme for comfortable viewing.
    *   Mel spectrogram visualization of the audio being processed.
*   **Input Options:**
    *   Process single audio/video files.
    *   Batch process all supported files within a selected directory (including its subfolders).
    *   Drag-and-drop support for easy file/directory selection.
*   **Model Configuration:**
    *   Selection from various Whisper model sizes (Tiny, Base, Small, Medium, Large-v1/v2/v3, and distilled versions) to balance speed and accuracy.
    *   Option to use quantized models (e.g., int8) for reduced size/memory and potentially faster CPU inference.
    *   Automatic download and caching of selected models in an "Audio Models" subfolder.
*   **Processing Control:**
    *   Choose processing device: CPU or GPU (NVIDIA CUDA, if available and PyTorch is correctly set up).
    *   Select compute type (e.g., float16, int8, float32) to optimize for speed and memory.
    *   Enable Voice Activity Detection (VAD) to filter out non-speech segments.
    *   Adjust beam size for decoding (higher values can improve accuracy but are slower).
    *   Option to overwrite existing output files.
*   **Output Formats:**
    *   **TXT (.txt):** Plain text transcription (always generated).
    *   **SRT (.srt):** SubRip subtitle format with timestamps (generated for video files).
    *   **LRC (.lrc):** LyRiCs format with timestamps (generated for audio files).
    *   Output files are saved in the same directory as the input media.
*   **Progress & Status:**
    *   Real-time progress bars for individual file processing and overall batch progress.
    *   Estimated Time Remaining (ETA) for batch operations.
    *   File queue display showing the status of each file (Pending, Processing, Completed, Skipped, Error).
*   **Dependency Management:**
    *   Checks for essential Python libraries on startup.
    *   Menu option to attempt installation/update of core dependencies (e.g., `faster-whisper`, `librosa`).
    *   Menu option to check PyTorch/CUDA setup and open the PyTorch installation website.
*   **Convenience:**
    *   Menu option to easily open the "Audio Models" folder.

## Supported Formats
*   **Audio:** `.wav`, `.mp3`, `.flac`, `.aac`, `.m4a`, `.ogg`
*   **Video:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`, `.flv`

*(Note: Video file processing relies on FFmpeg for audio extraction. Ensure FFmpeg is installed and accessible in your system's PATH for seamless video transcription.)*

## Requirements
*   Python 3.x
*   The following Python libraries (see Installation section):
    *   `faster-whisper`
    *   `librosa`
    *   `numpy`
    *   `matplotlib`
    *   `tkinterdnd2-universal`
    *   `torch` (PyTorch - optional, but required for GPU support and some advanced CPU quantization/compute types)
*   FFmpeg: Required for processing video files (extracting audio). Download from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it's added to your system's PATH.

## Installation
1.  **Clone the repository or download the script.**
2.  **Install Python:** If you don't have Python installed, download it from [python.org](https.python.org).
3.  **Install FFmpeg:**
    *   Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html).
    *   Extract it and add the `bin` directory (containing `ffmpeg.exe` on Windows or `ffmpeg` on Linux/macOS) to your system's PATH environment variable.
4.  **Install Python Dependencies:**
    *   Open a terminal or command prompt.
    *   You can try installing dependencies via the application's menu (**Tools > Install Core Dependencies**).
    *   Alternatively, install them manually using pip:
        ```bash
        pip install faster-whisper librosa numpy matplotlib tkinterdnd2-universal
        ```
    *   **For GPU Support (NVIDIA CUDA):**
        *   Ensure you have a compatible NVIDIA GPU and have the CUDA Toolkit installed.
        *   Install PyTorch with CUDA support. Visit the [PyTorch website](https://pytorch.org/get-started/locally/) and select the appropriate options for your system (e.g., CUDA version). The application menu (**Tools > Check/Install PyTorch**) can also guide you to this site.
        *   Example PyTorch installation command (verify on their site):
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX # Replace cuXXX with your CUDA version, e.g., cu118 or cu121
            ```

## Usage
1.  Run the `Transcribe Audio Video.pyw` script:
    ```bash
    python "Transcribe Audio Video.pyw"
    ```
2.  **Input Source:**
    *   **Single File:** Select this mode, then click "Browse..." to pick an audio/video file, or drag and drop a file onto the application window (or the input path field).
    *   **Batch Directory:** Select this mode, then click "Browse..." to pick a folder, or drag and drop a folder. The application will scan for all supported files in this folder and its subfolders.
3.  **Model Configuration:**
    *   **Model Size:** Choose the desired Whisper model. Larger models are more accurate but slower and require more resources. Models will be downloaded on first use to the `Audio Models` subfolder.
    *   **Quantization:** (Optional) Select a quantization type (e.g., `int8`) to use a model variant that might be faster on CPU and use less memory.
    *   **Device:**
        *   `cpu`: Uses the CPU for processing.
        *   `cuda`: Uses an NVIDIA GPU (if PyTorch with CUDA is correctly installed and a compatible GPU is found).
    *   **Compute Type:** Select the data type for computation (e.g., `float16` for GPU, `int8` or `float32` for CPU). `auto` or `default` usually picks a reasonable option.
4.  **Processing Options:**
    *   **Beam Size:** (Default: 5) Affects decoding. Higher might be more accurate but slower.
    *   **VAD Filter:** Check to enable Voice Activity Detection, which can help by skipping non-speech parts.
    *   **Overwrite Existing Output:** If checked, the application will re-process files and overwrite existing `.txt`, `.srt`, or `.lrc` files. If unchecked (default), files with existing outputs will be skipped.
5.  **Start Transcription:**
    *   Click the "🚀 Start Transcription" button.
    *   The file queue will show the status of each file.
    *   Progress bars will indicate current file and overall batch progress.
    *   A Mel spectrogram of the current audio will be displayed.
6.  **Stop Processing:**
    *   Click the "🛑 Stop Processing" button.
    *   The first click will prompt to stop after the current file.
    *   A second click (if confirmed) will request an immediate stop (transcription for the current file might be incomplete).
7.  **Output:**
    *   Transcription files (`.txt`, `.srt`, `.lrc`) will be saved in the same directory as their respective input media files.

## GPU Support
To use GPU acceleration (NVIDIA CUDA only):
1.  Ensure you have an NVIDIA GPU with up-to-date drivers.
2.  Install the NVIDIA CUDA Toolkit that is compatible with the PyTorch version you intend to use.
3.  Install PyTorch with CUDA support (see Installation section).
4.  In the application, select "cuda" from the "Device" dropdown.
5.  Choose an appropriate "Compute Type" (e.g., `float16` is common for NVIDIA GPUs).

The application menu under **Tools > Check/Install PyTorch (GPU Support)** can help verify if PyTorch is detected and if CUDA is available to it.

## Troubleshooting/Notes
*   **FFmpeg Not Found:** If you get errors related to video files (especially "Error loading audio" or messages about "audioread backend"), it's likely FFmpeg is not installed or not in your system's PATH.
*   **Model Downloads:** The first time you select a specific model size/quantization, it will be downloaded from Hugging Face Hub to the `Audio Models` folder in the script's directory. This might take some time depending on your internet speed and the model size.
*   **Performance:** Transcription speed depends on the model size, hardware (CPU/GPU), selected compute type, audio duration, and complexity.
*   **Memory Usage:** Larger models consume more memory. If you encounter memory issues, try a smaller model or use quantization.
*   **GUI Freezing:** During very intensive operations or model loading, the GUI might become briefly unresponsive. This is normal. The processing itself happens in a separate thread to keep the GUI as responsive as possible.

## UI Theme
The application features a custom "futuristic" dark theme for a visually distinct experience, with accent colors for interactive elements.
