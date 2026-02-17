# Qwen API Video Captioning

This repository contains tools for processing video captions using the Qwen-VL API (via DashScope). It includes scripts for generating captions, managing files, and analyzing results.

## Features

- **Video Captioning**: Generate captions for videos using Qwen-VL models.
- **Data Management**: Clean up orphaned caption files and format JSON outputs.
- **Analysis**: Count combinations of attributes (e.g., weather conditions, hazardous events) in the generated captions.

## Prerequisites

- Python 3.8+
- DashScope API Key (for `main.py`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add your DashScope API key:
   ```
   DASHSCOPE_API_KEY=your_api_key_here
   ```

## Usage

### 1. Generating Captions (`main.py`)

The main script processes videos from the `input` directory and saves JSON captions to the `output` directory.

```bash
python main.py
```

Configuration can be modified in `configs/settings.json` and `configs/prompts.json`.

### 2. Cleaning and Formatting Captions (`clean_and_format_captions.py`)

This utility script performs three main tasks:
1. Deletes JSON caption files that do not have a corresponding video file in the input directory.
2. Fixes common JSON syntax errors (e.g., trailing commas).
3. Removes metadata fields (`usage_metadata`, `file_metadata`) from the JSON files.
4. Formats the JSON with consistent indentation.

**Usage:**
```bash
python clean_and_format_captions.py --captions Human_Captions --input input
```

Arguments:
- `--captions`: Directory containing caption JSON files (default: `Human_Captions`)
- `--input`: Directory containing input video files (default: `input`)

### 3. Analyzing Captions (`count_combinations.py`)

Analyzes the generated captions to count the occurrence of specific attribute combinations (e.g., Winter Weather vs. Hazardous Event presence).

**Usage:**
```bash
python count_combinations.py [directory]
```

Arguments:
- `directory`: Directory containing caption JSON files (default: `Human_Captions`)

## Directory Structure

- `input/`: Directory for input video files (`.mp4`, etc.).
- `output/`: Directory where `main.py` saves generated captions.
- `Human_Captions/`: Directory for manually reviewed or processed captions (used by analysis scripts).
- `configs/`: Configuration files for the captioning process.
- `clean_and_format_captions.py`: Script to clean and format caption files.
- `count_combinations.py`: Script to analyze caption attributes.
- `main.py`: Main script for calling the Qwen-VL API.

## License

[License Name]
