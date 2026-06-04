# DailyMed Pharmaceutical Packaging Artwork Scraper

This is a complete scraping pipeline that automatically downloads pharmaceutical packaging artwork images from DailyMed. It uses multi-threading and image hashing to find and deduplicate relevant artwork.

## Features
- Multi-threaded downloading for fast fetching
- Automatic resumption of interrupted downloads
- Smart duplicate detection using perceptual hashing (`imagehash`)
- Rejection of tiny logos, thumbnails, and irrelevant UI elements
- Extensive logging to keep track of scraper progress

## Requirements
- Python 3.11+
- See `requirements.txt` for package dependencies

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the pipeline with `main.py`. The scraper can be configured to read a list of SETIDs. 

```bash
python src/main.py
```

## Directory Structure
- `src/` : Source code
- `output/images/` : Downloaded artwork images
- `output/metadata/` : Output CSV files (`images.csv`, `duplicates.csv`)
- `output/logs/` : Detailed execution logs
