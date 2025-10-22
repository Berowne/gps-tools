# GPS Data Processor

A Python tool to read and process various GPS data file formats including GPX, KML, and more.

## Supported Formats

- GPX (.gpx) - GPS Exchange Format
- KML/KMZ (.kml, .kmz) - Keyhole Markup Language
- FIT (.fit) - Garmin FIT format
- TCX (.tcx) - Garmin Training Center XML

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

```bash
python gps_processor.py <directory_path>
```

Example:
```bash
python gps_processor.py "C:\Path\To\GPS\Files"
```

## Output

The tool will:
1. Scan the specified directory for supported GPS files
2. Parse each file and extract relevant data
3. Display a summary of the processed files
4. Save detailed results to `gps_data_summary.json`

## Example Output

```
Processed 3 GPS files:

track1.gpx
  Waypoints: 5
  Tracks: 2
  Routes: 0

route.kml
  Features: 3

workout.tcx
  Activities: 1
  Tracks: 1
```

## Extending the Tool

To add support for additional file formats, create a new method in the `GPSDataProcessor` class following the pattern of the existing methods (e.g., `read_gpx`, `read_kml`). Then update the `process_all_files` method to handle the new file extension.
