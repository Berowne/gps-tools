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

### Time range filter and timezone offset

- **Local time range filter**: Limit processing to a clock-time window using `--time-range HH:MM-HH:MM`.
  - Example: `--time-range 12:34-13:45`
  - Windows that cross midnight are supported, e.g., `--time-range 23:30-00:30`.

- **Timezone offset**: Choose the local timezone for interpreting the time window with `--timezone`.
  - Accepts formats like `GMT+11`, `GMT-2`, or with minutes `GMT+10:30` (also `UTC+...` works).
  - Defaults to your system timezone if not specified.

Examples:
```bash
# Filter between 1:00pm and 1:30pm in GMT+11, output HTML next to the input folder
python gps_processor.py ".\sample\" --time-range 13:00-13:30 --timezone GMT+11

# Crossing midnight in UTC
python gps_processor.py ".\sample\" --time-range 23:30-00:30 --timezone UTC
```

## Output

The tool will:
1. Scan the specified directory for supported GPS files
2. Parse each file and extract relevant data
3. Display a summary of the processed files
4. Save detailed results to `gps_data_summary.json`
5. Save a sortable HTML table to `<directory>/gps_metrics.html` by default (override with `--html`)

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
