# Signal Annotator

**Signal Annotator** is a Python GUI application for visualizing time-series data and annotating events in signals. Built with Tkinter and Matplotlib.

## Features

- Load and visualize CSV time-series data  
- Annotate events by selecting start/end times  
- Undo last annotation  
- Save all annotations to CSV  
- Batch processing of multiple files  
- Interactive plotting with Matplotlib

## Requirements

- Python 3.6+
- Libraries:
  - `pandas`
  - `matplotlib`
  - `tkinter` (usually included with standard Python)

## Installation

Clone the repository:

```bash
git clone https://github.com/Hassanpragmatics/Kickback_Event_Predictor.git
cd code
cd Annotator
```
Install required dependencies:

```bash
pip install pandas matplotlib
```
## Usage

- Create a data/ directory (if it doesn't exist) and place your CSV time-series files inside.

- Run the application:
```bash
python annotator.py
```
Use the interface:
- Click "Load Next CSV" to load files sequentially
- Click on the plot to mark event start/end times
- Click "Undo" to remove the last annotation
- Click "Save All Annotations" to export annotations to CSV

## File Structure
```bash
Annotator-root/
├── data/            # Folder for input CSV files
├── annotator.py     # Main application code
└── README.md        # This documentation
```
## How to Run:
- Organize your CSV files:
- Create data/ folder in same directory as script
- Place CSV files in data/ (can have subdirectories)

## File format requirements:
- Must contain columns named exactly:
- Time (time values)
- Analog Input #14 P Backhoff Track (signal values)
- Uses semicolon (;) as delimiter
- First 31 rows are skipped (header information)

## Run the script:

```bash
python annotator.py
```

## Usage Tips

- Click sequentially to mark event start and end

- Files are loaded in alphabetical order

- Annotations persist between files until saved

- Adjust target_col in code for different signals

- Zero-value signals automatically get adjusted Y-axis

- Error messages appear directly on the plot

## Customization: 
Modify these variables for different datasets:

```bash
target_col = 'Your_Signal_Column_Name'
time_column = 'Your_Time_Column_Name'
Adjust CSV reading parameters if needed:

# In load_next_file method:
pd.read_csv(..., skiprows=31, sep=';', encoding='cp1252')
```
