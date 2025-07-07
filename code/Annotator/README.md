    # Signal Annotator

    ![Signal Annotator Screenshot](https://via.placeholder.com/800x400?text=Signal+Annotator+Screenshot)

    **Signal Annotator** is a Python GUI application for visualizing time-series data and annotating events in signals. Built with Tkinter and Matplotlib.

    ---

    ## Features

    - Load and visualize CSV time-series data  
    - Annotate events by selecting start/end times  
    - Undo last annotation  
    - Save all annotations to CSV  
    - Batch processing of multiple files  
    - Interactive plotting with Matplotlib

    ---

    ## Requirements

    - Python 3.6+
    - Libraries:
      - `pandas`
      - `matplotlib`
      - `tkinter` (usually included with standard Python)

    ---

    ## Installation

    Clone the repository:

    ```bash
    git clone https://github.com/yourusername/signal-annotator.git
    cd signal-annotator
    ```

    Install required dependencies:

    ```bash
    pip install pandas matplotlib
    ```

    ---

    ## Usage

    1. Create a `data/` directory (if it doesn't exist) and place your CSV time-series files inside.
    2. Run the application:

       ```bash
       python annotator.py
       ```

    3. Use the interface:
       - Click **"Load Next CSV"** to load files sequentially
       - Click on the plot to mark **event start/end times**
       - Click **"Undo"** to remove the last annotation
       - Click **"Save All Annotations"** to export annotations to CSV

    ---

    ## File Structure

    ```
    Annotator-root/
    ├── data/            # Folder for input CSV files
    ├── annotator.py     # Main application code
    └── README.md        # This documentation
    ```
