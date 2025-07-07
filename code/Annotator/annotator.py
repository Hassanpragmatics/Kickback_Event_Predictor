import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog

# Config
target_col = 'Analog Input #14 P Backhoff Track'
time_column = 'Time'

annotations = []

class EventAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Annotator")
        self.root.geometry("1200x800")

        # Control buttons
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10, fill=tk.X)
        
        self.load_button = tk.Button(control_frame, text="Load Next CSV", command=self.load_next_file)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(control_frame, text="Save All Annotations", command=self.save_annotations)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.undo_button = tk.Button(control_frame, text="Undo", command=self.undo_last_annotation)
        self.undo_button.pack(side=tk.LEFT, padx=5)

        # Plot area initialization
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        
        # Initialize with empty plot
        self.ax.set_title("Initializing...")
        self.ax.grid(True)
        self.canvas.draw()

        # File handling
        self.input_files = self.find_csv_files('data')
        self.current_file_idx = -1
        self.data = None
        self.file = None
        self.event_start_time = None
        self.event_lines = []

        # Load first file immediately
        if self.input_files:
            self.root.after(100, self.load_next_file)  # Ensures window is fully initialized
        else:
            self.ax.set_title("No CSV files found in 'data' directory!")
            self.canvas.draw()

    def find_csv_files(self, root_dir):
        return sorted([os.path.join(root, f) 
                     for root, _, files in os.walk(root_dir) 
                     for f in files if f.endswith('.csv')])

    def load_next_file(self):
        if not self.input_files:
            return

        self.current_file_idx += 1
        if self.current_file_idx >= len(self.input_files):
            self.current_file_idx = -1
            return

        file_path = self.input_files[self.current_file_idx]
        
        try:
            # Data loading and processing
            df = pd.read_csv(
                file_path,
                encoding='cp1252',
                sep=';',
                skiprows=31,
                dtype=str,
                on_bad_lines='warn'
            )
            
            # Clean and process data
            df.columns = [c.strip() for c in df.columns]
            required_cols = [time_column, target_col]
            
            if not set(required_cols).issubset(df.columns):
                return

            df = df[required_cols]
            
            for col in required_cols:
                df[col] = df[col].str.replace(r'[^\d.,-]', '', regex=True)
                df[col] = df[col].str.replace('.', '', regex=False)
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.fillna(0)
            
            # Plotting
            self.ax.clear()
            line, = self.ax.plot(df[time_column], df[target_col], 
                                color="#0e74bd",  # Modern blue
                                linewidth=1.5,     # Thicker line
                                alpha=0.8)
            self.ax.set_title(f"{os.path.basename(file_path)}", fontsize=12)
            self.ax.set_xlabel("Time", fontsize=10)
            self.ax.set_ylabel("Value", fontsize=10)
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # Handle zero values properly
            y_min, y_max = df[target_col].min(), df[target_col].max()
            if y_min == y_max:
                # If all values are the same, create a reasonable range
                if y_min == 0:
                    y_min, y_max = -1, 1
                else:
                    y_min -= 0.1 * abs(y_min)
                    y_max += 0.1 * abs(y_max)
            else:
                # Add padding to the range
                padding = 0.1 * (y_max - y_min)
                y_min -= padding
                y_max += padding
                
            self.ax.set_ylim(y_min, y_max)
            
            self.canvas.draw()
            
            # Update state
            self.event_start_time = None
            self.event_lines.clear()
            self.data = df
            self.file = file_path

        except Exception as e:
            self.ax.clear()
            self.ax.set_title(f"Error loading {os.path.basename(file_path)}")
            self.canvas.draw()

    def on_click(self, event):
        if event.xdata is None or self.data is None:
            return

        click_time = event.xdata

        if self.event_start_time is None:
            self.event_start_time = click_time
            line = self.ax.axvline(click_time, color='blue', linestyle='--')
            self.event_lines.append(line)
            self.canvas.draw()
        else:
            start = min(self.event_start_time, click_time)
            end = max(self.event_start_time, click_time)
            
            # Add annotation to global list
            annotations.append({
                'filename': os.path.basename(self.file),
                'file_path': os.path.relpath(self.file, 'data'),
                'start_time': start,
                'end_time': end,
                'duration': end - start
            })
            
            span = self.ax.axvspan(start, end, color='red', alpha=0.3)
            self.event_lines.append(span)
            self.canvas.draw()
            self.event_start_time = None

    def undo_last_annotation(self):
        if self.event_lines:
            last = self.event_lines.pop()
            last.remove()
            if annotations:
                annotations.pop()
            self.canvas.draw()

    def save_annotations(self):
        if not annotations:
            return
            
        # Create DataFrame from all annotations
        df = pd.DataFrame(annotations)
        
        # Ask for save location once
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save All Annotations"
        )
        
        if save_path:
            # Save all annotations in one file
            df.to_csv(save_path, index=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = EventAnnotator(root)
    root.mainloop()