import os
import pandas as pd
import plotly.graph_objects as go

# Base directories
input_root = 'data'
output_root = 'result1'

# Columns
sensor_cols = [
    'TRIAX X Zeitsignal', 'Analog Input #14 P Backhoff Track', 'TRIAX Y Zeitsignal', 'TRIAX Z Zeitsignal', 'Leistung_230', 'U1', 'I1'
]
time_column = 'Time'

# Walk through all subdirectories and files
for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith('.csv'):
            input_path = os.path.join(root, file)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            # Create the corresponding output path
            relative_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, relative_path).replace('.csv', '.html')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print(f"Processing: {input_path}")

            # Read the CSV correctly
            df = pd.read_csv(
                input_path,
                encoding="cp1252",
                sep=";",
                skiprows=31  # Skip until real header row
            )

            # Clean up unnamed columns and fix column names
            df.columns = [col.strip() for col in df.columns]
            df = df.dropna(subset=["Time"])  # Drop empty rows

            # Show the actual columns so we know what we're working with
            print(df.columns.tolist())
            print(df.head())


            sensor_cols = [
            'TRIAX X Zeitsignal', 'Analog Input #14 P Backhoff Track', 'TRIAX Y Zeitsignal', 'TRIAX Z Zeitsignal', 'Leistung_230', 'U1', 'I1'
            ]

            # Identify columns that are time or non-numeric (e.g., 'Time', 'U1', 'I1', etc.)
            time_column = 'Time'

            # Replace commas with periods for numeric columns
            df_cleaned = df.copy()

            # Replace commas for all string-like columns
            df_cleaned[df_cleaned.select_dtypes(include=['object']).columns] = df_cleaned[df_cleaned.select_dtypes(include=['object']).columns].apply(lambda x: x.str.replace(',', '.'))

            # Drop unnamed columns
            df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains('^Unnamed')]

            # Convert all columns (except Time) to numeric
            df_cleaned[df_cleaned.columns.difference([time_column])] = df_cleaned[df_cleaned.columns.difference([time_column])].apply(pd.to_numeric, errors='coerce')

            # Fill NaNs with 0
            df_cleaned = df_cleaned.fillna(0)

            # Keep the original Time column
            df_cleaned[time_column] = df[time_column]

            # Time conversion
            df_cleaned[time_column] = df_cleaned[time_column].str.replace(',', '.')
            df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True)
            df_cleaned[time_column] = pd.to_timedelta(df_cleaned[time_column])

            ### --------- Normalize sensor values --------- ###
            # Create a normalized copy of sensor data
            df_normalized = df_cleaned.copy()

            for col in sensor_cols:
                if col in df_normalized.columns:
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val - min_val != 0:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                    else:
                        df_normalized[col] = 0  # If constant value, set to 0

            # Create the plot
            fig = go.Figure()

            # Add each sensor column as a separate trace
            for col in sensor_cols:
                if col in df_cleaned.columns:
                    fig.add_trace(go.Scatter(
                        x=df_cleaned[time_column],
                        y=df_normalized[col],
                        mode='lines',
                        name=col,
                        visible=True
                    ))

            # Add interactive buttons to toggle traces
            fig.update_layout(
                title="Sensor Data Over Time",
                xaxis_title="Time",
                yaxis_title="Sensor Values",
                template="plotly_white",
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "label": "All",
                                "method": "update",
                                "args": [{"visible": [True] * len(sensor_cols)},
                                        {"title": "All Sensors"}]
                            }
                        ] + [
                            {
                                "label": col,
                                "method": "update",
                                "args": [
                                    {"visible": [c == col for c in sensor_cols]},
                                    {"title": f"{col} Over Time"}
                                ]
                            }
                            for col in sensor_cols
                        ],
                        "direction": "down",
                        "showactive": True,
                        "x": 1.15,
                        "xanchor": "left",
                        "y": 1,
                        "yanchor": "top"
                    }
                ],
                legend=dict(x=0.90, y=0.99),
                height=1000,
                width=1000
            )

            # Show plot
            fig.show()
            fig.write_html(output_path)
