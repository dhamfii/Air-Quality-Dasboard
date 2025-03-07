# Air Quality Dashboard

## Overview
This interactive dashboard allows users to monitor and analyze air quality parameters across different locations. Built with Streamlit and Plotly, it provides a user-friendly interface to visualize air quality data, including Air Quality Index (AQI) values and individual pollutant concentrations.

## Features
- **Interactive Filtering**: Select cities, date ranges, and AQI ranges to focus your analysis
- **Multi-tab Analysis**:
  - AQI Overview: Visualize the distribution of AQI values and categories by city
  - Pollutant Analysis: Explore specific pollutants and their correlation with AQI
  - Correlation Analysis: Understand relationships between different air quality parameters
  - City Comparison: Compare air quality metrics across different cities
- **Customizable Visualizations**: Adjust plot heights, toggle dark mode, and set pollutant thresholds
- **Data Export**: Download filtered data as CSV or Excel files with summary statistics
- **Responsive Design**: Works well on both desktop and mobile devices

## Data Requirements
The dashboard expects a CSV file named `air_qualitydata.csv` with the following columns:
- `City`: Name of the city/location
- `AQI`: Air Quality Index value
- `Date` (optional): Date of measurement
- Pollutant columns: Measurements for various pollutants such as PM2.5, PM10, NO2, SO2, etc.

If the `AQI_Category` column is not present, the application will automatically categorize AQI values according to standard ranges:
- Good (0-50)
- Satisfactory (51-100)
- Moderate (101-200)
- Poor (201-300)
- Very Poor (301-400)
- Severe (>400)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Setup
1. Clone this repository or download the source files
```bash
git clone https://github.com/yourusername/air-quality-dashboard.git
cd air-quality-dashboard
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
```

3. Activate the virtual environment
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

4. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage

1. Place your air quality data CSV file in the project directory with the name `air_qualitydata.csv`

2. Run the Streamlit application
```bash
streamlit run AQI_app.py
```

3. The dashboard will open in your default web browser. If it doesn't, navigate to the URL shown in the terminal (typically http://localhost:8501)

4. Click the "Load Data" button in the sidebar to load and preprocess your data

5. Use the sidebar controls to filter and analyze your air quality data

## Application Structure

- `AQI_app.py`: Main application file
- `requirements.txt`: List of required Python packages
- `air_qualitydata.csv`: Your air quality data file (not included in the repository)

## Data Preprocessing

The application performs several preprocessing steps on the loaded data:
1. Removes rows with missing AQI values
2. Fills other missing values with column means
3. Handles outliers using the IQR method
4. Creates AQI categories if not present in the original data

## Customization

You can customize the dashboard by modifying:
- The pollutant groups in the `pollutant_groups` dictionary
- The AQI category definitions and colors
- Default selections and visualization parameters

## Troubleshooting

- **Missing data file**: Ensure your CSV file is named `air_qualitydata.csv` and is in the same directory as the script
- **Column errors**: Verify that your data has the required columns, especially 'City' and 'AQI'
- **Performance issues**: For large datasets, consider pre-aggregating your data or using a smaller subset for analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
