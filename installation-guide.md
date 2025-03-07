# Air Quality Dashboard Installation Guide

This guide provides detailed instructions for setting up the environment and running the Air Quality Dashboard application. Follow these steps to get the dashboard working on your system.

## System Requirements

- Python 3.7 or higher
- Internet connection for downloading packages
- At least 4GB of free RAM (recommended)
- 500MB of free disk space

## Installation Steps

### 1. Install Python

If you don't already have Python installed:

**For Windows:**
1. Download the latest Python installer from [python.org](https://www.python.org/downloads/)
2. Run the installer, checking "Add Python to PATH" during installation
3. Verify installation by opening Command Prompt and typing: `python --version`

**For macOS:**
1. Install Homebrew if not already installed:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python using Homebrew:
   ```bash
   brew install python
   ```
3. Verify installation: `python3 --version`

**For Linux (Ubuntu/Debian):**
1. Update package lists:
   ```bash
   sudo apt update
   ```
2. Install Python and pip:
   ```bash
   sudo apt install python3 python3-pip
   ```
3. Verify installation: `python3 --version`

### 2. Set Up a Virtual Environment (Recommended)

A virtual environment keeps dependencies for different projects separate. This step is highly recommended but optional.

**For Windows:**
```bash
# Navigate to your project directory
cd path\to\air-quality-dashboard

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
# Navigate to your project directory
cd path/to/air-quality-dashboard

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

After activation, your command prompt should show `(venv)` at the beginning, indicating the virtual environment is active.

### 3. Install Required Libraries

With your virtual environment activated, install all required packages using pip:

```bash
# Install packages listed in requirements.txt
pip install -r requirements.txt
```

If you encounter any errors during installation:

- For Windows users, you might need Microsoft C++ Build Tools for some packages:
  ```bash
  pip install --upgrade setuptools wheel
  ```

- For macOS users, you might need to install Xcode command-line tools:
  ```bash
  xcode-select --install
  ```

### 4. Verify Installations

Ensure all libraries were correctly installed:

```bash
pip list
```

You should see all the packages listed in the requirements.txt file with their respective versions.

## Preparing Your Data

1. Ensure your air quality dataset is in CSV format
2. Name the file `air_qualitydata.csv`
3. Place this file in the same directory as `AQI_app.py`

The CSV file should include at minimum:
- A `City` column with location names
- An `AQI` column with Air Quality Index values
- Columns for various pollutants (PM2.5, PM10, NO2, etc.)

## Running the Streamlit Application

### 1. Start the Application

With your virtual environment still activated:

```bash
# Navigate to the directory containing AQI_app.py (if you're not already there)
cd path/to/air-quality-dashboard

# Launch the Streamlit application
streamlit run AQI_app.py
```

### 2. Access the Dashboard

After running the command:

1. Streamlit will start a local server
2. Your default web browser should automatically open with the dashboard
3. If the browser doesn't open automatically, you'll see a URL in the terminal (typically http://localhost:8501) - copy and paste this URL into your browser

### 3. Using the Dashboard

1. Click the "Load Data" button in the sidebar
2. Use the sidebar controls to select cities, date ranges, and pollutants of interest
3. Explore the different tabs to analyze your air quality data

## Troubleshooting Common Issues

### Application Won't Start

**Error: "No module named 'streamlit'"**  
Solution: Ensure you've activated your virtual environment and installed all requirements:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Error: "Port 8501 is already in use"**  
Solution: Find and close any existing Streamlit processes, or specify a different port:
```bash
streamlit run AQI_app.py --server.port 8502
```

### Data Loading Issues

**Error: "File 'air_qualitydata.csv' not found"**  
Solution: Ensure your data file is named correctly and in the same directory as the script.

**Error related to missing columns**  
Solution: Check that your CSV has the required columns (especially 'City' and 'AQI').

### Memory Issues

If you encounter memory errors with large datasets:
1. Try reducing the amount of data by pre-filtering
2. Increase your system's swap space/virtual memory
3. Run the application on a machine with more RAM

## Stopping the Application

To stop the running application:
1. Press `Ctrl+C` in the terminal/command prompt where Streamlit is running
2. Close the browser tab with the dashboard

## Deactivating the Virtual Environment

When you're done working with the dashboard:

```bash
deactivate
```

This will return you to your global Python environment.

## Updating the Application

To update the application when new versions are available:

1. Pull the latest code (if using Git):
   ```bash
   git pull origin main
   ```

2. Activate the virtual environment and update packages:
   ```bash
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt --upgrade
   ```

## Getting Help

If you encounter persistent issues:
- Check the Streamlit documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
- Search for specific error messages online
- File an issue in the project's repository
