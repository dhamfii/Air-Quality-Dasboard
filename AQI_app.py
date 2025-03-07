import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.impute import SimpleImputer
from warnings import filterwarnings
import datetime
import io

filterwarnings('ignore')

# Setting page configuration
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# Application title and description
st.title("Interactive Air Quality Dashboard")
st.markdown("""
This dashboard allows you to monitor air quality parameters for different locations.
Select cities and pollutants to visualize air quality data including AQI and individual pollutants.
""")

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Loads the air quality data and performs preprocessing steps including:
    - Handling missing values
    - Removing outliers
    - Preparing data for visualization
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Drop rows where 'AQI' has missing values
    df.dropna(subset=['AQI'], inplace=True)
    
    # Fill missing values with mean for numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    # Create a dictionary of means for each column
    mean_values = df[numeric_columns].mean().to_dict()
    
    # Replace missing values with means
    for col, mean_val in mean_values.items():
        df[col] = df[col].fillna(mean_val)
    
    # Handle outliers using IQR method
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].apply(
            lambda x: Q1 if x < lower_bound else (Q3 if x > upper_bound else x)
        )
    
    # Create categories for AQI if not already present
    if 'AQI_Category' not in df.columns:
        # Define AQI categories based on standard ranges
        conditions = [
            (df['AQI'] <= 50),
            (df['AQI'] > 50) & (df['AQI'] <= 100),
            (df['AQI'] > 100) & (df['AQI'] <= 200),
            (df['AQI'] > 200) & (df['AQI'] <= 300),
            (df['AQI'] > 300) & (df['AQI'] <= 400),
            (df['AQI'] > 400)
        ]
        
        categories = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
        df['AQI_Category'] = np.select(conditions, categories, default='Unknown')
    
    return df

# Helper function to convert NumPy types to Python native types
def to_python_type(value):
    """
    Convert NumPy types to Python native types.
    This is necessary because Streamlit's metric() function only accepts 
    int, float, str, or None for the delta parameter.
    """
    if isinstance(value, (np.integer, np.floating)):
        return value.item()  # .item() converts numpy scalar to Python scalar
    return value

# Sidebar for controls
st.sidebar.header("Controls")

# Load data button
data_path = "air_qualitydata.csv"
if st.sidebar.button("Load Data"):
    try:
        df = load_and_preprocess_data(data_path)
        st.session_state.df = df
        st.sidebar.success("Data loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        st.stop()

# Check if data is already loaded
if 'df' not in st.session_state:
    st.info("Please click 'Load Data' in the sidebar to begin.")
    st.stop()

df = st.session_state.df

# Define pollutant groups for dropdown selection
pollutant_groups = {
    'Particulate Matter': ['PM2.5', 'PM10'],
    'Nitrogen Compounds': ['NO', 'NO2', 'NOx', 'NH3'],
    'Other Gases': ['O3', 'SO2', 'CO'],
    'Volatile Compounds': ['Benzene', 'Toluene', 'Xylene']
}

# Flatten the groups for individual pollutant selection
all_pollutants = [item for sublist in pollutant_groups.values() for item in sublist]

# Get unique cities for dropdown
cities = sorted(df['City'].unique())

# City selection
selected_cities = st.sidebar.multiselect("Select Cities", options=cities, default=cities[:3] if len(cities) >= 3 else cities)

# Date range selection if available
if 'Date' in df.columns:
    min_date = pd.to_datetime(df['Date'].min())
    max_date = pd.to_datetime(df['Date'].max())
    selected_date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

# AQI range selection
min_aqi = int(df['AQI'].min())
max_aqi = int(df['AQI'].max())
selected_aqi_range = st.sidebar.slider(
    "AQI Range",
    min_value=min_aqi,
    max_value=max_aqi,
    value=(min_aqi, max_aqi)
)

# Pollutant selection
st.sidebar.subheader("Pollutant Selection")
selected_pollutant_group = st.sidebar.selectbox(
    "Select Pollutant Group",
    options=list(pollutant_groups.keys()),
    index=0
)

# Update individual pollutant options based on group
selected_pollutants = pollutant_groups[selected_pollutant_group]
selected_pollutant = st.sidebar.selectbox(
    "Select Individual Pollutant",
    options=selected_pollutants,
    index=0
)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    pollutant_threshold = st.slider("Danger Threshold (%)", 0, 100, 70)
    plot_height = st.slider("Plot Height", 300, 800, 400)
    dark_mode = st.checkbox("Dark Mode", True)

# Filter the data based on selections
filtered_df = df.copy()

if len(selected_cities) > 0:
    filtered_df = filtered_df[filtered_df['City'].isin(selected_cities)]

filtered_df = filtered_df[(filtered_df['AQI'] >= selected_aqi_range[0]) & (filtered_df['AQI'] <= selected_aqi_range[1])]

# Convert the date objects to pandas datetime
if 'Date' in df.columns and len(selected_date_range) == 2:
    start_date = pd.to_datetime(selected_date_range[0])
    end_date = pd.to_datetime(selected_date_range[1])
    filtered_df = filtered_df[(pd.to_datetime(filtered_df['Date']) >= start_date) & 
                              (pd.to_datetime(filtered_df['Date']) <= end_date)]
    

if filtered_df.empty:
    st.error("No data available with the current filter settings. Please adjust your selections.")
    st.stop()

# Display city information and key metrics
st.header(f"Air Quality Analysis for Selected Cities")

# Create columns for summary metrics
col1, col2, col3, col4 = st.columns(4)

# Current AQI stats
latest_aqi = to_python_type(round(filtered_df['AQI'].mean(), 1))
with col1:
    st.metric(
        "Average AQI", 
        latest_aqi,
        delta_color="inverse"
    )
    # Get the category for the average AQI
    conditions = [
        (latest_aqi <= 50),
        (latest_aqi > 50) & (latest_aqi <= 100),
        (latest_aqi > 100) & (latest_aqi <= 200),
        (latest_aqi > 200) & (latest_aqi <= 300),
        (latest_aqi > 300) & (latest_aqi <= 400),
        (latest_aqi > 400)
    ]
    categories = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    colors = ['green', 'yellowgreen', 'orange', 'red', 'purple', 'maroon']
    aqi_category = np.select(conditions, categories, default='Unknown')
    aqi_color = np.select(conditions, colors, default='gray')
    st.markdown(f"<span style='color:{aqi_color}'>{aqi_category}</span>", unsafe_allow_html=True)

# Highest AQI
with col2:
    max_aqi = to_python_type(filtered_df['AQI'].max())
    max_aqi_city = filtered_df.loc[filtered_df['AQI'].idxmax(), 'City']
    st.metric("Highest AQI", max_aqi)
    st.markdown(f"City: {max_aqi_city}")

# Lowest AQI
with col3:
    min_aqi = to_python_type(filtered_df['AQI'].min())
    min_aqi_city = filtered_df.loc[filtered_df['AQI'].idxmin(), 'City']
    st.metric("Lowest AQI", min_aqi)
    st.markdown(f"City: {min_aqi_city}")

# Total measurements
with col4:
    total_records = len(filtered_df)
    measured_cities = len(filtered_df['City'].unique())
    st.metric("Total Measurements", total_records)
    st.markdown(f"Cities: {measured_cities}")

# AQI level descriptions
aqi_levels = {
    'Good': {"color": "green", "description": "Air quality is satisfactory, and air pollution poses little or no risk."},
    'Satisfactory': {"color": "yellowgreen", "description": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."},
    'Moderate': {"color": "orange", "description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."},
    'Poor': {"color": "red", "description": "Health alert: The risk of health effects is increased for everyone."},
    'Very Poor': {"color": "purple", "description": "Health warning of emergency conditions: everyone is more likely to be affected."},
    'Severe': {"color": "maroon", "description": "Health emergency: everyone may experience more serious health effects."}
}

# Main tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["AQI Overview", "Pollutant Analysis", "Correlation Analysis", "City Comparison"])

with tab1:
    st.subheader("Air Quality Index Distribution")
    
    # AQI Histogram
    fig_hist = px.histogram(
        filtered_df, 
        x='AQI', 
        color='AQI_Category',
        title='Distribution of Air Quality Index (AQI)',
        labels={'AQI': 'Air Quality Index'},
        color_discrete_sequence=px.colors.qualitative.Set3,
        height=plot_height
    )
    fig_hist.update_layout(
        bargap=0.1,
        template='plotly_dark' if dark_mode else 'plotly_white'
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # AQI Categories by City
    st.subheader("AQI Categories by City")
    fig_cat = px.bar(
        filtered_df.groupby(['City', 'AQI_Category']).size().reset_index(name='Count'),
        x='City',
        y='Count',
        color='AQI_Category',
        title='AQI Categories by City',
        labels={'Count': 'Number of Readings'},
        color_discrete_sequence=px.colors.qualitative.Set3,
        height=plot_height
    )
    fig_cat.update_layout(
        xaxis={'categoryorder':'total descending'},
        template='plotly_dark' if dark_mode else 'plotly_white'
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    
    # AQI level descriptions
    with st.expander("AQI Level Descriptions"):
        for level, info in aqi_levels.items():
            st.markdown(f"**{level}** - {info['description']}")

with tab2:
    st.subheader("Pollutant Analysis")
    
    # Pollutant Distribution by City
    fig_poll = px.box(
        filtered_df,
        x='City',
        y=selected_pollutant,
        title=f'{selected_pollutant} Distribution by City',
        color='City',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        height=plot_height
    )
    fig_poll.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig_poll, use_container_width=True)
    
    # Pollutant Correlation with AQI
    fig_corr = px.scatter(
        filtered_df,
        x=selected_pollutant,
        y='AQI',
        color='City',
        title=f'Correlation between {selected_pollutant} and AQI',
        trendline='ols',
        trendline_color_override='black',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        height=plot_height
    )
    fig_corr.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Pollutant statistics
    st.subheader(f"Statistics for {selected_pollutant}")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Average", f"{to_python_type(filtered_df[selected_pollutant].mean()):.2f}")
    
    with stats_col2:
        st.metric("Maximum", f"{to_python_type(filtered_df[selected_pollutant].max()):.2f}")
    
    with stats_col3:
        st.metric("Minimum", f"{to_python_type(filtered_df[selected_pollutant].min()):.2f}")
    
    with stats_col4:
        threshold = filtered_df[selected_pollutant].max() * (pollutant_threshold / 100)
        pct_above = to_python_type((filtered_df[selected_pollutant] > threshold).mean() * 100)
        st.metric("% Above Threshold", f"{pct_above:.1f}%")
    
    # Compare multiple pollutants
    st.subheader("Compare Multiple Pollutants")
    multi_pollutants = st.multiselect(
        "Select Pollutants to Compare",
        options=all_pollutants,
        default=[all_pollutants[0], all_pollutants[1]] if len(all_pollutants) > 1 else all_pollutants[:1]
    )
    
    if multi_pollutants:
        # Option to normalize values
        normalize = st.checkbox("Normalize Values (0-100% scale)", True)
        
        if normalize:
            # Create a normalized dataframe
            norm_df = filtered_df.copy()
            
            for poll in multi_pollutants:
                min_val = norm_df[poll].min()
                max_val = norm_df[poll].max()
                if max_val > min_val:  # Avoid division by zero
                    norm_df[poll] = ((norm_df[poll] - min_val) / (max_val - min_val)) * 100
            
            fig_multi = px.line(
                norm_df.groupby('City')[multi_pollutants].mean().reset_index(),
                x='City',
                y=multi_pollutants,
                labels={"value": "Normalized Value (%)"},
                height=plot_height
            )
            
            fig_multi.update_layout(
                title='Normalized Pollutant Comparison by City',
                template='plotly_dark' if dark_mode else 'plotly_white'
            )
        else:
            fig_multi = make_subplots(specs=[[{"secondary_y": True}]])
            
            city_avg = filtered_df.groupby('City')[multi_pollutants].mean().reset_index()
            
            for i, poll in enumerate(multi_pollutants):
                secondary_y = i > 0
                
                fig_multi.add_trace(
                    go.Bar(
                        x=city_avg['City'],
                        y=city_avg[poll],
                        name=poll
                    ),
                    secondary_y=secondary_y
                )
            
            fig_multi.update_layout(
                title='Pollutant Comparison by City',
                height=plot_height,
                template='plotly_dark' if dark_mode else 'plotly_white'
            )
        
        st.plotly_chart(fig_multi, use_container_width=True)

with tab3:
    st.subheader("Correlation Analysis")
    
    # Select pollutants for correlation
    corr_pollutants = st.multiselect(
        "Select Pollutants for Correlation Analysis",
        options=['AQI'] + all_pollutants,
        default=['AQI'] + all_pollutants[:5] if len(all_pollutants) >= 5 else ['AQI'] + all_pollutants
    )
    
    if corr_pollutants:
        # Calculate correlation matrix
        corr_matrix = filtered_df[corr_pollutants].corr()
        
        # Create heatmap
        fig_corr_matrix = px.imshow(
            corr_matrix,
            text_auto='.2f',
            labels=dict(x="Parameter", y="Parameter", color="Correlation"),
            color_continuous_scale='RdBu_r',
            height=600
        )
        
        fig_corr_matrix.update_layout(
            title="Correlation Between Air Quality Parameters",
            template='plotly_dark' if dark_mode else 'plotly_white'
        )
        
        st.plotly_chart(fig_corr_matrix, use_container_width=True)
        
        # Scatter plot analysis
        st.subheader("Parameter Correlation Scatter Plot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox(
                "Select X-axis Parameter",
                options=corr_pollutants,
                index=0 if 'AQI' in corr_pollutants else 0
            )
        
        with col2:
            y_param = st.selectbox(
                "Select Y-axis Parameter",
                options=corr_pollutants,
                index=1 if len(corr_pollutants) > 1 and 'AQI' in corr_pollutants else 0
            )
        
        if x_param != y_param:
            # Create scatter plot
            fig_scatter = px.scatter(
                filtered_df,
                x=x_param,
                y=y_param,
                color='City',
                trendline="ols",
                title=f"Relationship Between {x_param} and {y_param}",
                height=plot_height
            )
            
            fig_scatter.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Please select different parameters for X and Y axes.")

with tab4:
    st.subheader("City Comparison")
    
    # City comparison for AQI
    fig_city_comp = px.bar(
        filtered_df.groupby('City')['AQI'].mean().reset_index(),
        x='City',
        y='AQI',
        title='Average AQI by City',
        color='City',
        height=plot_height
    )
    fig_city_comp.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig_city_comp, use_container_width=True)
    
    # City comparison for selected pollutant
    fig_city_poll = px.bar(
        filtered_df.groupby('City')[selected_pollutant].mean().reset_index(),
        x='City',
        y=selected_pollutant,
        title=f'Average {selected_pollutant} by City',
        color='City',
        height=plot_height
    )
    fig_city_poll.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig_city_poll, use_container_width=True)
    
    # City ranking table
    st.subheader("City Ranking by Air Quality")
    
    # Calculate rankings
    city_ranks = filtered_df.groupby('City').agg({
        'AQI': 'mean',
        selected_pollutant: 'mean'
    }).reset_index()
    
    city_ranks['AQI_Rank'] = city_ranks['AQI'].rank()
    city_ranks[f'{selected_pollutant}_Rank'] = city_ranks[selected_pollutant].rank()
    city_ranks['Overall_Rank'] = (city_ranks['AQI_Rank'] + city_ranks[f'{selected_pollutant}_Rank']) / 2
    
    city_ranks = city_ranks.sort_values('Overall_Rank')
    
    # Format for display
    display_ranks = city_ranks[['City', 'AQI', selected_pollutant, 'Overall_Rank']]
    display_ranks.columns = ['City', 'Average AQI', f'Average {selected_pollutant}', 'Overall Rank']
    display_ranks['Overall Rank'] = display_ranks['Overall Rank'].round(1)
    
    st.dataframe(
        display_ranks,
        hide_index=True,
        use_container_width=True
    )

# Data download section
st.header("Download Data")
col1, col2 = st.columns(2)

with col1:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"air_quality_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    # Create Excel file
    excel_file = io.BytesIO()
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        filtered_df.to_excel(writer, sheet_name='AQI Data', index=False)
        
        # Create summary sheet
        summary_data = pd.DataFrame({
            'Metric': ['Average AQI', 'Max AQI', 'Min AQI'] + 
                     [f'Average {poll}' for poll in all_pollutants[:10]],  # Limit to first 10 pollutants
            'Value': [to_python_type(filtered_df['AQI'].mean()), 
                      to_python_type(filtered_df['AQI'].max()), 
                      to_python_type(filtered_df['AQI'].min())] +
                     [to_python_type(filtered_df[poll].mean()) for poll in all_pollutants[:10]]
        })
        
        summary_data.to_excel(writer, sheet_name='Summary', index=False)
    
    excel_file.seek(0)
    
    st.download_button(
        "Download Excel",
        data=excel_file,
        file_name=f"air_quality_data_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.ms-excel"
    )

# Raw data view
with st.expander("View Raw Data"):
    st.dataframe(
        filtered_df.style.highlight_max(axis=0, subset=['AQI'] + all_pollutants[:10]),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("Air Quality Dashboard | Created with Streamlit and Plotly")