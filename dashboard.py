import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Housing Price Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'data/raw/Housing.csv')
    df = pd.read_csv(csv_path)
    return df

df = load_data()

# Sidebar
st.sidebar.title("🏠 Housing Dashboard")
st.sidebar.markdown("---")

# Page selection
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Data Analysis", "Feature Exploration", "Statistics"]
)

# ==================== PAGE: OVERVIEW ====================
if page == "Overview":
    st.title("🏠 Housing Price Prediction Dashboard")
    st.markdown("Explore the housing dataset with interactive visualizations")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(df), delta=None)
    
    with col2:
        st.metric("Avg Price", f"${df['price'].mean():,.0f}", delta=None)
    
    with col3:
        st.metric("Price Range", f"${df['price'].min():,.0f} - ${df['price'].max():,.0f}", delta=None)
    
    with col4:
        st.metric("Avg Bedrooms", f"{df['bedrooms'].mean():.1f}", delta=None)
    
    st.markdown("---")
    
    # Price distribution
    st.subheader("📊 Price Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, 
            x='price', 
            nbins=50,
            title='Price Distribution',
            labels={'price': 'Price ($)'},
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df, 
            y='price',
            title='Price Box Plot',
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: DATA ANALYSIS ====================
elif page == "Data Analysis":
    st.title("📈 Data Analysis")
    
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_price = st.slider(
            "Min Price ($)",
            int(df['price'].min()),
            int(df['price'].max()),
            int(df['price'].min()),
            step=10000
        )
    
    with col2:
        max_price = st.slider(
            "Max Price ($)",
            int(df['price'].min()),
            int(df['price'].max()),
            int(df['price'].max()),
            step=10000
        )
    
    with col3:
        bedrooms = st.multiselect(
            "Bedrooms",
            sorted(df['bedrooms'].unique()),
            default=sorted(df['bedrooms'].unique())
        )
    
    # Apply filters
    filtered_df = df[
        (df['price'] >= min_price) & 
        (df['price'] <= max_price) &
        (df['bedrooms'].isin(bedrooms))
    ]
    
    st.markdown(f"**Showing {len(filtered_df)} properties** (out of {len(df)})")
    
    st.markdown("---")
    
    # Correlation Analysis
    st.subheader("🔗 Correlation with Price")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()['price'].sort_values(ascending=False)
    
    fig = px.bar(
        x=correlation.values, 
        y=correlation.index,
        orientation='h',
        title='Feature Correlation with Price',
        labels={'x': 'Correlation Coefficient'},
        color='x',
        color_continuous_scale='RdBu'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter plots
    st.subheader("📍 Price vs Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            filtered_df,
            x='sqft_living',
            y='price',
            title='Price vs Living Area',
            trendline='ols',
            labels={'sqft_living': 'Sq Ft Living', 'price': 'Price ($)'},
            color='price',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            filtered_df,
            x='grade',
            y='price',
            title='Price vs Grade',
            trendline='ols',
            labels={'grade': 'Grade', 'price': 'Price ($)'},
            color='price',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: FEATURE EXPLORATION ====================
elif page == "Feature Exploration":
    st.title("🔬 Feature Exploration")
    
    st.markdown("---")
    
    # Feature selector
    st.subheader("Select a Feature to Explore")
    
    features = [col for col in df.columns if col not in ['id', 'date', 'lat', 'long']]
    selected_feature = st.selectbox("Choose a feature:", features)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution
        fig = px.histogram(
            df,
            x=selected_feature,
            title=f'Distribution of {selected_feature}',
            nbins=30,
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # vs Price
        if selected_feature != 'price':
            fig = px.scatter(
                df,
                x=selected_feature,
                y='price',
                title=f'{selected_feature} vs Price',
                trendline='ols',
                color='price',
                color_continuous_scale='Viridis'
            )
        else:
            fig = px.box(
                df,
                y='price',
                title=f'{selected_feature} Distribution',
                color_discrete_sequence=['#636EFA']
            )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Statistics
    st.subheader(f"📊 Statistics for {selected_feature}")
    
    stats = {
        'Count': df[selected_feature].count(),
        'Mean': df[selected_feature].mean(),
        'Median': df[selected_feature].median(),
        'Std Dev': df[selected_feature].std(),
        'Min': df[selected_feature].min(),
        'Max': df[selected_feature].max(),
        'Unique Values': df[selected_feature].nunique()
    }
    
    stats_df = pd.DataFrame(stats.items(), columns=['Metric', 'Value'])
    st.dataframe(stats_df, use_container_width=True)

# ==================== PAGE: STATISTICS ====================
elif page == "Statistics":
    st.title("📊 Detailed Statistics")
    
    st.markdown("---")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    summary_stats = df.describe().T
    st.dataframe(summary_stats, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.subheader("🔗 Correlation Matrix")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(width=900, height=900)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Data info
    st.subheader("Data Info")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Rows", len(df))
        st.metric("Total Columns", len(df.columns))
    
    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    
    st.markdown("---")
    
    # Column details
    st.subheader("Column Details")
    
    col_info = []
    for col in df.columns:
        col_info.append({
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Missing': df[col].isnull().sum(),
            'Unique': df[col].nunique()
        })
    
    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Use the sidebar to navigate between different sections of the dashboard.")
