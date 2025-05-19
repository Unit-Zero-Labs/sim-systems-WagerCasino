"""
Styles module for the Unit Zero Labs Tokenomics Engine.
Contains CSS styles and theming for the Streamlit application.
"""

import streamlit as st


def apply_custom_css():
    """
    Apply custom CSS styles for the application.
    """
    # Use the new CSS styles
    load_css()


def set_page_config():
    """
    Set Streamlit page configuration.
    """
    st.set_page_config(
        page_title="UZL Tokenomics Engine", 
        layout="wide"
    )


def load_css():
    """
    Load CSS styles for the application.
    """
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    /* Header Styles */
    .header-title {
        color: #FFFFFF;
        margin-top: 0;
    }
    
    /* Plot Container Styles */
    .plot-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Adjust plot background and grid colors for better contrast and readability */
    .js-plotly-plot .plotly .bg {
        fill: rgba(30, 30, 30, 0.8) !important;
    }
    
    .js-plotly-plot .plotly .gridlayer path {
        stroke: rgba(255, 255, 255, 0.15) !important;
    }
    
    .js-plotly-plot .plotly .zerolinelayer path {
        stroke: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Data Table Styles */
    .dataframe {
        color: #FFFFFF !important;
        background-color: #2D2D2D !important;
    }
    
    .dataframe th {
        background-color: #3D3D3D !important;
        color: white !important;
        font-weight: bold !important;
        border: 1px solid #4D4D4D !important;
    }
    
    .dataframe td {
        background-color: #2D2D2D !important;
        color: white !important;
        border: 1px solid #3D3D3D !important;
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 4px 4px 0px 0px;
        border: none;
        color: white;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-bottom: 2px solid #4D89FB;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background-color: #2D2D2D;
    }
    
    /* Plotly Chart Color Scheme */
    :root {
        --color-0: #4285F4;  /* Google Blue */
        --color-1: #EA4335;  /* Google Red */
        --color-2: #FBBC05;  /* Google Yellow */
        --color-3: #34A853;  /* Google Green */
        --color-4: #8AB4F8;  /* Light Blue */
        --color-5: #F6AEA9;  /* Light Red */
        --color-6: #FDE293;  /* Light Yellow */
        --color-7: #A8DAB5;  /* Light Green */
    }
    
    /* Custom streamlit elements */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 5px;
    }
    
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 5px;
        padding: 10px;
    }
    
    .stButton > button {
        background-color: #4D89FB;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    
    .stButton > button:hover {
        background-color: #2A6AF5;
    }
    </style>
    """, unsafe_allow_html=True)