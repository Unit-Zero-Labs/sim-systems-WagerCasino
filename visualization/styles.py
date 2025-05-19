"""
Styles module for the Unit Zero Labs Tokenomics Engine.
Contains CSS styles and theming for the Streamlit application.
"""

import streamlit as st


def apply_custom_css():
    """
    Apply custom CSS for dark purple gradient background and Inconsolata font.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inconsolata:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #2A0845 0%, #6441A5 100%);
        color: white;
        font-family: 'Inconsolata', monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        font-family: 'Inconsolata', monospace;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 4px 4px 0px 0px;
        color: white;
        padding: 10px 20px;
        font-family: 'Inconsolata', monospace;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
    }
    .plot-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    h1, h2, h3, .stMarkdown, p, div {
        color: white;
        font-family: 'Inconsolata', monospace;
    }
    .header-title {
        margin-top: 0;
        margin-bottom: 0;
        font-size: 1.8rem;
        display: inline-block;
        vertical-align: middle;
        margin-left: 10px;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 20px;
    }
    .logo-title-container {
        display: flex;
        align-items: center;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        padding: 8px 16px;
        font-family: 'Inconsolata', monospace;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background-color: rgba(100, 65, 165, 0.3);
    }
    .stSlider > div > div > div > div {
        background-color: white;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px dashed rgba(255, 255, 255, 0.3);
        padding: 20px;
        border-radius: 5px;
    }
    .stFileUploader > div:hover {
        background-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Error/info/success message styling */
    .stAlert > div {
        padding: 12px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


def set_page_config():
    """
    Set Streamlit page configuration.
    """
    st.set_page_config(
        page_title="UZL Tokenomics Engine", 
        layout="wide"
    )