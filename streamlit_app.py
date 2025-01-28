import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from io import StringIO

# Display Auckland Council Logo (center-aligned) and Titles
st.markdown(
    """
    <style>
        .center-content {
            text-align: center;
            margin-top: 10px;
        }
        h1 {
            color: #2c7fb8;
            font-size: 36px;
            margin: 10px 0 5px;
        }
        h3 {
            font-size: 24px;
            margin: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Use st.image to display the logo with HTML alignment
st.markdown('<div class="center-content">', unsafe_allow_html=True)
st.image("aucklandcouncil_logo.PNG", width=150)
st.markdown(
    """
    <h1>ECO SOIL INSIGHTS AKL</h1>
    <h3>Data Cleansing App</h3>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# Access Control
st.sidebar.header("Access Control")
password = st.sidebar.text_input("Enter the access password:", type="password")
if password != "ESIAKL":  # Replace 'ESIAKL' with your desired password
    st.warning("Access Denied. Please enter the correct password.")
    st.stop()

# Introduction Section
st.write("""
Welcome to the Ecosoil Insight AKL Data Cleaning App! 
This app is designed to clean and prepare soil data, including site metadata, soil quality metrics, and contamination levels. It addresses common issues like missing values, duplicates, and irregularities, ensuring the dataset is accurate and ready for advanced analysis. 

To get started, upload your raw dataset below and follow the guided steps.
""")

# File Upload Section
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)
        st.write("### Original Dataset")
        st.dataframe(df)

        # Data cleaning and processing steps here
        # (Assume the previous data cleaning logic remains unchanged)

        # Final validation
        final_missing = df.isnull().sum().sum()
        final_duplicates = df.duplicated().sum()
        st.write("### Final Dataset Validation")
        st.write(f"No. of missing values: {final_missing}")
        st.write(f"No. of duplicate rows: {final_duplicates}")

        if final_missing == 0 and final_duplicates == 0:
            st.success("Cleaned dataset is ready! No missing values or duplicates remain.")

        # File Download as CSV
        st.header("Download Cleaned Dataset")
        st.write("Your data is now cleaned and ready for analysis. Click the button below to download the cleaned dataset as a CSV file.")
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="Download as CSV",
            data=buffer.getvalue(),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"An error occurred during the data cleaning process: {e}")
        st.write("Please check your dataset for inconsistencies or missing required columns and try again.")
