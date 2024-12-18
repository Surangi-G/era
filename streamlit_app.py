import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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

The app uses the Iterative Imputer method, a machine learning technique that predicts and fills missing values by modeling each numerical column as a function of others. 

**Note:** Categorical data will be preserved and reattached to the final dataset. Imputation is performed only on numerical columns.

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

        # Display basic dataset information
        st.header("Dataset Information")
        st.write("**Shape of the dataset:**", df.shape)
        st.write("**Missing Values in Each Column:**")
        st.write(df.isnull().sum())

        # Validation: Check for critical columns
        critical_columns = ['pH', 'TC %', 'TN %', 'Olsen P', 'AMN', 'BD']
        st.write("### Critical Columns")
        st.write("These are the critical columns required for soil quality analysis:")
        st.write(critical_columns)

        missing_critical = [col for col in critical_columns if col not in df.columns]
        if missing_critical:
            st.error(f"The following critical columns are missing: {missing_critical}")
            st.stop()

        # Highlight missing values in critical columns
        critical_missing = df[critical_columns].isnull().sum()
        if critical_missing.sum() > 0:
            st.warning("Missing values detected in critical columns. Rows with missing values in critical columns will be removed.")
            st.write(critical_missing)

        # Drop rows with missing critical values
        rows_before = len(df)
        df = df.dropna(subset=critical_columns, how='any')
        rows_after = len(df)
        st.write(f"Step 1: Rows removed due to missing critical values: {rows_before - rows_after}")

        # Display updated dataset after removing rows with missing critical values
        st.write("### Dataset After Removing Missing Critical Values")
        st.dataframe(df.head())

        # Check for duplicates
        duplicates = df.duplicated().sum()
        st.write(f"Step 2: Number of duplicate rows identified: {duplicates}")
        if duplicates > 0:
            st.write(f"Percentage of duplicate rows: {duplicates / len(df) * 100:.2f}%")
            if st.button("Remove Duplicates"):
                df = df.drop_duplicates()
                st.write("All duplicate rows have been removed!")
                st.write("### Dataset After Removing Duplicates")
                st.dataframe(df.head())

        # Separate numerical and categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

        # Perform imputation on numerical columns
        imputer = IterativeImputer(max_iter=10, random_state=0)
        imputed_data = imputer.fit_transform(df[numerical_columns])
        df_imputed = pd.DataFrame(imputed_data, columns=numerical_columns)

        # Reattach categorical columns
        df_final = pd.concat([df[categorical_columns].reset_index(drop=True), df_imputed], axis=1)

        # Round numerical columns to 2 decimal places
        for col in numerical_columns:
            df_final[col] = df_final[col].round(2)

        st.write("### Final Dataset After Imputation and Reattachment")
        st.dataframe(df_final.head())

        # File Download
        st.header("Download Cleaned Dataset")
        st.write("Your data is now cleaned and ready for analysis. Click the button below to download the cleaned dataset.")
        from io import BytesIO
        buffer = BytesIO()
        df_final.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)

        st.download_button(
            label="Download as Excel",
            data=buffer,
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"An error occurred during the data cleaning process: {e}")
        st.write("Please check your dataset for inconsistencies or missing required columns and try again.")

