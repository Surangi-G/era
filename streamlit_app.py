import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
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

**Note:** Currently, this app supports imputation only for numerical columns. Categorical data imputation is not yet supported and must be handled externally. 

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
        
        # Detailed dataset information: columns, data types, and count of non-null values
        st.write("**Detailed Dataset Information:**")
        dataset_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Missing Count": df.isnull().sum()
        }).reset_index(drop=True)

        st.write(dataset_info)

        # Data Type Conversion Section
        st.header("Data Type Conversion")

        # Check if 'Site Num' and 'Year' columns are present in the dataset
        columns_to_convert = ['Site Num']
        present_columns = [col for col in columns_to_convert if col in df.columns]

        if present_columns:
            for col in present_columns:
                df[col] = df[col].astype('category')
                st.write(f"Converted {col} to categorical data type.")
    
            # Display updated dataset information after conversion
            st.write("### Updated Dataset Information After Data Type Conversion")
            updated_dataset_info = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": df.dtypes,
                "Non-Null Count": df.notnull().sum(),
                "Missing Count": df.isnull().sum()
            }).reset_index(drop=True)

            st.write(updated_dataset_info)
        else:
            st.warning("Columns 'Site Num' and 'Year' are not present in the dataset. Skipping data type conversion.")
       

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

        # Extract sample count from 'Site No.1'
        if 'Site No.1' in df.columns:
            df['Sample Count'] = df['Site No.1'].str.extract(r'-(\d{2})$').astype(int)
            st.write("Step 3: Sample count extracted from 'Site No.1'.")
            st.write("### Dataset After Sample Count Extraction")
            st.dataframe(df[['Site No.1', 'Sample Count']].head())
        else:
            st.warning("Column 'Site No.1' is missing. Sample count extraction skipped.")

        # Add period labels
        if 'Year' in df.columns:
            conditions = [
                (df['Year'] >= 1995) & (df['Year'] <= 2000),
                (df['Year'] >= 2008) & (df['Year'] <= 2012),
                (df['Year'] >= 2013) & (df['Year'] <= 2017),
                (df['Year'] >= 2018) & (df['Year'] <= 2023)
            ]
            period_labels = ['1995-2000', '2008-2012', '2013-2017', '2018-2023']
            df['Period'] = np.select(conditions, period_labels, default='Unknown')
            st.write("Step 4: Period labels assigned based on 'Year'.")
            st.write("### Dataset After Period Label Assignment")
            st.dataframe(df[['Year', 'Period']].head())
        else:
            st.warning("Column 'Year' is missing. Period assignment skipped.")

        # Keep latest sample count for each site-period
        if 'Site Num' in df.columns and 'Period' in df.columns:
            df = df.loc[df.groupby(['Site Num', 'Period'], observed=True)['Sample Count'].idxmax()].reset_index(drop=True)
            st.write("Step 5: Retained latest sample count for each site-period.")
            st.write("### Dataset After Retaining Latest Samples")
            st.dataframe(df.head())
        else:
            st.warning("Columns 'Site Num' or 'Period' are missing. Filtering latest samples skipped.")

        # Replace '<' values
        columns_with_less_than = [col for col in df.columns if df[col].astype(str).str.contains('<').any()]
        for column in columns_with_less_than:
            df[column] = df[column].apply(lambda x: float(x[1:]) / 2 if isinstance(x, str) and x.startswith('<') else x)
        st.write(f"Step 6: Replaced '<' values in columns: {columns_with_less_than}")
        st.write("### Dataset After Replacing '<' Values")
        st.dataframe(df[columns_with_less_than].head())

         # Separate numerical and categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

        # Imputation using IterativeImputer (only for numerical columns)
        non_predictive_columns = [ 'Year', 'Sample Count']
        df_for_imputation = df.drop(columns=non_predictive_columns, errors='ignore')
        numerical_columns = df_for_imputation.select_dtypes(include=['number']).columns.tolist()

        imputer = IterativeImputer(max_iter=10, random_state=0)
        imputed_data = imputer.fit_transform(df_for_imputation[numerical_columns])
        df_imputed = pd.DataFrame(imputed_data, columns=numerical_columns)

         # Round numerical columns to 2 decimal places
        for col in numerical_columns:
            df_imputed[col] = df_imputed[col].round(2)

        
         # Reattach categorical columns
        df_final = pd.concat([df[categorical_columns + non_predictive_columns].reset_index(drop=True), df_imputed], axis=1)

        # Reorder columns
        columns_order = ['Site No.1', 'Site Num', 'Sample Count', 'Year', 'Period']
        remaining_columns = [col for col in df_final.columns if col not in columns_order]
        df_final = df_final[columns_order + remaining_columns]
        
        st.write("Step 7: Missing values imputed using Iterative Imputer.")
        st.write("### Dataset After Imputation")
        st.dataframe(df_final.head())

        # Visualize before and after imputation
        st.header("Column Distribution Before and After Imputation")
        columns_imputed = [col for col in numerical_columns if col in df.columns and col in df_final.columns]
        for column in columns_imputed:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[column], color='red', label='Before Imputation', kde=True, bins=30, alpha=0.6, ax=ax)
            sns.histplot(df_final[column], color='green', label='After Imputation', kde=True, bins=30, alpha=0.6, ax=ax)
            plt.title(f"Distribution Comparison: {column}")
            plt.legend()
            st.pyplot(fig)

        # Kolmogorov-Smirnov Test
        st.header("Kolmogorov-Smirnov Test Results")
        ks_results = {}
        for column in columns_imputed:
            before = df[column].dropna()
            after = df_final[column].dropna()
            ks_stat, p_value = ks_2samp(before, after)
            ks_results[column] = {'KS Statistic': ks_stat, 'p-value': p_value}
        ks_results_df = pd.DataFrame(ks_results).T
        st.write(ks_results_df)

        # Contamination Index
        native_means = {
            "As": 6.2, "Cd": 0.375, "Cr": 28.5, "Cu": 23.0, "Ni": 17.95, "Pb": 33.0, "Zn": 94.5
        }

        for element, mean_value in native_means.items():
            df_final[f"CI_{element}"] = (df_final[element] / mean_value).round(2)

        ci_columns = [f"CI_{element}" for element in native_means.keys()]
        df_final["ICI"] = df_final[ci_columns].mean(axis=1).round(2)

        def classify_ici(ici):
            if ici <= 1:
                return "Low"
            elif 1 < ici <= 3:
                return "Moderate"
            else:
                return "High"

        df_final["ICI_Class"] = df_final["ICI"].apply(classify_ici)

        st.write("### Final Dataset with Contamination Index")
        st.dataframe(df_final)

        # Final validation
        final_missing = df_final.isnull().sum().sum()
        final_duplicates = df_final.duplicated().sum()
        st.write("### Final Dataset Validation")
        st.write(f"No. of missing values: {final_missing}")
        st.write(f"No. of duplicate rows: {final_duplicates}")

        if final_missing == 0 and final_duplicates == 0:
            st.success("Cleaned dataset is ready! No missing values or duplicates remain.")

                # File Download
        st.header("Download Cleaned Dataset")
        st.write("Your data is now cleaned and ready for analysis. Click the button below to download the cleaned dataset.")
        from io import StringIO
        buffer = StringIO()
        df_final.to_csv(buffer, index=False)
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
