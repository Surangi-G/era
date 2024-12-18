import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Center-Aligned Logo and Title
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths for centering
with col2:
    st.image("aucklandcouncil_logo.PNG", width=150)
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #2c7fb8; font-size: 36px;">ECO SOIL INSIGHTS</h1>
            <h3 style="font-size: 24px; margin-top: -10px;">Data Cleansing App</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

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
""")

# File Upload Section
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("### Original Dataset")
        st.dataframe(df)

        # Step 1: Handle missing critical columns
        critical_columns = ['pH', 'TC %', 'TN %', 'Olsen P', 'AMN', 'BD']
        if missing := [col for col in critical_columns if col not in df.columns]:
            st.error(f"The following critical columns are missing: {missing}")
            st.stop()

        # Remove rows with missing critical column values
        df = df.dropna(subset=critical_columns, how='any')

        # Step 2: Handle duplicates
        if duplicates := df.duplicated().sum():
            df = df.drop_duplicates()

        # Step 3: Sample count extraction
        if 'Site No.1' in df.columns:
            df['Sample Count'] = df['Site No.1'].str.extract(r'-(\d{2})$').astype(int)

        # Step 4: Period labeling
        if 'Year' in df.columns:
            conditions = [
                (df['Year'] >= 1995) & (df['Year'] <= 2000),
                (df['Year'] >= 2008) & (df['Year'] <= 2012),
                (df['Year'] >= 2013) & (df['Year'] <= 2017),
                (df['Year'] >= 2018) & (df['Year'] <= 2023)
            ]
            labels = ['1995-2000', '2008-2012', '2013-2017', '2018-2023']
            df['Period'] = np.select(conditions, labels, default='Unknown')

        # Step 5: Replace '<' values
        columns_with_less_than = [col for col in df.columns if df[col].astype(str).str.contains('<').any()]
        for col in columns_with_less_than:
            df[col] = df[col].apply(lambda x: float(x[1:]) / 2 if isinstance(x, str) and x.startswith('<') else x)

        # Separate numerical and categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

        # Step 6: Imputation
        non_predictive_columns = ['Site Num', 'Year', 'Sample Count']
        numerical_columns = [col for col in numerical_columns if col not in non_predictive_columns]
        imputer = IterativeImputer(max_iter=10, random_state=0)
        imputed_data = imputer.fit_transform(df[numerical_columns])
        df_imputed = pd.DataFrame(imputed_data, columns=numerical_columns)

        # Reattach categorical columns
        df_final = pd.concat([df[non_predictive_columns + categorical_columns].reset_index(drop=True), df_imputed], axis=1)

        # Round numerical columns
        for col in numerical_columns:
            df_final[col] = df_final[col].round(2)

        st.write("### Dataset After Imputation")
        st.dataframe(df_final.head())

        # Step 7: Visualization before and after imputation
        st.header("Column Distribution Before and After Imputation")
        for col in numerical_columns:
            if col in df and col in df_final:
                fig, ax = plt.subplots()
                sns.histplot(df[col], label='Before Imputation', kde=True, color='red', alpha=0.6, ax=ax)
                sns.histplot(df_final[col], label='After Imputation', kde=True, color='green', alpha=0.6, ax=ax)
                ax.legend()
                st.pyplot(fig)

        # Step 8: Kolmogorov-Smirnov Test
        st.header("Kolmogorov-Smirnov Test Results")
        ks_results = {col: ks_2samp(df[col].dropna(), df_final[col]) for col in numerical_columns}
        ks_df = pd.DataFrame({k: {'KS Statistic': v[0], 'p-value': v[1]} for k, v in ks_results.items()}).T
        st.write(ks_df)

        # Step 9: Contamination Index
        native_means = {
            "As": 6.2, "Cd": 0.375, "Cr": 28.5, "Cu": 23.0, "Ni": 17.95, "Pb": 33.0, "Zn": 94.5
        }
        for element, mean_value in native_means.items():
            if element in df_final.columns:
                df_final[f"CI_{element}"] = (df_final[element] / mean_value).round(2)
        ci_columns = [f"CI_{element}" for element in native_means if f"CI_{element}" in df_final.columns]
        df_final["ICI"] = df_final[ci_columns].mean(axis=1).round(2)
        df_final["ICI_Class"] = df_final["ICI"].apply(lambda x: "Low" if x <= 1 else "Moderate" if x <= 3 else "High")

        st.write("### Final Dataset with Contamination Index")
        st.dataframe(df_final)

        # Step 10: File Download
        buffer = BytesIO()
        df_final.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        st.download_button("Download Cleaned Dataset", buffer, "cleaned_dataset.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"An error occurred: {e}")



