import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Saudi Used Car Dashboard",
    layout="wide"
)

# load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/data_saudi_used_cars.csv")
    return df

# data cleaning
def clean_data(df):
    df_clean = df[df['Price'] > 0].copy()
    df_clean.drop_duplicates(inplace=True)
    return df_clean

try:
    df_raw = load_data()
    df_eda = clean_data(df_raw)
except Exception as e:
    st.error(f"Gagal memuat data. Pastikan internet aktif. Error: {e}")
    st.stop()

st.sidebar.title("Menu")
menu = st.sidebar.radio("Select Menu :", ["Main", "Data Overview", "Exploratory Data Analysis", "Recommendations"])

if menu == "Main":
    st.title("Saudi Arabia Used Car Analytics")
    
    st.header("Context & Data Description")
    st.info("""
    This dashboard analyzes the used car market transactions in Saudi Arabia (sourced from Syarah.com). 
    The dataset captures various vehicle specifications such as Year, Mileage, Make, and Engine Size.
    """)

    st.info("""
    **Primary Goal:** To uncover market price patterns and solve the issue of price uncertainty (e.g., 'Negotiable' or hidden prices) 
    that often confuses both buyers and sellers.
    """)

    st.markdown("### Objectives")
    st.success("1. Analyze Price Distribution")
    st.success("2. Identify Key Price Factors")
    st.success("3. Provide Transparent Insights")

elif menu == "Data Overview":
    st.title("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Raw Data", df_raw.shape[0])
    col2.metric("Cleaned Data (Usable)", df_eda.shape[0])
    col3.metric("Removed Data (Price=0 / Duplicates)", df_raw.shape[0] - df_eda.shape[0])

    st.subheader("Dataset Preview")
    st.dataframe(df_eda.head())

    st.subheader("Descriptive Statistics")
    st.write(df_eda.describe())
    
    st.subheader("Data Types")
    buffer = pd.DataFrame(df_eda.dtypes, columns=['Data Type']).astype(str)
    st.table(buffer)

elif menu == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Data")
    
    year_min = int(df_eda['Year'].min())
    year_max = int(df_eda['Year'].max())
    selected_year = st.sidebar.slider("Select Car Make: ", year_min, year_max, (2010, year_max))
    
    all_makes = ['All'] + sorted(df_eda['Make'].unique().tolist())
    selected_make = st.sidebar.selectbox("Select Car Make:", all_makes)

    df_filtered = df_eda[(df_eda['Year'] >= selected_year[0]) & (df_eda['Year'] <= selected_year[1])]
    if selected_make != 'All':
        df_filtered = df_filtered[df_filtered['Make'] == selected_make]

    st.caption(f"Showing data for:{selected_year} | Merek: {selected_make} | Total Data: {df_filtered.shape[0]}")

    if df_filtered.empty:
        st.warning("No data found with the selected filters.")
    else:
        tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

        with tab1:
            st.subheader("Target Analysis: Price Distribution")
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            sns.histplot(df_filtered['Price'], kde=True, color='blue', ax=ax[0])
            ax[0].set_title(f'Price Distribution ({selected_make})')
            
            sns.boxplot(x=df_filtered['Price'], color='green', ax=ax[1])
            ax[1].set_title(f'Price Boxplot & Outliers ({selected_make})')
            
            st.pyplot(fig)
            
            st.subheader("Categorical Analysis: Make & Gear Type")
            col_u1, col_u2 = st.columns(2)
            
            with col_u1:
                if selected_make == 'All':
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    top_makes = df_filtered['Make'].value_counts().head(10)
                    sns.barplot(x=top_makes.values, y=top_makes.index, palette='viridis', ax=ax2)
                    ax2.set_title('Top 10 Most Listed Brands')
                    st.pyplot(fig2)
                else:
                    st.info("You selected a specific brand, Top 10 chart is hidden.")

            with col_u2:
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.countplot(x='Gear_Type', data=df_filtered, palette='pastel', ax=ax3)
                ax3.set_title('Distribution by Gear Type')
                st.pyplot(fig3)

        with tab2:
            st.subheader("Numerical Features vs. Price")
            
            scatter_option = st.selectbox("Select X Variable:", ['Mileage', 'Year', 'Engine_Size'])
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=scatter_option, y='Price', data=df_filtered, alpha=0.6, color='orange', ax=ax4)
            ax4.set_title(f'Relationship: {scatter_option} vs Price')
            st.pyplot(fig4)
            
            st.subheader("Categorical Features vs. Price")
            cat_option = st.selectbox("Select Category:", ['Gear_Type', 'Options'])
            
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Price', y=cat_option, data=df_filtered, ax=ax5)
            ax5.set_title(f'Price Distribution by{cat_option}')
            st.pyplot(fig5)

        with tab3:
            st.subheader("Correlation Heatmap")
            
            numeric_cols = ['Price', 'Year', 'Mileage', 'Engine_Size']
            corr = df_filtered[numeric_cols].corr()
            
            fig6, ax6 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax6)
            st.pyplot(fig6)

elif menu == "Recommendations":
    st.header("Insights & Recommendations")
    st.write("Based on the Exploratory Data Analysis, here are the key takeaways and future steps.")

    col_rec1, col_rec2 = st.columns(2)

    with col_rec1:
        st.success("**Strategic Business Insights**")
        st.write("""
        1. **Mileage Matters:** Cars with lower mileage command significantly higher prices. Sellers should highlight low mileage as a key selling point.
        2. **Market Preference:** 'Standard' options are the most common, but 'Full Option' cars have a wider price variance.
        3. **Age Factor:** Depreciation is evident. Cars older than 5 years see a sharper price decline.
        """)

    with col_rec2:
        st.info("**Future Technical Development**")
        st.write("""
        1. **Feature Engineering:**
           - Create a 'Car_Age' feature (Current Year - Year).
           - Group 'Make' into 'Luxury' vs 'Economy' segments.
        2. **Machine Learning Modelling:**
           - Develop a Price Prediction Model using **Random Forest** or **XGBoost**.
           - Evaluation Metric: Use **MAPE** to understand error in percentage.
        """)

st.markdown("---")
st.markdown("Haqi Dhiya' Firmana - Mini Project Scripting Language")