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

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Business Understanding", "Data Overview", "Exploratory Data Analysis"])

if menu == "Business Understanding":
    st.title("Saudi Arabia Used Car Price Prediction")
    
    st.header("Business Problem")
    st.info("""
    **Context:** Pasar mobil bekas di Arab Saudi menghadapi tantangan dalam penentuan harga (pricing).
    Penjual sering kesulitan menentukan harga yang kompetitif, dan pembeli takut overpaying.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Problem Statement")
        st.write("""
        - Asimetri informasi harga.
        - Banyak listing 'Negotiable' (Harga 0) yang membingungkan.
        - Risiko kerugian bagi penjual jika harga terlalu rendah, atau tidak laku jika terlalu tinggi.
        """)
    with col2:
        st.subheader("Goals")
        st.write("""
        - **Bagi Penjual:** Rekomendasi harga ideal & cepat terjual.
        - **Bagi Pembeli:** Transparansi harga wajar (Fair Value).
        - **Output:** Machine Learning Model untuk prediksi harga.
        """)

elif menu == "Data Overview":
    st.title("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data Asli", df_raw.shape[0])
    col2.metric("Data Bersih (Siap Analisis)", df_eda.shape[0])
    col3.metric("Data Dibuang (Harga 0 / Duplikat)", df_raw.shape[0] - df_eda.shape[0])

    st.subheader("Cuplikan Dataset")
    st.dataframe(df_eda.head())

    st.subheader("Statistik Deskriptif")
    st.write(df_eda.describe())
    
    st.subheader("Cek Tipe Data")
    buffer = pd.DataFrame(df_eda.dtypes, columns=['Data Type']).astype(str)
    st.table(buffer)

elif menu == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Data")
    
    year_min = int(df_eda['Year'].min())
    year_max = int(df_eda['Year'].max())
    selected_year = st.sidebar.slider("Pilih Rentang Tahun:", year_min, year_max, (2010, year_max))
    
    all_makes = ['All'] + sorted(df_eda['Make'].unique().tolist())
    selected_make = st.sidebar.selectbox("Pilih Merek Mobil:", all_makes)

    df_filtered = df_eda[(df_eda['Year'] >= selected_year[0]) & (df_eda['Year'] <= selected_year[1])]
    if selected_make != 'All':
        df_filtered = df_filtered[df_filtered['Make'] == selected_make]

    st.caption(f"Menampilkan data untuk Tahun: {selected_year} | Merek: {selected_make} | Total Data: {df_filtered.shape[0]}")

    if df_filtered.empty:
        st.warning("Data tidak ditemukan dengan filter tersebut.")
    else:
        tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

        with tab1:
            st.subheader("Analisis Target: Price Distribution")
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            sns.histplot(df_filtered['Price'], kde=True, color='blue', ax=ax[0])
            ax[0].set_title(f'Distribusi Harga ({selected_make})')
            
            sns.boxplot(x=df_filtered['Price'], color='green', ax=ax[1])
            ax[1].set_title(f'Boxplot Harga ({selected_make})')
            
            st.pyplot(fig)
            
            st.subheader("Analisis Kategori: Merek & Gear Type")
            col_u1, col_u2 = st.columns(2)
            
            with col_u1:
                if selected_make == 'All':
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    top_makes = df_filtered['Make'].value_counts().head(10)
                    sns.barplot(x=top_makes.values, y=top_makes.index, palette='viridis', ax=ax2)
                    ax2.set_title('Top 10 Merek Terbanyak')
                    st.pyplot(fig2)
                else:
                    st.info("Anda memilih satu merek spesifik, grafik Top 10 dinonaktifkan.")

            with col_u2:
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.countplot(x='Gear_Type', data=df_filtered, palette='pastel', ax=ax3)
                ax3.set_title('Jumlah Mobil berdasarkan Tipe Gear')
                st.pyplot(fig3)

        with tab2:
            st.subheader("Hubungan Fitur Numerik vs Harga")
            
            scatter_option = st.selectbox("Pilih Variabel X:", ['Mileage', 'Year', 'Engine_Size'])
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=scatter_option, y='Price', data=df_filtered, alpha=0.6, color='orange', ax=ax4)
            ax4.set_title(f'Hubungan {scatter_option} vs Price')
            st.pyplot(fig4)
            
            st.subheader("Hubungan Kategori vs Harga")
            cat_option = st.selectbox("Pilih Kategori:", ['Gear_Type', 'Options'])
            
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Price', y=cat_option, data=df_filtered, ax=ax5)
            ax5.set_title(f'Harga berdasarkan {cat_option}')
            st.pyplot(fig5)

        with tab3:
            st.subheader("Korelasi Antar Fitur Numerik")
            
            numeric_cols = ['Price', 'Year', 'Mileage', 'Engine_Size']
            corr = df_filtered[numeric_cols].corr()
            
            fig6, ax6 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax6)
            st.pyplot(fig6)

st.markdown("---")
st.markdown("Created by Haqi Dhiya' Firmana - Mini Project Scripting Language")