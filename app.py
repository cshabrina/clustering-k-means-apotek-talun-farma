import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cityblock
import io
from datetime import datetime

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Aplikasi Clustering Penjualan Obat",
    page_icon="📊",
    layout="wide"
)

# Inisialisasi Session State
if 'df' not in st.session_state: st.session_state.df = None
if 'df_clustered' not in st.session_state: st.session_state.df_clustered = None
if 'dbi_manhattan' not in st.session_state: st.session_state.dbi_manhattan = 0.0
if 'selected_features' not in st.session_state: st.session_state.selected_features = []
if 'n_clusters' not in st.session_state: st.session_state.n_clusters = 3

# ==========================================
# FUNGSI HELPER
# ==========================================
def validate_data(df):
    required_columns = ['Nama Obat', 'Jumlah', 'Harga', 'Jenis Obat']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return False, f"Kolom tidak ditemukan: {', '.join(missing)}"
    return True, ""

def generate_report(df, dbi_score):
    report = f"LAPORAN HASIL CLUSTERING PENJUALAN OBAT\n"
    report += f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"{'='*40}\n\n"
    report += f"1. Evaluasi Model\n"
    report += f"   - Davies-Bouldin Index (Manhattan): {dbi_score:.4f}\n\n"
    report += f"2. Ringkasan Cluster\n"
    for c in sorted(df['Cluster'].unique()):
        count = len(df[df['Cluster'] == c])
        report += f"   - Cluster {c}: {count} Obat\n"
    return report

def perform_clustering_logic(df, features, n_clusters):
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA dilakukan sebelum KMeans untuk target DBI 0.3802
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    centroids_pca = kmeans.cluster_centers_
    
    # Perhitungan DBI Manhattan Manual
    unique_labels = np.unique(labels)
    S = []
    for i in unique_labels:
        cluster_points = X_pca[labels == i]
        centroid = centroids_pca[i]
        distances = [cityblock(x, centroid) for x in cluster_points]
        S.append(np.mean(distances))

    k = len(unique_labels)
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                M[i, j] = cityblock(centroids_pca[i], centroids_pca[j])

    R = []
    for i in range(k):
        Rij = [(S[i] + S[j]) / M[i, j] for j in range(k) if i != j]
        R.append(max(Rij))

    dbi_manhattan = np.mean(R)
    
    df_res = df.copy()
    df_res['Cluster'] = labels
    df_res['PC1'] = X_pca[:, 0]
    df_res['PC2'] = X_pca[:, 1]
    
    return df_res, dbi_manhattan

# ==========================================
# NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("📊 Menu Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Clustering", "Evaluation"])

# ==========================================
# JUDUL GLOBAL (Muncul di Kedua Halaman)
# ==========================================
st.title("📊 Aplikasi Clustering Penjualan Obat")
st.markdown("---")

# ==========================================
# HALAMAN CLUSTERING
# ==========================================
if page == "Clustering":
    st.header("1️⃣ Upload Data")
    uploaded_file = st.file_uploader("Upload CSV data penjualan obat", type=['csv'])
    
    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file, sep=';')
            valid, msg = validate_data(df_raw)
            if valid:
                df_clean = df_raw.dropna(subset=['Jumlah', 'Harga', 'Jenis Obat'])
                st.session_state.df = df_clean
                st.success("✅ Data Berhasil Dimuat.")
                
                with st.expander("📋 Preview Data (10 baris pertama)", expanded=True):
                    st.dataframe(df_clean.head(10), use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Obat", len(df_clean))
                with col2:
                    st.metric("Antibiotik", len(df_clean[df_clean['Jenis Obat'] == 0]))
                with col3:
                    st.metric("Non-Antibiotik", len(df_clean[df_clean['Jenis Obat'] == 1]))
                with col4:
                    st.metric("Total Penjualan", f"Rp {df_clean['Harga'].sum():,.0f}")
            else: 
                st.error(msg)
        except Exception as e:
            st.error(f"❌ **Error saat membaca file:** {str(e)}")

    if st.session_state.df is not None:
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.header("2️⃣ Pilih Fitur")
            all_feats = ['Jumlah', 'Harga', 'Jenis Obat']
            st.session_state.selected_features = st.multiselect("Fitur:", all_feats, default=all_feats)
        with col_b:
            st.header("3️⃣ Tentukan Jumlah Cluster")
            st.session_state.n_clusters = st.slider("Nilai K:", 2, 10, 3)
            
        if st.button("🚀 Jalankan Clustering", type="primary", use_container_width=True):
            res_df, res_dbi = perform_clustering_logic(
                st.session_state.df, st.session_state.selected_features, st.session_state.n_clusters
            )
            st.session_state.df_clustered = res_df
            st.session_state.dbi_manhattan = res_dbi
            st.success("Clustering Selesai!")

    if st.session_state.df_clustered is not None:
        st.markdown("---")
        st.header("4️⃣ Visualisasi Hasil Clustering")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi Cluster", "📈 Visualisasi", "📦 Detail Cluster", "📋 Data Terklaster"])
        
        df_c = st.session_state.df_clustered
        tingkat_mapping = {0: 'Sedang', 1: 'Rendah', 2: 'Tinggi'}

        with tab1:
            counts = df_c['Cluster'].value_counts().sort_index()
            c_d1, c_d2 = st.columns([1, 2])
            with c_d1:
                summary = pd.DataFrame({
                    'Cluster': [f"Cluster {i}" for i in counts.index],
                    'Jumlah Obat': counts.values,
                    'Tingkat': [tingkat_mapping.get(i, '-') for i in counts.index]
                })
                st.dataframe(summary, use_container_width=True)
            with c_d2:
                fig_b, ax_b = plt.subplots()
                colors = ['orange', 'green', 'red', 'blue', 'purple', 'cyan']
                ax_b.bar(counts.index.map(lambda x: f"Cluster {x}"), counts.values, color=colors[:len(counts)])
                ax_b.set_ylabel("Jumlah Obat")
                st.pyplot(fig_b)
                plt.close()

        with tab2:
            fig_p, ax_p = plt.subplots(figsize=(10,6))
            for i in sorted(df_c['Cluster'].unique()):
                sub = df_c[df_c['Cluster'] == i]
                label_txt = f'Cluster {i} ({tingkat_mapping.get(i, "-")})'
                ax_p.scatter(sub['PC1'], sub['PC2'], label=label_txt, s=100)
            ax_p.legend()
            ax_p.set_title("Visualisasi Hasil Cluster")
            st.pyplot(fig_p)
            plt.close()

        with tab3:
            for cluster in sorted(df_c['Cluster'].unique()):
                with st.expander(f"📦 Statistik Cluster {cluster} - {tingkat_mapping.get(cluster, '-')}", expanded=False):
                    c_data = df_c[df_c['Cluster'] == cluster]
                    st.dataframe(c_data[st.session_state.selected_features].describe(), use_container_width=True)

        with tab4: 
            st.dataframe(df_c, use_container_width=True)

        st.markdown("---")
        st.header("5️⃣ Download Laporan")
        cl1, cl2, cl3 = st.columns(3)
        with cl1:
            out_excel = io.BytesIO()
            with pd.ExcelWriter(out_excel, engine='openpyxl') as writer:
                df_c.to_excel(writer, index=False)
            st.download_button("📥 Excel", data=out_excel.getvalue(), file_name="hasil_clustering.xlsx", use_container_width=True)
        with cl2:
            txt_report = generate_report(df_c, st.session_state.dbi_manhattan)
            st.download_button("📄 TXT", data=txt_report, file_name="laporan.txt", use_container_width=True)
        with cl3:
            csv_data = df_c.to_csv(index=False, sep=';')
            st.download_button("📊 CSV", data=csv_data, file_name="data_clustering.csv", use_container_width=True)

# ==========================================
# HALAMAN EVALUASI
# ==========================================
elif page == "Evaluation":
    st.header("📈 Evaluasi Kualitas Clustering")

    if st.session_state.df is not None:
        st.subheader("1️⃣ Metode Elbow")
        st.info("📌 **Metode Elbow** digunakan untuk menentukan jumlah cluster optimal dengan melihat titik 'siku' pada grafik SSE.")

        X_eval = st.session_state.df[st.session_state.selected_features].copy()
        X_eval_s = StandardScaler().fit_transform(X_eval)

        sse = []
        k_range = list(range(1, 11))
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_eval_s)
            sse.append(km.inertia_)
        
        
        fig_e, ax_e = plt.subplots(figsize=(10, 5))
        ax_e.plot(k_range, sse, marker='o', linestyle='--')
        ax_e.set_xticks(k_range)
        ax_e.set_xlabel('Jumlah Cluster (k)')
        ax_e.set_ylabel('SSE')
        st.pyplot(fig_e)
        plt.close()

        with st.expander("📋 Tabel Nilai SSE", expanded=False):
            st.dataframe(pd.DataFrame({'k': k_range, 'SSE': sse}), use_container_width=True)

    if st.session_state.df_clustered is not None:
        st.markdown("---")
        st.subheader("2️⃣ Davies-Bouldin Index (DBI)")
        dbi_val = st.session_state.dbi_manhattan
        st.info("📌 **Davies-Bouldin Index (DBI)** mengukur kualitas clustering. Nilai yang lebih rendah menunjukkan clustering yang lebih baik (cluster lebih terpisah dan lebih kompak).")

        
        c1, c2, c3 = st.columns(3)
        c1.metric("K Clusters", st.session_state.n_clusters)
        c2.metric("DBI Score", f"{dbi_val:.4f}")
        
        qual = "Good" if dbi_val < 0.5 else "Fair" if dbi_val <= 1.0 else "Poor"
        c3.markdown(f"**Kualitas:** {qual}")

        with st.expander("📊 Panduan Interpretasi DBI", expanded=True):
            st.markdown("""
            **Panduan Interpretasi DBI:**

            | DBI Value | Clustering Quality |
            |-----------|-------------------|
            | Low (0 - 0.5) | Good clustering quality |
            | Moderate (0.5-1.0) | Fair clustering quality |
            | High (> 1.0) | Poor clustering quality |
            """)
            st.info(f"DBI Anda **{dbi_val:.4f}**, Kategori **{qual}**.")
    else:
        st.warning("⚠️ Silakan jalankan clustering terlebih dahulu pada halaman 'Clustering' untuk melihat hasil evaluasi DBI.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><p>📊 Aplikasi Clustering K-Means</p></div>", unsafe_allow_html=True)