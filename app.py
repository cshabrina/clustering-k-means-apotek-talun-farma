import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import euclidean
import io
from datetime import datetime

st.set_page_config(
    page_title="Aplikasi Clustering K-Means",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Aplikasi Clustering K-Means untuk Analisis Penjualan Obat")
st.markdown("---")


if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3

def validate_data(df):
    """Validasi struktur dan isi dataset"""
    errors = []
    warnings = []

    required_columns = ['Nama Obat', 'Jumlah', 'Harga', 'Jenis Obat']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        errors.append(f"⚠️ Kolom berikut tidak ditemukan: {', '.join(missing_columns)}")
        return False, errors, warnings

    if df.empty:
        errors.append("⚠️ Dataset kosong. Silakan upload file yang berisi data.")
        return False, errors, warnings

    try:
        df['Jumlah'] = pd.to_numeric(df['Jumlah'], errors='coerce')
        df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')
        df['Jenis Obat'] = pd.to_numeric(df['Jenis Obat'], errors='coerce')
    except:
        errors.append("⚠️ Gagal mengkonversi kolom numerik. Pastikan kolom Jumlah, Harga, dan Jenis Obat berisi angka.")
        return False, errors, warnings

    null_counts = df[['Jumlah', 'Harga', 'Jenis Obat']].isnull().sum()
    if null_counts.sum() > 0:
        warnings.append(f"⚠️ Ditemukan {null_counts.sum()} nilai yang tidak valid dan akan dihapus.")

    if (df['Jumlah'] < 0).any() or (df['Harga'] < 0).any():
        warnings.append("⚠️ Ditemukan nilai negatif pada kolom Jumlah atau Harga.")

    unique_jenis = df['Jenis Obat'].dropna().unique()
    if not all(val in [0, 1, 0.0, 1.0] for val in unique_jenis):
        warnings.append("⚠️ Kolom 'Jenis Obat' seharusnya berisi nilai 0 (Antibiotik) atau 1 (Non-Antibiotik).")

    if len(df) < 3:
        errors.append("⚠️ Data terlalu sedikit. Minimal 3 baris data diperlukan untuk clustering.")
        return False, errors, warnings

    return True, errors, warnings

def create_elbow_plot(X_scaled, max_k=10):
    """Membuat elbow plot untuk menentukan k optimal"""
    sse = []
    K_range = range(2, min(max_k + 1, len(X_scaled)))

    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K_range, sse, marker='o', linestyle='-', linewidth=2, markersize=8)

    for i, txt in enumerate(sse):
        ax.text(K_range[i], sse[i], f"{txt:.2f}", ha='center', va='bottom', fontsize=9, color='red')

    ax.set_xlabel('Jumlah Cluster (k)', fontsize=12)
    ax.set_ylabel('SSE (Sum of Squared Errors)', fontsize=12)
    ax.set_title('Metode Elbow untuk Menentukan K Optimal', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    return fig, sse, list(K_range)

def perform_clustering(df, features, n_clusters):
    """Melakukan clustering K-Means"""
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    centroids = kmeans.cluster_centers_

    for i in range(n_clusters):
        df[f'Jarak_ke_Centroid_{i}'] = [
            euclidean(row, centroids[i]) for row in X_scaled
        ]

    dbi_score = davies_bouldin_score(X_scaled, df['Cluster'])

    return df, centroids, scaler, X_scaled, dbi_score

def create_2d_visualization(df, feature_x, feature_y, centroids, scaler, selected_features):
    """Membuat visualisasi 2D scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['orange', 'green', 'red', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for cluster in sorted(df['Cluster'].unique()):
        data_cluster = df[df['Cluster'] == cluster]
        ax.scatter(
            data_cluster[feature_x],
            data_cluster[feature_y],
            c=colors[cluster % len(colors)],
            label=f'Cluster {cluster}',
            s=100,
            alpha=0.6,
            edgecolors='black'
        )

    idx_x = selected_features.index(feature_x)
    idx_y = selected_features.index(feature_y)

    ax.scatter(
        centroids[:, idx_x],
        centroids[:, idx_y],
        c='black',
        marker='X',
        s=300,
        label='Centroid',
        edgecolors='white',
        linewidths=2
    )

    for i, (x, y) in enumerate(zip(centroids[:, idx_x], centroids[:, idx_y])):
        ax.text(x, y, f'C{i}', color='white', fontsize=10, weight='bold',
                ha='center', va='center')

    ax.set_xlabel(feature_x, fontsize=12)
    ax.set_ylabel(feature_y, fontsize=12)
    ax.set_title(f'Visualisasi Cluster: {feature_x} vs {feature_y}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig

def create_3d_visualization(df, features, centroids):
    """Membuat visualisasi 3D scatter plot"""
    if len(features) < 3:
        return None

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['orange', 'green', 'red', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for cluster in sorted(df['Cluster'].unique()):
        data_cluster = df[df['Cluster'] == cluster]
        ax.scatter(
            data_cluster[features[0]],
            data_cluster[features[1]],
            data_cluster[features[2]],
            c=colors[cluster % len(colors)],
            label=f'Cluster {cluster}',
            s=50,
            alpha=0.6
        )

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c='black',
        marker='X',
        s=200,
        label='Centroid',
        edgecolors='yellow',
        linewidths=2
    )

    for i, (x, y, z) in enumerate(centroids):
        ax.text(x, y, z, f'C{i}', color='black', fontsize=10, weight='bold')

    ax.set_xlabel(features[0], fontsize=12)
    ax.set_ylabel(features[1], fontsize=12)
    ax.set_zlabel(features[2], fontsize=12)
    ax.set_title('Visualisasi 3D Cluster', fontsize=14, fontweight='bold')
    ax.legend()

    return fig

def generate_report(df, centroids, selected_features, n_clusters, dbi_score):
    """Generate laporan dalam format text"""
    report = []
    report.append("=" * 80)
    report.append("LAPORAN HASIL CLUSTERING K-MEANS")
    report.append("=" * 80)
    report.append(f"\nTanggal: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    report.append(f"\nJumlah Data: {len(df)}")
    report.append(f"Jumlah Cluster: {n_clusters}")
    report.append(f"Fitur yang Digunakan: {', '.join(selected_features)}")
    report.append(f"\nDavies-Bouldin Index (DBI): {dbi_score:.4f}")
    report.append("\n" + "=" * 80)

    report.append("\nSTATISTIK PER CLUSTER")
    report.append("=" * 80)

    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        report.append(f"\n--- Cluster {cluster} ({len(cluster_data)} obat) ---")

        for feature in selected_features:
            mean_val = cluster_data[feature].mean()
            report.append(f"  {feature} (rata-rata): {mean_val:.2f}")

        report.append(f"\n  Daftar Obat:")
        for idx, row in cluster_data.iterrows():
            jenis_text = "Antibiotik" if row['Jenis Obat'] == 0 else "Non-Antibiotik"
            report.append(f"    - {row['Nama Obat']} (Jenis: {jenis_text})")

    report.append("\n" + "=" * 80)
    report.append("\nCENTROID (Nilai Standar)")
    report.append("=" * 80)

    for i, centroid in enumerate(centroids):
        report.append(f"\nCluster {i}:")
        for j, feature in enumerate(selected_features):
            report.append(f"  {feature}: {centroid[j]:.4f}")

    report.append("\n" + "=" * 80)
    report.append("\nKeterangan:")
    report.append("  - Jenis Obat: 0 = Antibiotik, 1 = Non-Antibiotik")
    report.append("  - Nilai centroid dalam bentuk standar (setelah normalisasi)")
    report.append("  - DBI Score: Semakin rendah, semakin baik clustering")
    report.append("=" * 80)

    return "\n".join(report)

# ========== BAGIAN 1: UPLOAD DATA ==========
st.header("1️⃣ Upload Data")

uploaded_file = st.file_uploader(
    "Upload file CSV dataset penjualan obat",
    type=['csv'],
    help="Format file: CSV dengan separator titik koma (;) dan kolom: Nama Obat, Jumlah, Harga, Jenis Obat"
)

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        except:
            try:
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
            except:
                df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')

        is_valid, errors, warnings = validate_data(df)

        if errors:
            st.error("❌ **Validasi Data Gagal:**")
            for error in errors:
                st.error(error)
        else:
            df_clean = df.dropna(subset=['Jumlah', 'Harga', 'Jenis Obat'])

            if warnings:
                with st.expander("⚠️ Peringatan", expanded=False):
                    for warning in warnings:
                        st.warning(warning)

            st.success(f"✅ Data berhasil diupload! Total: {len(df_clean)} baris data")

            st.session_state.df = df_clean

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

    except Exception as e:
        st.error(f"❌ **Error saat membaca file:** {str(e)}")
        st.info("💡 **Tips:** Pastikan file CSV menggunakan separator titik koma (;) dan encoding UTF-8")

# ========== BAGIAN 2: PILIH FITUR ==========
if st.session_state.df is not None:
    st.markdown("---")
    st.header("2️⃣ Pilih Fitur untuk Clustering")

    df = st.session_state.df

    available_features = ['Jumlah', 'Harga', 'Jenis Obat']

    st.info("📌 **Petunjuk:** Pilih minimal 2 fitur untuk clustering. Disarankan menggunakan 3 fitur untuk visualisasi 3D.")

    selected_features = st.multiselect(
        "Pilih fitur yang akan digunakan:",
        available_features,
        default=available_features,
        help="Fitur-fitur ini akan digunakan untuk mengelompokkan data obat"
    )

    if len(selected_features) < 2:
        st.warning("⚠️ Pilih minimal 2 fitur untuk melanjutkan!")
    else:
        st.session_state.selected_features = selected_features
        st.success(f"✅ Fitur terpilih: {', '.join(selected_features)}")

        with st.expander("📊 Statistik Deskriptif", expanded=False):
            st.dataframe(df[selected_features].describe(), use_container_width=True)

# ========== BAGIAN 3: JUMLAH CLUSTER ==========
if st.session_state.df is not None and len(st.session_state.selected_features) >= 2:
    st.markdown("---")
    st.header("3️⃣ Tentukan Jumlah Cluster")

    df = st.session_state.df
    selected_features = st.session_state.selected_features

    X = df[selected_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("📈 Metode Elbow")

    with st.spinner("Membuat plot Elbow Method..."):
        max_k = min(10, len(df) - 1)
        fig_elbow, sse_values, k_range = create_elbow_plot(X_scaled, max_k)
        st.pyplot(fig_elbow)
        plt.close()

    with st.expander("📋 Tabel Nilai SSE", expanded=False):
        sse_df = pd.DataFrame({
            'Jumlah Cluster (k)': k_range,
            'SSE': sse_values
        })
        st.dataframe(sse_df, use_container_width=True)

    st.subheader("🎯 Pilih Jumlah Cluster")

    col1, col2 = st.columns([2, 1])

    with col1:
        n_clusters = st.slider(
            "Jumlah Cluster (k):",
            min_value=2,
            max_value=min(10, len(df) - 1),
            value=3,
            help="Pilih jumlah cluster berdasarkan plot Elbow Method di atas"
        )

    with col2:
        st.metric("Cluster Terpilih", n_clusters)

    st.session_state.n_clusters = n_clusters

    if st.button("🚀 Jalankan Clustering", type="primary", use_container_width=True):
        with st.spinner("Melakukan clustering..."):
            df_clustered, centroids, scaler_final, X_scaled_final, dbi_score = perform_clustering(
                df.copy(), selected_features, n_clusters
            )

            st.session_state.df_clustered = df_clustered
            st.session_state.centroids = centroids
            st.session_state.scaler = scaler_final
            st.session_state.X_scaled = X_scaled_final
            st.session_state.dbi_score = dbi_score

            st.success(f"✅ Clustering berhasil! Davies-Bouldin Index (DBI): {dbi_score:.4f}")

            # Interpretasi DBI berdasarkan nilai
            if dbi_score < 0.5:
                quality = "**Good clustering quality** ✅"
                color = "green"
            elif 0.5 <= dbi_score <= 1.0:
                quality = "**Fair clustering quality** ⚠️"
                color = "orange"
            else:
                quality = "**Poor clustering quality** ❌"
                color = "red"

            st.markdown(f"**Kualitas Clustering:** :{color}[{quality}]")

            with st.expander("📊 Interpretasi Davies-Bouldin Index (DBI)", expanded=False):
                st.markdown("""
                Nilai DBI yang lebih rendah menunjukkan clustering yang lebih baik (cluster lebih terpisah dan lebih kompak).

                **Panduan Interpretasi DBI:**

                | DBI Value | Clustering Quality |
                |-----------|-------------------|
                | Low (e.g., < 0.5) | Good clustering quality |
                | Moderate (e.g., 0.5-1.0) | Fair clustering quality |
                | High (e.g., > 1.0) | Poor clustering quality |

                **DBI Anda:** {:.4f} → {}
                """.format(dbi_score, quality))

# ========== BAGIAN 4: VISUALISASI HASIL ==========
if st.session_state.df_clustered is not None:
    st.markdown("---")
    st.header("4️⃣ Visualisasi Hasil Clustering")

    df_clustered = st.session_state.df_clustered
    centroids = st.session_state.centroids
    selected_features = st.session_state.selected_features
    n_clusters = st.session_state.n_clusters
    dbi_score = st.session_state.dbi_score

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi Cluster", "📈 Visualisasi 2D", "🎨 Visualisasi 3D", "📋 Detail Cluster"])

    with tab1:
        st.subheader("Distribusi Data per Cluster")

        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(
                pd.DataFrame({
                    'Cluster': cluster_counts.index,
                    'Jumlah Obat': cluster_counts.values,
                    'Persentase': [f"{(v/len(df_clustered)*100):.1f}%" for v in cluster_counts.values]
                }),
                use_container_width=True
            )

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['orange', 'green', 'red', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            ax.bar(cluster_counts.index, cluster_counts.values,
                   color=[colors[i % len(colors)] for i in cluster_counts.index])
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel('Jumlah Obat', fontsize=12)
            ax.set_title('Distribusi Obat per Cluster', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            for i, v in enumerate(cluster_counts.values):
                ax.text(cluster_counts.index[i], v, str(v), ha='center', va='bottom')

            st.pyplot(fig)
            plt.close()

    with tab2:
        st.subheader("Visualisasi 2D Scatter Plot")

        col1, col2 = st.columns(2)

        with col1:
            feature_x = st.selectbox("Pilih fitur untuk sumbu X:", selected_features, index=0, key="x_axis")

        with col2:
            feature_y = st.selectbox("Pilih fitur untuk sumbu Y:", selected_features,
                                     index=min(1, len(selected_features)-1), key="y_axis")

        if feature_x != feature_y:
            with st.spinner("Membuat visualisasi 2D..."):
                fig_2d = create_2d_visualization(df_clustered, feature_x, feature_y,
                                                 centroids, st.session_state.scaler,
                                                 selected_features)
                st.pyplot(fig_2d)
                plt.close()
        else:
            st.warning("⚠️ Pilih fitur yang berbeda untuk sumbu X dan Y!")

    with tab3:
        st.subheader("Visualisasi 3D Scatter Plot")

        if len(selected_features) >= 3:
            with st.spinner("Membuat visualisasi 3D..."):
                fig_3d = create_3d_visualization(df_clustered, selected_features, centroids)
                if fig_3d:
                    st.pyplot(fig_3d)
                    plt.close()
        else:
            st.info("ℹ️ Visualisasi 3D memerlukan minimal 3 fitur. Silakan pilih 3 fitur pada bagian 'Pilih Fitur'.")

    with tab4:
        st.subheader("Detail Informasi per Cluster")

        for cluster in sorted(df_clustered['Cluster'].unique()):
            with st.expander(f"📦 Cluster {cluster} ({len(df_clustered[df_clustered['Cluster'] == cluster])} obat)", expanded=False):
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster]

                st.write("**Statistik Fitur:**")
                stats_df = cluster_data[selected_features].describe()
                st.dataframe(stats_df, use_container_width=True)

                st.write("**Daftar Obat:**")
                display_cols = ['Nama Obat'] + selected_features + ['Cluster']
                st.dataframe(
                    cluster_data[display_cols].reset_index(drop=True),
                    use_container_width=True,
                    height=300
                )

# ========== BAGIAN 5: DOWNLOAD LAPORAN ==========
if st.session_state.df_clustered is not None:
    st.markdown("---")
    st.header("5️⃣ Download Laporan")

    df_clustered = st.session_state.df_clustered
    centroids = st.session_state.centroids
    selected_features = st.session_state.selected_features
    n_clusters = st.session_state.n_clusters
    dbi_score = st.session_state.dbi_score

    col1, col2, col3 = st.columns(3)

    with col1:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Data dengan cluster
            df_clustered.to_excel(writer, sheet_name='Data Clustering', index=False)

            # Sheet 2: Centroid
            centroid_df = pd.DataFrame(centroids, columns=[f"{feat}_scaled" for feat in selected_features])
            centroid_df['Cluster'] = range(n_clusters)
            centroid_df.to_excel(writer, sheet_name='Centroid', index=False)

            # Sheet 3: Statistik per cluster
            stats_list = []
            for cluster in sorted(df_clustered['Cluster'].unique()):
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
                stats = {'Cluster': cluster, 'Jumlah_Obat': len(cluster_data)}
                for feat in selected_features:
                    stats[f'{feat}_mean'] = cluster_data[feat].mean()
                    stats[f'{feat}_std'] = cluster_data[feat].std()
                stats_list.append(stats)

            stats_df = pd.DataFrame(stats_list)
            stats_df.to_excel(writer, sheet_name='Statistik Cluster', index=False)

        output.seek(0)

        st.download_button(
            label="📥 Download Excel",
            data=output,
            file_name=f"hasil_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col2:
        # Download laporan text
        report_text = generate_report(df_clustered, centroids, selected_features, n_clusters, dbi_score)

        st.download_button(
            label="📄 Download Laporan TXT",
            data=report_text,
            file_name=f"laporan_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col3:
        # Download data CSV
        csv = df_clustered.to_csv(index=False, sep=';')

        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"data_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.success("✅ Laporan siap diunduh! Pilih format yang diinginkan di atas.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>📊 Aplikasi Clustering K-Means | Analisis Penjualan Obat</p>
    </div>
    """,
    unsafe_allow_html=True
)
