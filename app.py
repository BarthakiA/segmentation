import streamlit as st
import pandas as pd
import plotly.express as px
from rfm_analysis import prepare_rfm, score_rfm, segment_rfm, cluster_rfm

st.set_page_config(page_title="NYKA Customer Segmentation", layout="wide")
st.title("NYKA Customer Segmentation Dashboard")

@st.cache_data
def load_data(filepath='NYKA.csv'):
    return pd.read_csv(filepath)

df = load_data()
rfm = prepare_rfm(df)
rfm = score_rfm(rfm)
rfm = segment_rfm(rfm)
rfm = cluster_rfm(rfm)

# Sidebar filters
st.sidebar.header("Filter Options")
segments = sorted(rfm['Segment'].unique())
selected_segments = st.sidebar.multiselect("Select Segments", segments, default=segments)
clusters = sorted(rfm['Cluster'].unique())
selected_clusters = st.sidebar.multiselect("Select Clusters", clusters, default=clusters)
filtered = rfm[(rfm['Segment'].isin(selected_segments)) & (rfm['Cluster'].isin(selected_clusters))]

# RFM Distributions
st.header("RFM Distributions")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Recency")
    fig_r = px.histogram(rfm, x='Recency', nbins=20)
    st.plotly_chart(fig_r, use_container_width=True)
with col2:
    st.subheader("Frequency")
    fig_f = px.histogram(rfm, x='Frequency', nbins=20)
    st.plotly_chart(fig_f, use_container_width=True)
with col3:
    st.subheader("Monetary")
    fig_m = px.histogram(rfm, x='Monetary', nbins=20)
    st.plotly_chart(fig_m, use_container_width=True)

# Segment and Cluster Counts
st.header("Segment Counts")
seg_counts = rfm['Segment'].value_counts().reset_index()
seg_counts.columns = ['Segment','Count']
fig_s = px.bar(seg_counts, x='Segment', y='Count')
st.plotly_chart(fig_s, use_container_width=True)

st.header("Cluster Counts")
cluster_counts = rfm['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster','Count']
fig_c = px.bar(cluster_counts, x='Cluster', y='Count')
st.plotly_chart(fig_c, use_container_width=True)

# 3D Scatter Plot of Clusters
st.header("3D Cluster Visualization")
fig_3d = px.scatter_3d(
    rfm.reset_index(),
    x='Recency',
    y='Frequency',
    z='Monetary',
    color='Cluster',
    symbol='Cluster',
    title='3D RFM Cluster Plot'
)
st.plotly_chart(fig_3d, use_container_width=True)

# Filtered Customer Details
st.header("Filtered Customer Details")
st.dataframe(filtered)