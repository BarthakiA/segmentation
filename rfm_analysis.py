import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data(filepath='NYKA.csv'):
    """
    Load the NYKA customer dataset.
    """
    df = pd.read_csv(filepath, parse_dates=['signup_date','last_purchase_date'])
    return df

def prepare_rfm(df):
    """
    Prepare the RFM table from the loaded DataFrame.
    """
    rfm = df[['customer_id','recency_days','frequency_3m','monetary_value_3m']].copy()
    rfm.rename(columns={
        'recency_days':'Recency',
        'frequency_3m':'Frequency',
        'monetary_value_3m':'Monetary'
    }, inplace=True)
    rfm.set_index('customer_id', inplace=True)
    return rfm

def score_rfm(rfm):
    """
    Score customers on Recency, Frequency, Monetary using quartiles.
    """
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], 4, labels=[1,2,3,4]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4]).astype(int)
    rfm['RFM_Score'] = (
        rfm['R_Score'].map(str) +
        rfm['F_Score'].map(str) +
        rfm['M_Score'].map(str)
    )
    return rfm

def segment_rfm(rfm):
    """
    Assign human-readable segment labels based on RFM score patterns.
    """
    import re
    seg_map = {
        r'5[4-5][4-5]': 'Champions',
        r'4[4-5][4-5]': 'Loyal Customers',
        r'[3-4][1-3][1-3]': 'Potential Loyalist',
        r'[1-2][1-2][1-2]': 'At Risk',
    }
    def label_segment(score):
        for pattern, label in seg_map.items():
            if re.match(pattern, score):
                return label
        return 'Others'

    rfm['Segment'] = rfm['RFM_Score'].apply(label_segment)
    return rfm

def cluster_rfm(rfm, n_clusters=4):
    """
    Apply K-Means clustering on RFM values to find behavioral clusters.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(X)
    return rfm

def save_rfm(rfm, filepath='rfm_results.csv'):
    """
    Save the RFM table with scores, segments, and clusters.
    """
    rfm.to_csv(filepath)

if __name__ == '__main__':
    df = load_data()
    rfm = prepare_rfm(df)
    rfm = score_rfm(rfm)
    rfm = segment_rfm(rfm)
    rfm = cluster_rfm(rfm)
    save_rfm(rfm)
    print("RFM analysis complete. Results saved to rfm_results.csv.")