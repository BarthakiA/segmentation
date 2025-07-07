import pandas as pd
import matplotlib.pyplot as plt
import re
from lifetimes import BetaGeoFitter, GammaGammaFitter

# 1. Load and prepare data
df = pd.read_csv('NYKA.csv', parse_dates=['signup_date','last_purchase_date'])
analysis_date = df['last_purchase_date'].max()

# Observation period (T): days since first signup
df['T'] = (analysis_date - df['signup_date']).dt.days

# Recency for BG/NBD: time from first purchase to last purchase
df['recency'] = df['T'] - df['recency_days']

# Frequency in calibration period
df['frequency'] = df['frequency_3m']

# Average monetary value per transaction
df['monetary_avg'] = df.apply(
    lambda r: r['monetary_value_3m'] / r['frequency_3m']
    if r['frequency_3m'] > 0 else 0,
    axis=1
)

# Keep only customers with at least one purchase
calib = df[df['frequency'] > 0].copy()

# 2. Fit the BG/NBD model for purchase frequency
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(
    calib['frequency'],
    calib['recency'],
    calib['T']
)

# 3. Fit the Gamma–Gamma model for monetary value
ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(
    calib['frequency'],
    calib['monetary_avg']
)

# 4. Predict CLV for next 6 months ( ≈ 180 days )
calib['clv_6m'] = ggf.customer_lifetime_value(
    bgf,
    calib['frequency'],
    calib['recency'],
    calib['T'],
    calib['monetary_avg'],
    time=6,
    freq='D'
)

# 5. Plot distribution of predicted 6-month CLV
plt.figure()
plt.hist(calib['clv_6m'], bins=30)
plt.title('Predicted 6-Month CLV Distribution')
plt.xlabel('CLV (Currency Units)')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()
print("Most customers have low predicted CLV over the next 6 months.")
print("A small group of high-value customers stands out for targeted outreach.")

# 6. Average CLV by RFM segment
def prepare_rfm(df):
    rfm = df[['customer_id','recency','frequency','monetary_avg']].copy()
    rfm.rename(columns={
        'recency':'Recency',
        'frequency':'Frequency',
        'monetary_avg':'Monetary'
    }, inplace=True)
    rfm.set_index('customer_id', inplace=True)
    return rfm

def score_rfm(rfm):
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
    seg_map = {
        r'5[4-5][4-5]': 'Champions',
        r'4[4-5][4-5]': 'Loyal Customers',
        r'[3-4][1-3][1-3]': 'Potential Loyalist',
        r'[1-2][1-2][1-2]': 'At Risk'
    }
    def label_segment(score):
        for pattern, label in seg_map.items():
            if re.match(pattern, score):
                return label
        return 'Others'
    rfm['Segment'] = rfm['RFM_Score'].apply(label_segment)
    return rfm

rfm = prepare_rfm(calib.reset_index())
rfm = score_rfm(rfm)
rfm = segment_rfm(rfm)
rfm['CLV_6m'] = calib.set_index('customer_id')['clv_6m']

avg_clv = rfm.groupby('Segment')['CLV_6m'].mean().sort_values()

plt.figure()
avg_clv.plot(kind='bar')
plt.title('Avg 6-Month CLV by RFM Segment')
plt.xlabel('Segment')
plt.ylabel('Avg CLV (Currency Units)')
plt.tight_layout()
plt.show()
print("Top segments like Champions exhibit the highest average CLV.")
print("At-Risk and Others show the lowest CLV, indicating retention opportunities.")
