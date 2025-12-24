import streamlit as st
import pandas as pd
import banana_model
import os
import importlib
import time

# Force reload backend logic on every run (dev mode hack)
importlib.reload(banana_model)

# Page Config
st.set_page_config(page_title="Banana Harvest Predictor", page_icon="🍌", layout="wide")

st.title("🍌 Banana Harvest Intelligence System")
st.markdown("### Long-Term Forecasting (Mother -> Daughter Cycle)")

# 1. Live Data Loader
DATA_FILE = 'banana_farm_data.csv'

@st.cache_data(ttl=60) # Cache for 1 min, then auto-reload
def load_live_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return None

data = load_live_data()

# Refresh checking
if st.button("🔄 Refresh Data Source"):
    st.cache_data.clear()
    st.experimental_rerun()

if data is not None:
    # Sidebar
    st.sidebar.header("Farm Status")
    last_row = data.iloc[-1]
    st.sidebar.info(f"**Current Week:** {int(last_row['Year'])}-W{int(last_row['Week'])}")
    st.sidebar.write(f"**Latest Bagging:** {int(last_row['Bagging_Count'])}")
    st.sidebar.write(f"**Latest Harvest:** {int(last_row['Harvest_Count'])}")
    
    # 2. Model Training (Two-Stage)
    with st.spinner("Analyzing Farm Cycles (Succession & Maturation)..."):
        system = banana_model.train_full_system(data)
        
    # Metrics
    m1 = system['bagging_metrics']
    m2 = system['harvest_metrics']
    
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    exp_col1.metric("Stage 1: Succession Accuracy", f"{m1['R2']*100:.1f}%", help="Predicting Daughter Bagging from Mother Harvest")
    exp_col2.metric("Stage 2: Maturation Accuracy", f"{m2['R2']*100:.1f}%", help="Predicting Harvest from Bagging")
    exp_col3.metric("Prediction Confidence", f"±{int(m2['Std_Error']*1.96)} Bunches")
    
    st.markdown("---")
    
    # 3. Long Term Forecast (1 Year)
    st.subheader("📅 1-Year Strategic Forecast")
    
    forecast_df = banana_model.predict_long_term(data, system, weeks_ahead=52)
    
    # Create the text label for the chart
    forecast_df['Week_Label'] = forecast_df['Year'].astype(str) + "-W" + forecast_df['Week'].astype(str)
    
    # Chart
    st.line_chart(forecast_df.set_index('Week_Label')[['Harvest_Count', 'Lower_Bound', 'Upper_Bound']])
    
    # Detailed Table with Colors
    st.subheader("📋 Weekly Schedule & Colors")
    
    # Format for display
    display_df = forecast_df[['Year', 'Week', 'Bag_Color', 'String_Color', 'Bagging_Count', 'Harvest_Count', 'Lower_Bound', 'Upper_Bound']].copy()
    display_df['Bagging_Count'] = display_df['Bagging_Count'].astype(int)
    display_df['Harvest_Count'] = display_df['Harvest_Count'].astype(int)
    
    # Apply some styling (using pandas Styler if we wanted, but standard dataframe is fine for now)
    st.dataframe(
        display_df,
        column_config={
            "Year": st.column_config.NumberColumn(format="%d"),
            "Bagging_Count": st.column_config.NumberColumn("Predicted Bagging", help="Due to Mother Harvest 6 months ago"),
            "Harvest_Count": st.column_config.NumberColumn("Predicted Harvest", help="Due to Bagging 4 months ago"),
            "Lower_Bound": st.column_config.NumberColumn("Min Expected"),
            "Upper_Bound": st.column_config.NumberColumn("Max Expected"),
        },
        use_container_width=True,
        height=600
    )
    
    # Download
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast Plan", csv, "1_year_banana_plan.csv", "text/csv")

else:
    st.error(f"⚠️ Live Data File Not Found: `{DATA_FILE}`")
    st.info("Please ensure your Excel/CSV file is saved in the project folder with this name.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Prepared by **Rajaneesh Anidil**")
