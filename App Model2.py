import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from streamlit_lottie import st_lottie

# SECI BANNER
col_left, col_mid, col_right = st.columns([1, 2, 1])
with col_mid:
    try:
        st.image("SECI.png", use_container_width=True)
    except:
        st.markdown("<h1 style='text-align: center; color: #1a4a7a;'>SECI SOLAR DASHBOARD</h1>", unsafe_allow_html=True)

st.markdown("<hr style='border: 1.5px solid #1a4a7a; margin-top: 0;'>", unsafe_allow_html=True)

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="SECI Solar Intelligence", layout="wide")

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

LOTTIE_SUN = "https://assets5.lottiefiles.com/private_files/lf30_moat3dxk.json"
LOTTIE_CLOUD = "https://assets3.lottiefiles.com/packages/lf20_fiq9v72r.json"
LOTTIE_NIGHT = "https://assets8.lottiefiles.com/packages/lf20_96py9mke.json"

@st.cache_resource
def load_assets():
    # Loading your specific non-monsoon model
    model = joblib.load('solar_model_2.pkl')
    df = pd.read_csv('enhanced_model_results_2.csv')
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['Actual'] = df['Actual'].clip(lower=0)
    df['Predicted'] = df['Predicted'].clip(lower=0)
    return model, df

try:
    model, results_df = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

#  2. SIDEBAR NAVIGATION 
st.sidebar.title("Navigation Menu")
page = st.sidebar.radio("Go to:", ["Main Dashboard", "Live Prediction Tool", "Model Analytics"])

#  PAGE 1: MAIN DASHBOARD 

if page == "Main Dashboard":
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0288d1 0%, #e1f5fe 100%);
        }
        /* Ensuring text is bold and visible against the blue background */
        h1, h2, h3, p, label {
            font-weight: 900 ;
            color: #000000 ;
        }
        
        div[data-testid="stDataFrame"] {
            background-color: rgba(255, 255, 255, 0.9) !important;
            border-radius: 10px;
            padding: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: white !important;'>📊 SECI Main Dashboard</h1>", unsafe_allow_html=True)

if page == "Main Dashboard":
    st.markdown("<h2 style='text-align: center;'>Plant Performance Analytics</h2>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    
    # Metric 1: Peak Generation
    c1.metric("Projected Peak", "775.10 MWh")
    
    # Metric 2: Monthly Total (from your data)
    c2.metric("Monthly Generation", "23,159.90 MWh")
    
    # Metric 3: Model Accuracy (MAE)
    c3.metric("Model Precision", "94.06%", delta="MAE: 5.66 MWh")
    
    # Metric 4: Average Daily Output
    c4.metric("Avg Daily Output", "772.00 MWh")

    st.write("---")
    st.info("💡 Use the slider below the graph to zoom into specific days or cloud events.")
    st.subheader("🔍 Generation Trend Analysis (Historical)")
    
    fig_line = px.line(results_df, x='TimeStamp', y=['Actual', 'Predicted'],
                       labels={"value": "Power (MW)", "TimeStamp": "Time"},
                       color_discrete_map={"Actual": "#1fb41f", "Predicted": "#ef553b"})
    
    fig_line.update_layout(
        font={'family': "Times New Roman"},
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line, use_container_width=True)

# PAGE 2: LIVE PREDICTION TOOL 
elif page == "Live Prediction Tool":
    st.markdown("<h2 style='text-align: center; color: #1a4a7a;'>📅 10-Day Automated Forecast</h2>", unsafe_allow_html=True)
    
    # 1. Site Constants (Rajnandgaon)
    LAT, LON = 21.10, 80.99 
    
    # 2. Date Selection (Limited to 10 days ahead as per API limits)
    max_forecast_date = datetime.now() + timedelta(days=10)
    selected_future_date = st.date_input(
        "Select a Target Date for Prediction", 
        min_value=datetime.now().date(),
        max_value=max_forecast_date.date(),
        help="The model will automatically fetch satellite weather data for this day."
    )
    
    if st.button("🔮 Generate 24h Prediction"):
        with st.spinner(f"Fetching Satellite Data for {selected_future_date}..."):
            # API Call for 10-day horizon
            f_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=shortwave_radiation,temperature_2m,surface_pressure,cloudcover&forecast_days=11"
            res = requests.get(f_url)
            
            if res.status_code == 200:
                f_data = res.json()['hourly']
                all_df = pd.DataFrame({
                    'TimeStamp': pd.to_datetime(f_data['time']),
                    'GHI': [x * 1.3 for x in f_data['shortwave_radiation']], 
                    'GII': [x * 1.4 for x in f_data['shortwave_radiation']], 
                    'AMB_TEMP': f_data['temperature_2m'],
                    'AIR_PRESS': f_data['surface_pressure'],
                    'CLOUD_COVER': f_data['cloudcover']
                })

                # Filter for the day you selected
                day_df = all_df[all_df['TimeStamp'].dt.date == selected_future_date].copy()

                if not day_df.empty:
                    # 3. Feature Engineering (Auto-Calculated)
                    day_df['MOD_TEMP'] = day_df['AMB_TEMP'] + 5
                    day_df['Hour'] = day_df['TimeStamp'].dt.hour
                    day_df['Minute'] = day_df['TimeStamp'].dt.minute
                    day_df['Day_of_Year'] = day_df['TimeStamp'].dt.dayofyear
                    day_df['Day_of_Week'] = day_df['TimeStamp'].dt.dayofweek

                    # 4. Prediction Logic
                    cols = ['GHI', 'GII', 'MOD_TEMP', 'AMB_TEMP', 'AIR_PRESS', 'CLOUD_COVER', 'Hour', 'Minute', 'Day_of_Year', 'Day_of_Week']
                    raw_pred = model.predict(day_df[cols])
                    
                    
                    # Applying your site-specific 1.85x scaling for 100MW Hybrid
                    day_df['Predicted_MW'] = (raw_pred * 0.8).clip(0, 100)

                    # 5. Calculations
                    total_mwh = (day_df['Predicted_MW'].sum()*1) # Sum of hourly MW = Total MWh
                    peak_mw = day_df['Predicted_MW'].max()*1.3
                    max_temp = day_df['AMB_TEMP'].max()

                    # --- UI DISPLAY ---
                    st.success(f"✅ Prediction Complete for {selected_future_date}")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Generation", f"{total_mwh:.2f} MWh")
                    m2.metric("Peak Power Output", f"{peak_mw:.2f} MW")
                    m3.metric("Max Ambient Temp", f"{max_temp:.1f} °C")

                else:
                    st.error("Data for selected date is currently unavailable in the forecast.")
            else:
                st.error("Error connecting to weather service API.")


# PAGE 3: MODEL ANALYTICS 
elif page == "Model Analytics":
    
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0288d1 0%, #e1f5fe 100%);
        }
        .analytics-card {
            background-color: #FFFDE7; /* Light Sun Yellow */
            padding: 25px;
            border-radius: 20px;
            border-bottom: 6px solid #FBC02D; /* Golden accent */
            box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
            color: #1a4a7a;
            margin-bottom: 25px;
        }
        h2, h3 { color: #1a4a7a !important; font-family: 'Times New Roman'; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 5px rgba(0,0,0,0.3);'>☀️ Model Accuracy Analytics</h1>", unsafe_allow_html=True)

    # 1. TOP SUMMARY CARDS 
    st.markdown("### 📊 Operational Benchmarks")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown(f"""<div class='analytics-card'><h4>Daily Avg Generation</h4><h2>772.00 MWh</h2></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='analytics-card'><h4>Average Daily Error</h4><h2>5.66 MWh</h2></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='analytics-card'><h4>Safe Daily Output</h4><h2>766.34 MWh</h2></div>""", unsafe_allow_html=True)

    # 2. TABLE 1: MODEL PREDICTION VALUES 
    st.write("---")
    st.markdown("### 📋 Periodical Forecast Accuracy")
    
    data1 = {
        "Time Period": ["1 DAY", "7 DAYS", "15 DAYS", "1 MONTH"],
        "Actual (MWh)": [771.98, 4703.75, 11579.95, 23159.90],
        "Predicted (MWh)": [765.89, 4777.96, 11488.47, 22976.94],
        "Error (%)": [0.79, 1.58, 0.79, 0.78]
    }
    df1 = pd.DataFrame(data1)
    
    # Styled Table with Dynamic Yellow/Blue Shading
    st.dataframe(df1.style.background_gradient(cmap='YlGnBu', subset=["Error (%)"]).format({
        "Actual (MWh)": "{:,.2f}", "Predicted (MWh)": "{:,.2f}", "Error (%)": "{:.2f}%"
    }), use_container_width=True)

    #  3. DYNAMIC GRAPHS 
    st.write("---")
    g1, g2 = st.columns(2)

    with g1:
        # Graph 1: Actual vs Predicted (Bar Chart)
        st.markdown("#### Actual vs Predicted Horizon")
        fig_bar = go.Figure(data=[
            go.Bar(name='Actual', x=df1['Time Period'], y=df1['Actual (MWh)'], marker_color='#1a4a7a'),
            go.Bar(name='Predicted', x=df1['Time Period'], y=df1['Predicted (MWh)'], marker_color='#FBC02D')
        ])
        fig_bar.update_layout(barmode='group', height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

    with g2:
        # Graph 2: Error Trend (Line Chart)
        st.markdown("#### Precision Stability (Error %)")
        fig_line = px.line(df1, x="Time Period", y="Error (%)", markers=True, 
                           color_discrete_sequence=["#1a4a7a"])
        fig_line.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_line, use_container_width=True)

    # Graph 3: Safe Output Projection (Area Chart)
    st.markdown("#### Cumulative Generation Security")
    fig_area = px.area(df1, x="Time Period", y=["Actual (MWh)", "Predicted (MWh)"], 
                       color_discrete_map={"Actual (MWh)": "#1a4a7a", "Predicted (MWh)": "#FBC02D"})
    fig_area.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_area, use_container_width=True)

    # --- 4. TABLE 2: AVERAGE STATS (Updated Data) ---
    st.write("---")
    st.markdown("### 🔍 Statistical Confidence Levels")
    
    # Updated values 
    data2 = {
        "Metric Type": ["Daily Average", "Monthly Total"],
        "Generation (MWh)": [772.00, 23159.90],
        "Avg Error (MWh)": [5.66, 169.80],
        "Safe Output (MWh)": [766.34, 22990.10],
        "Error Percentage": ["0.738%", "0.738%"]
    }
    st.table(pd.DataFrame(data2))

    # --- DYNAMIC GRAPHS FOR TABLE 2 ---
    st.markdown("#### 📈 Confidence & Error Impact Analysis")
    
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # 1. Comparison of Generation vs Safe Output (Daily)
        fig_daily = go.Figure(data=[
            go.Bar(name='Expected Generation', x=['Daily Avg'], y=[772.00], marker_color='#FBC02D'),
            go.Bar(name='Safe Output', x=['Daily Avg'], y=[766.34], marker_color='#1a4a7a')
        ])
        fig_daily.update_layout(
            title="Daily Reliability (MWh)",
            barmode='group',
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': "Times New Roman"}
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    with col_g2:
        # 2. Error Breakdown Pie Chart (Monthly)
        fig_pie = px.pie(
            values=[22990.10, 169.80], 
            names=['Safe Output', 'Avg Error'],
            color_discrete_sequence=['#1a4a7a', '#ef553b'],
            title="Monthly Margin of Error"
        )
        fig_pie.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    # 3. Precision Indicator (Bullet Chart) - Updated to 0.738%
    st.markdown("#### 🎯 Model Precision Indicator")
    fig_bullet = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 0.738,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Error Percentage (%)", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 2], 'tickwidth': 1},
            'bar': {'color': "#ef553b"},
            'steps': [
                {'range': [0, 0.5], 'color': "#e8f5e9"},
                {'range': [0.5, 1.0], 'color': "#fff3e0"},
                {'range': [1.0, 2.0], 'color': "#ffebee"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 0.738}
        }
    ))
    
    # FIXED: Height must be 10 or more. Changed from 2 to 250.
    fig_bullet.update_layout(
        height=250, 
        margin=dict(t=50, b=0), 
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_bullet, use_container_width=True)


st.markdown("<div style='text-align: center; color: gray; margin-top: 50px;'>SECI Analytics Division © 2026</div>", unsafe_allow_html=True)
