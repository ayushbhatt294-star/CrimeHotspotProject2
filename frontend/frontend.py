import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import numpy as np
from pathlib import Path

# Utility functions for risk analysis
def build_cluster_hour_risk(df):
    """Build risk scores for each cluster by hour"""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
            
        # Create pivot table of crime counts by cluster and hour
        pivot = pd.pivot_table(
            df,
            values='crime_type',
            index='cluster',
            columns='hour',
            aggfunc='count',
            fill_value=0
        )
        
        if pivot.empty:
            return pd.DataFrame()
        
        # Normalize by hour
        hour_totals = pivot.sum()
        if hour_totals.sum() == 0:
            return pd.DataFrame()
            
        normalized = pivot.div(hour_totals, axis=1)
        
        # Calculate risk scores (0-100)
        risk_scores = (normalized * 100).round(1)
        risk_scores.columns = [f"{hour:02d}:00" for hour in risk_scores.columns]
        
        return risk_scores
    except Exception as e:
        print(f"Error building risk scores: {e}")
        return pd.DataFrame()

def top_risk_windows(risk_df, n=5):
    """Find top N highest risk time windows for each cluster"""
    try:
        if risk_df is None or risk_df.empty:
            return pd.DataFrame()
            
        results = []
        for cluster in risk_df.index:
            cluster_risks = risk_df.loc[cluster]
            top_hours = cluster_risks.nlargest(n)
            for hour, risk in top_hours.items():
                results.append({
                    'cluster': cluster,
                    'time_window': hour,
                    'risk_score': risk
                })
        return pd.DataFrame(results).sort_values('risk_score', ascending=False)
    except Exception as e:
        print(f"Error finding top risk windows: {e}")
        return pd.DataFrame()

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Smart Patrol AI - Real-time",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR STYLING
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .risk-very-high {
        background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    
    .realtime-badge {
        display: inline-block;
        background: #ff4757;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .alert-box {
        background: #fff3cd;
        border-left: 4px solid #ff4757;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .patrol-card {
        background: white;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .immediate-priority {
        border-left-color: #ff0844;
        background: #fff5f5;
    }
    
    .high-priority {
        border-left-color: #ff6b6b;
        background: #fff8f8;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# API CONFIGURATION
# ============================================
API_BASE_URL = "https://crimehotspotproject2-1.onrender.com/api"



def fetch_api_data(endpoint):
    """Fetch data from backend API with proper error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=5)
        if response.status_code == 200:
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError:
                st.error(f"⚠️ Invalid JSON response from {endpoint}")
                return None
        else:
            st.error(f"⚠️ API Error {response.status_code} from {endpoint}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Cannot connect to API at {API_BASE_URL}. Please ensure the backend server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error(f"⚠️ API request timeout for {endpoint}")
        return None
    except Exception as e:
        st.error(f"⚠️ API Error: {str(e)}")
        return None

def post_api_data(endpoint, data):
    """Post data to backend API with proper error handling"""
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint}", json=data, timeout=10)
        if response.status_code == 200:
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError:
                return {'error': 'Invalid JSON response from server'}
        else:
            try:
                error_msg = response.json().get('error', f'HTTP {response.status_code}')
            except:
                error_msg = f'HTTP Error {response.status_code}'
            return {'error': error_msg}
    except requests.exceptions.ConnectionError:
        return {'error': f"Cannot connect to API at {API_BASE_URL}. Please ensure backend is running."}
    except requests.exceptions.Timeout:
        return {'error': "Request timeout - server took too long to respond"}
    except Exception as e:
        return {'error': f"API Error: {str(e)}"}

# ============================================
# HEADER WITH LIVE STATUS
# ============================================
st.markdown('<h1 class="main-header">🚓 Smart Patrol AI - Real-time Dashboard</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>AI-Powered Crime Prediction & Live Patrol Optimization</p>", unsafe_allow_html=True)

# Live status indicator
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    status_data = fetch_api_data("status")
    if status_data and status_data.get('api') == 'running':
        realtime_info = status_data.get('realtime_status', {})
        st.success(f"🟢 System Online | Mode: Real-time | New Crimes: {realtime_info.get('new_crimes_reported', 0)}")
    else:
        st.error("🔴 System Offline - Please start the backend server")
        with st.expander("🔧 Troubleshooting"):
            st.code("""
# To start the backend server:
cd backend
uvicorn app.main:app --reload --port 8000

# Or if using Python directly:
python -m uvicorn app.main:app --reload --port 8000
            """)
            st.info(f"Trying to connect to: {API_BASE_URL}")
st.markdown("---")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/police-badge.png", width=80)
    st.title("Navigation")
    
    page = st.radio(
        "Select View:",
        ["🏠 Real-time Overview", "📝 Report Crime", "🗺️ Crime Map", "📊 Analytics", 
         "⚠️ Risk Predictions", "🚨 Live Patrol", "📍 Area Details"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.subheader("Quick Stats")
    
    if status_data:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Historical", status_data.get('historical_data', {}).get('total_crimes', 0))
        with col2:
            realtime_crimes = status_data.get('realtime_status', {}).get('new_crimes_reported', 0)
            st.metric("New Today", realtime_crimes, delta=realtime_crimes if realtime_crimes > 0 else None)
        
        st.metric("Total Areas", status_data.get('historical_data', {}).get('total_areas', 0))
    
    st.markdown("---")
    
    # Real-time updates toggle
    auto_refresh = st.checkbox("🔴 Auto Refresh (30s)", value=False)
    
    if status_data and status_data.get('realtime_status'):
        last_update = status_data['realtime_status'].get('last_update', '')
        if last_update:
            try:
                update_time = datetime.fromisoformat(last_update).strftime('%H:%M:%S')
                st.caption(f"Last Update: {update_time}")
            except:
                st.caption(f"Last Update: {last_update}")
    
    if auto_refresh:
        st.caption("⏱️ Next refresh in 30s")

# Initialize session state
if 'show_hotspots' not in st.session_state:
    st.session_state['show_hotspots'] = True
if 'show_realtime' not in st.session_state:
    st.session_state['show_realtime'] = True

# Default map style
map_style = "open-street-map"

# ============================================
# PAGE: REAL-TIME OVERVIEW
# ============================================
if page == "🏠 Real-time Overview":
    st.header("📊 Real-time System Overview")
    
    # Fetch data
    crime_stats = fetch_api_data("crime-stats")
    realtime_stats = fetch_api_data("realtime-stats")
    patrol_data = fetch_api_data("patrol-recommendations")
    
    if crime_stats:
        # Real-time Alert Banner
        if realtime_stats and realtime_stats.get('total_new_crimes', 0) > 0:
            try:
                last_update = datetime.fromisoformat(realtime_stats['last_update']).strftime('%H:%M:%S')
            except:
                last_update = realtime_stats.get('last_update', 'Unknown')
                
            st.markdown(f"""
            <div class="alert-box">
                <h3>⚠️ LIVE ALERT: {realtime_stats['total_new_crimes']} New Crime(s) Reported</h3>
                <p>Areas affected: {realtime_stats['areas_affected']} | Last update: {last_update}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            total = crime_stats.get('historical', {}).get('total_crimes', 0)
            new = crime_stats.get('realtime', {}).get('new_crimes_count', 0)
            st.metric("Total Crimes", f"{total + new}", delta=f"+{new} new")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Areas Monitored", crime_stats.get('historical', {}).get('total_areas', 0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            trend = crime_stats.get('combined', {}).get('trend', 'stable')
            trend_emoji = "📈" if trend == "increasing" else "📊"
            st.metric("Trend", trend.title(), delta=trend_emoji)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            active_areas = crime_stats.get('realtime', {}).get('areas_affected', 0)
            st.metric("Active Areas", active_areas, delta=f"{active_areas} today")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Real-time vs Historical
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Historical vs Real-time")
            comparison_data = {
                'Category': ['Historical Crimes', 'New Crimes Today'],
                'Count': [
                    crime_stats.get('historical', {}).get('total_crimes', 0),
                    crime_stats.get('realtime', {}).get('new_crimes_count', 0)
                ]
            }
            fig = px.bar(pd.DataFrame(comparison_data), x='Category', y='Count',
                        color='Category', color_discrete_sequence=['#667eea', '#ff4757'])
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Top Crime Areas (Historical)")
            top_areas_dict = crime_stats.get('historical', {}).get('top_crime_areas', {})
            if top_areas_dict:
                top_areas = pd.DataFrame(
                    list(top_areas_dict.items()),
                    columns=['Area', 'Crimes']
                ).head(5)
                fig = px.bar(top_areas, x='Crimes', y='Area', orientation='h',
                            color='Crimes', color_continuous_scale='Reds')
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No area data available")
        
        # Live Patrol Recommendations
        if patrol_data and patrol_data.get('patrol_suggestions'):
            st.subheader("🚨 IMMEDIATE PATROL ALERTS")
            for alert in patrol_data['patrol_suggestions'][:3]:
                priority_text = alert.get('recommended_time', 'SCHEDULE')
                priority_color = "🔴" if priority_text == 'IMMEDIATE' else "🟡"
                area_name = alert.get('area', 'Unknown Area')
                reason = alert.get('reason', 'High crime activity')
                st.warning(f"{priority_color} **{area_name}**: {reason} - Priority: {priority_text}")

# ============================================
# PAGE: REPORT CRIME (MAIN FEATURE)
# ============================================
elif page == "📝 Report Crime":
    st.header("📝 Report New Crime - Real-time Prediction")
    
    st.info("🔴 **Live System**: Enter crime details to get instant AI risk prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("crime_report_form"):
            st.subheader("Crime Details")
            
            # Essential Information
            area_name = st.text_input("📍 Area/Location Name", 
                                     placeholder="e.g., Connaught Place, Rohini, Dwarka")
            
            crime_type = st.selectbox(
                "🔍 Crime Type",
                ["Cybercrime", "Theft", "Robbery", "Fraud", "Assault", 
                 "Burglary", "Hacking", "Identity Theft", "Phishing", "Other"]
            )
            
            st.markdown("---")
            st.subheader("Area Demographics (Optional - for better prediction)")
            
            col_a, col_b = st.columns(2)
            with col_a:
                population = st.number_input("Population", min_value=1000, max_value=10000000, 
                                            value=100000, step=10000)
                area_sq_km = st.number_input("Area (sq. km)", min_value=0.1, max_value=500.0, 
                                             value=10.0, step=0.5)
                police_stations = st.number_input("Police Stations", min_value=1, max_value=20, 
                                                 value=2)
            
            with col_b:
                avg_response_time = st.number_input("Avg Response Time (min)", 
                                                   min_value=1, max_value=60, value=15)
                crime_rate = st.number_input("Crime Rate per 1000", min_value=0.1, 
                                            max_value=100.0, value=5.0, step=0.1)
            
            st.markdown("---")
            st.subheader("Crime Pattern Data (Optional)")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                weekday_crimes = st.slider("Recent Weekday Crimes", 0, 50, 10)
                weekend_crimes = st.slider("Recent Weekend Crimes", 0, 50, 5)
            with col_b:
                night_crimes = st.slider("Night Crimes", 0, 50, 4)
                evening_crimes = st.slider("Evening Crimes", 0, 50, 3)
            with col_c:
                crime_diversity = st.slider("Crime Type Diversity", 1, 10, 4)
                theft_count = st.slider("Recent Thefts", 0, 20, 2)
            
            submit_button = st.form_submit_button("🚀 Submit & Get Prediction", use_container_width=True)
        
        if submit_button:
            if not area_name:
                st.error("❌ Please enter an area name")
            else:
                with st.spinner("🔄 Processing crime report and generating prediction..."):
                    payload = {
                        "area_name": area_name,
                        "crime_type": crime_type,
                        "population": population,
                        "area_sq_km": area_sq_km,
                        "police_stations": police_stations,
                        "avg_response_time": avg_response_time,
                        "crime_rate_per_1000": crime_rate,
                        "weekday_crimes": weekday_crimes,
                        "weekend_crimes": weekend_crimes,
                        "night_crimes": night_crimes,
                        "evening_crimes": evening_crimes,
                        "crime_diversity": crime_diversity,
                        "theft_count": theft_count,
                        "assault_count": 1,
                        "burglary_count": 0,
                        "population_density": population / area_sq_km
                    }
                    
                    result = post_api_data("report-crime", payload)
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success("✅ Crime Reported Successfully!")
                        
                        # Display prediction results
                        st.markdown("---")
                        st.subheader("🎯 AI Prediction Results")
                        
                        prediction = result.get('prediction', {})
                        if isinstance(prediction, dict):
                            risk_category = str(prediction.get('risk_category', 'N/A'))
                            risk_score = float(prediction.get('risk_score', 0))
                            predicted_crimes = int(prediction.get('predicted_crimes', 0))
                        else:
                            risk_category = 'N/A'
                            risk_score = 0.0
                            predicted_crimes = 0
                        
                        # Risk-based styling
                        risk_class = {
                            'Very High': 'risk-very-high',
                            'High': 'risk-high',
                            'Medium': 'risk-medium',
                            'Low': 'risk-low'
                        }.get(risk_category, 'risk-medium')
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
                            st.metric("Risk Category", risk_category)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_b:
                            st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
                            st.metric("Risk Score", f"{risk_score:.1f}/100")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_c:
                            st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
                            st.metric("Predicted Crimes", f"{predicted_crimes}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Recommendation
                        if risk_category in ['Very High', 'High']:
                            st.error(f"⚠️ **ALERT**: Area '{area_name}' is marked as {risk_category} risk. Immediate patrol recommended!")
                        elif risk_category == 'Medium':
                            st.warning(f"📊 Area '{area_name}' has Medium risk. Regular monitoring suggested.")
                        else:
                            st.info(f"✅ Area '{area_name}' has Low risk. Standard patrol schedule.")
                        
                        # Crime ID
                        try:
                            timestamp_str = datetime.fromisoformat(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            timestamp_str = result.get('timestamp', 'Unknown')
                            
                        st.caption(f"Crime ID: {result.get('crime_id', 'N/A')} | Reported: {timestamp_str}")
    
    with col2:
        st.subheader("📋 Recent Reports")
        realtime_crimes = fetch_api_data("realtime-crimes")
        
        if realtime_crimes and realtime_crimes.get('crimes'):
            st.metric("Total New Reports", realtime_crimes.get('total_new_crimes', 0))
            
            for crime in realtime_crimes['crimes'][-5:]:  # Last 5
                with st.container():
                    st.markdown(f"**{crime.get('area_name', 'Unknown')}**")
                    st.caption(f"{crime.get('crime_type', 'Unknown')} | {crime.get('id', 'N/A')}")
                    try:
                        time_str = datetime.fromisoformat(crime['timestamp']).strftime('%H:%M:%S')
                    except:
                        time_str = crime.get('timestamp', 'Unknown')
                    st.caption(time_str)
                    st.markdown("---")
        else:
            st.info("No new reports yet")

# ============================================
# PAGE: CRIME MAP
# ============================================
elif page == "🗺️ Crime Map":
    st.header("🗺️ Interactive Crime Map - Real-time")
    
    # Load hotspot and crime data
    hotspots_data = fetch_api_data("hotspots") or []
    crimes_data_map = fetch_api_data("crimes") or []

    show_hotspots = st.session_state.get('show_hotspots', True)
    show_realtime = st.session_state.get('show_realtime', True)

    # Normalize hotspots into a DataFrame
    hotspot_df = None
    if hotspots_data and isinstance(hotspots_data, list) and len(hotspots_data) > 0:
        rows = []
        for i, h in enumerate(hotspots_data):
            try:
                center = h.get('center') if isinstance(h, dict) else getattr(h, 'center', None)
                lat = center.get('lat') if isinstance(center, dict) else (getattr(center, 'lat', None) if center is not None else None)
                lon = center.get('lng') if isinstance(center, dict) else (getattr(center, 'lng', None) if center is not None else None)
                crime_count = h.get('crime_count', 0) if isinstance(h, dict) else getattr(h, 'crime_count', 0)
                area_name = h.get('area_name') if isinstance(h, dict) else getattr(h, 'area_name', f'hotspot_{i}')
                crime_types = h.get('crime_types', {}) if isinstance(h, dict) else getattr(h, 'crime_types', {})
                dominant = None
                
                if isinstance(crime_types, dict) and crime_types:
                    dominant = max(crime_types.items(), key=lambda x: x[1])[0]

                rows.append({
                    'area_name': area_name if area_name else f'hotspot_{i}',
                    'center_lat': float(lat) if lat is not None else 0.0,
                    'center_lon': float(lon) if lon is not None else 0.0,
                    'crime_count': int(crime_count),
                    'dominant_crime': dominant or 'Unknown',
                    'has_realtime_activity': bool(h.get('has_realtime_activity', False) if isinstance(h, dict) else getattr(h, 'has_realtime_activity', False))
                })
            except Exception as e:
                print(f"Error processing hotspot {i}: {e}")
                continue

        if rows:
            hotspot_df = pd.DataFrame(rows)
        
    # Render map when we have hotspot data
    if hotspot_df is not None and len(hotspot_df) > 0:
        fig = go.Figure()

        # Add hotspots layer
        if show_hotspots:
            # Separate active and inactive hotspots
            active_hotspots = hotspot_df[hotspot_df['has_realtime_activity'] == True]
            inactive_hotspots = hotspot_df[hotspot_df['has_realtime_activity'] == False]

            # Add inactive hotspots
            if len(inactive_hotspots) > 0:
                fig.add_trace(go.Scattermapbox(
                    lat=inactive_hotspots['center_lat'],
                    lon=inactive_hotspots['center_lon'],
                    mode='markers',
                    marker=dict(
                        size=inactive_hotspots['crime_count'] * 2,
                        color='orange',
                        opacity=0.6,
                        sizemode='diameter'
                    ),
                    text=inactive_hotspots['area_name'],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Crimes: %{customdata[0]}<br>' +
                                  'Type: %{customdata[1]}<br>' +
                                  'Status: Inactive<extra></extra>',
                    customdata=inactive_hotspots[['crime_count', 'dominant_crime']].values,
                    name='Historical Hotspots'
                ))

            # Add active hotspots
            if len(active_hotspots) > 0:
                fig.add_trace(go.Scattermapbox(
                    lat=active_hotspots['center_lat'],
                    lon=active_hotspots['center_lon'],
                    mode='markers',
                    marker=dict(
                        size=active_hotspots['crime_count'] * 2.5,
                        color='red',
                        opacity=0.8,
                        sizemode='diameter'
                    ),
                    text=active_hotspots['area_name'],
                    hovertemplate='<b>🔴 %{text}</b><br>' +
                                  'Crimes: %{customdata[0]}<br>' +
                                  'Type: %{customdata[1]}<br>' +
                                  'Status: ACTIVE<extra></extra>',
                    customdata=active_hotspots[['crime_count', 'dominant_crime']].values,
                    name='🔴 Active Hotspots'
                ))

        # Add real-time crimes layer
        if show_realtime and crimes_data_map:
            try:
                # Jitter crimes around hotspots for demo
                if len(hotspot_df) > 0:
                    np.random.seed(42)
                    base_lat = hotspot_df['center_lat'].mean()
                    base_lon = hotspot_df['center_lon'].mean()
                    num_crimes = len(crimes_data_map)
                    realtime_lats = base_lat + np.random.uniform(-0.05, 0.05, num_crimes)
                    realtime_lons = base_lon + np.random.uniform(-0.05, 0.05, num_crimes)

                    crime_labels = [c.get('area_name', 'Unknown') for c in crimes_data_map]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=realtime_lats,
                        lon=realtime_lons,
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='#ff4757',
                            opacity=1,
                            symbol='circle'
                        ),
                        text=crime_labels,
                        hovertemplate='<b>NEW CRIME</b><br>' +
                                      'Area: %{text}<br>' +
                                      '<extra></extra>',
                        name='🆕 Live Reports'
                    ))
            except Exception as e:
                print(f"Error adding realtime crimes: {e}")

        # Calculate map center
        center_lat = hotspot_df['center_lat'].mean()
        center_lon = hotspot_df['center_lon'].mean()

        # Update layout
        fig.update_layout(
            mapbox=dict(
                style=map_style,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11
            ),
            height=600,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Map legend
        st.markdown("""
        **Legend:**
        - 🔴 **Red markers (large)**: Active hotspots with recent crime activity
        - 🟠 **Orange markers**: Historical hotspots
        - 🔴 **Small red dots**: Live crime reports (today)
        - **Size**: Proportional to crime count
        """)

        st.markdown("---")

        # Hotspot details
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔥 All Hotspots")
            hotspot_display = hotspot_df.copy()
            hotspot_display['Status'] = hotspot_display['has_realtime_activity'].apply(
                lambda x: '🔴 ACTIVE' if x else '⚪ Inactive'
            )
            hotspot_display['Coordinates'] = hotspot_display.apply(
                lambda row: f"{row['center_lat']:.4f}, {row['center_lon']:.4f}", axis=1
            )
            st.dataframe(
                hotspot_display[['area_name', 'crime_count', 'dominant_crime', 'Status', 'Coordinates']]
                .sort_values('crime_count', ascending=False),
                use_container_width=True, height=300
            )

        with col2:
            st.subheader("📊 Hotspot Activity")
            active_count = int(hotspot_df[hotspot_df['has_realtime_activity'] == True].shape[0])
            total_count = int(hotspot_df.shape[0])

            activity_data = pd.DataFrame({
                'Status': ['Active (New Crimes)', 'Inactive'],
                'Count': [active_count, total_count - active_count]
            })

            fig_pie = px.pie(activity_data, values='Count', names='Status',
                        color_discrete_sequence=['#ff4757', '#95a5a6'],
                        hole=0.4)
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Heatmap intensity visualization
        st.subheader("🌡️ Crime Density Heatmap")

        crime_intensity = hotspot_df.nlargest(10, 'crime_count')[['area_name', 'crime_count']]
        fig_heat = px.bar(
            crime_intensity,
            x='crime_count',
            y='area_name',
            orientation='h',
            color='crime_count',
            color_continuous_scale='Reds',
            title="Top 10 Crime Density Areas"
        )
        fig_heat.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Crime Count",
            yaxis_title=""
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
    else:
        # Show instructional message
        st.warning("⚠️ No crime data available to display on the map.")
        
        st.info("""
        **To populate the map:**
        1. Ensure the backend API server is running (check connection status at the top)
        2. Navigate to "📝 Report Crime" to add crime incidents
        3. The map will automatically display hotspots and crime locations
        4. Refresh this page after adding crimes
        """)
        
        # Show API status
        status_check = fetch_api_data("health")
        if status_check and status_check.get('status') == 'ok':
            st.success("✅ Backend API is connected and healthy")
        else:
            st.error("❌ Cannot connect to backend API. Please start the FastAPI server on port 8000.")
    
    # Controls below the map
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("🔥 Toggle Hotspots", use_container_width=True):
            st.session_state['show_hotspots'] = not st.session_state.get('show_hotspots', True)
            st.rerun()
    with col3:
        if st.button("🆕 Toggle Live Reports", use_container_width=True):
            st.session_state['show_realtime'] = not st.session_state.get('show_realtime', True)
            st.rerun()

# ============================================
# PAGE: ANALYTICS
# ============================================
elif page == "📊 Analytics":
    st.header("📊 Advanced Analytics - Real-time")
    
    crime_stats = fetch_api_data("crime-stats")
    realtime_stats = fetch_api_data("realtime-stats")
    
    # Real-time stats banner
    if realtime_stats and realtime_stats.get('total_new_crimes', 0) > 0:
        st.warning(f"🔴 {realtime_stats['total_new_crimes']} new crimes reported | {realtime_stats['areas_affected']} areas affected")
    
    if crime_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Crime Type Distribution (Historical)")
            crime_types_dict = crime_stats.get('historical', {}).get('crime_types', {})
            if crime_types_dict:
                crime_types_df = pd.DataFrame(
                    list(crime_types_dict.items()),
                    columns=['Crime Type', 'Count']
                ).sort_values('Count', ascending=False)
                
                fig = px.bar(crime_types_df, x='Crime Type', y='Count',
                            color='Count', color_continuous_scale='Blues')
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical crime type data available")
        
        with col2:
            st.subheader("🆕 New Crime Types (Today)")
            if realtime_stats and realtime_stats.get('crime_types'):
                new_crimes_df = pd.DataFrame(
                    list(realtime_stats['crime_types'].items()),
                    columns=['Crime Type', 'Count']
                )
                
                fig = px.bar(new_crimes_df, x='Crime Type', y='Count',
                            color='Count', color_continuous_scale='Reds')
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No new crimes reported yet")
        
        # Top affected areas comparison
        st.subheader("🎯 Most Affected Areas Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Historical Top 5**")
            top_areas_dict = crime_stats.get('historical', {}).get('top_crime_areas', {})
            if top_areas_dict:
                hist_areas = pd.DataFrame(
                    list(top_areas_dict.items()),
                    columns=['Area', 'Crimes']
                ).head(5)
                st.dataframe(hist_areas, use_container_width=True, hide_index=True)
            else:
                st.info("No historical area data")
        
        with col2:
            st.markdown("**Today's Top Areas**")
            if realtime_stats and realtime_stats.get('top_affected_areas'):
                today_areas = pd.DataFrame(
                    list(realtime_stats['top_affected_areas'].items()),
                    columns=['Area', 'New Crimes']
                )
                st.dataframe(today_areas, use_container_width=True, hide_index=True)
            else:
                st.info("No new crime data")

# ============================================
# PAGE: RISK PREDICTIONS
# ============================================
elif page == "⚠️ Risk Predictions":
    st.header("⚠️ Crime Risk Predictions - Dynamic")
    
    # Fetch crime data for risk analysis
    crimes_data = fetch_api_data("crimes")
    
    if crimes_data and len(crimes_data) > 0:
        try:
            # Convert to DataFrame
            crime_records = []
            for c in crimes_data:
                try:
                    timestamp = c.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    elif not isinstance(timestamp, datetime):
                        timestamp = datetime.now()
                    
                    crime_records.append({
                        'cluster': c.get('cluster', 0),
                        'crime_type': c.get('crime_type'),
                        'timestamp': timestamp,
                        'hour': timestamp.hour
                    })
                except Exception as e:
                    print(f"Error processing crime record: {e}")
                    continue
            
            if crime_records:
                df = pd.DataFrame(crime_records)
                
                # Calculate risk scores
                risk_scores = build_cluster_hour_risk(df)
                
                if not risk_scores.empty:
                    # Show hourly risk heatmap
                    st.subheader("🕒 Hourly Risk Scores by Cluster")
                    fig = px.imshow(
                        risk_scores,
                        labels=dict(x="Hour of Day", y="Cluster", color="Risk Score"),
                        title="Risk Score Heatmap (0-100)",
                        color_continuous_scale="RdYlBu_r"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top risk windows
                    st.subheader("⚠️ Top Risk Windows")
                    top_risks = top_risk_windows(risk_scores)
                    
                    if not top_risks.empty:
                        for _, row in top_risks.head(10).iterrows():
                            st.markdown(f"""
                            <div class="alert-box">
                                <strong>Cluster {int(row['cluster'])}</strong><br>
                                Time: {row['time_window']}<br>
                                Risk Score: {row['risk_score']:.1f}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("Not enough data to calculate risk scores")
            else:
                st.info("No valid crime records to analyze")
        except Exception as e:
            print(f"Error in risk analysis: {e}")
            st.error("Error calculating risk scores. Please check the data.")
    
    # Regular risk predictions
    risk_data = fetch_api_data("risk-predictions")
    if risk_data and risk_data.get('risk_predictions'):
        st.info(f"📊 Monitoring {risk_data.get('total_areas', 0)} areas | 🔴 {risk_data.get('realtime_updated_areas', 0)} with recent updates")
        st.subheader("🚨 Top 10 High-Risk Areas")

        risk_list = risk_data['risk_predictions']
        risk_df = pd.DataFrame(risk_list)
        risk_df = risk_df.sort_values('risk_score', ascending=False).head(10)

        for idx, area in risk_df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])

            risk_category = str(area.get('risk_category', 'Medium'))
            risk_class = {
                'Very High': 'risk-very-high',
                'High': 'risk-high',
                'Medium': 'risk-medium',
                'Low': 'risk-low'
            }.get(risk_category, 'risk-medium')

            with col1:
                has_update = area.get('has_realtime_update', False)
                realtime_badge = '<span class="realtime-badge">LIVE UPDATE</span>' if has_update else ''
                st.markdown(f"### {area.get('area_name', 'Unknown')} {realtime_badge}", unsafe_allow_html=True)

            with col2:
                risk_score = float(area.get('risk_score', 0))
                st.markdown(f"<div class=\"metric-card {risk_class}\" style=\"padding:0.8rem;\">{risk_score:.1f}/100</div>", unsafe_allow_html=True)

            with col3:
                st.metric("Category", risk_category)

            with col4:
                predicted = int(area.get('predicted_crimes', 0))
                st.metric("Predicted", f"{predicted}")

            with col5:
                stations = int(area.get('police_stations', 0))
                st.metric("Stations", stations)

            st.markdown("---")

        # Risk distribution visualization
        st.subheader("📊 Risk Distribution Analysis")
        col1, col2 = st.columns(2)

        with col1:
            all_risk_df = pd.DataFrame(risk_list)
            if not all_risk_df.empty and 'risk_category' in all_risk_df.columns:
                risk_counts = all_risk_df['risk_category'].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                            title="Risk Category Distribution",
                            color_discrete_map={
                                'Very High': '#ff0844',
                                'High': '#f5576c',
                                'Medium': '#4facfe',
                                'Low': '#43e97b'
                            })
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No risk prediction categories available")

        with col2:
            if not all_risk_df.empty and {'population','risk_score'}.issubset(all_risk_df.columns):
                fig = px.scatter(all_risk_df, x='population', y='risk_score',
                               size='predicted_crimes', color='risk_category',
                               hover_name='area_name',
                               title="Risk vs Population",
                               color_discrete_map={
                                   'Very High': '#ff0844',
                                   'High': '#f5576c',
                                   'Medium': '#4facfe',
                                   'Low': '#43e97b'
                               })
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for Risk vs Population plot")

# ============================================
# PAGE: LIVE PATROL
# ============================================
elif page == "🚨 Live Patrol":
    st.header("🚨 Live Patrol Recommendations")
    
    patrol_data = fetch_api_data("patrol-recommendations")
    
    if patrol_data:
        # Current time info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            current_period = patrol_data.get('current_time_period', 'unknown').title()
            current_time = datetime.now().strftime('%H:%M:%S')
            st.info(f"🕐 Current Period: **{current_period}** | {current_time}")
        with col2:
            suggestions = patrol_data.get('patrol_suggestions', [])
            immediate_alerts = sum(1 for s in suggestions if s.get('recommended_time') == 'IMMEDIATE')
            st.metric("🚨 Immediate", immediate_alerts)
        with col3:
            total_suggestions = len(suggestions)
            st.metric("📍 Total", total_suggestions)
        
        st.markdown("---")
        
        # Patrol suggestions
        st.subheader("🚔 Recommended Patrol Routes")
        
        for suggestion in suggestions:
            priority = suggestion.get('priority', 999)
            rec_time = suggestion.get('recommended_time', 'SCHEDULE')
            card_class = "immediate-priority" if rec_time == 'IMMEDIATE' else \
                        "high-priority" if priority <= 3 else ""
            
            sug_type = suggestion.get('type', 'patrol').replace('_', ' ').title()
            area = suggestion.get('area', 'Unknown Area')
            reason = suggestion.get('reason', 'Crime activity detected')
            
            st.markdown(f"""
            <div class="patrol-card {card_class}">
                <h4>Priority #{priority} - {sug_type}</h4>
                <p><strong>📍 Area:</strong> {area}</p>
                <p><strong>📋 Reason:</strong> {reason}</p>
                <p><strong>⏰ Action:</strong> {rec_time}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'coordinates' in suggestion:
                coords = suggestion['coordinates']
                st.caption(f"📌 Coordinates: {coords.get('lat', 0):.4f}, {coords.get('lon', 0):.4f}")
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Priority hotspots map
        priority_hotspots = patrol_data.get('priority_hotspots', [])
        if priority_hotspots:
            st.subheader("🗺️ Priority Hotspot Locations - Patrol Routes")
            
            hotspot_df = pd.DataFrame(priority_hotspots)
            
            # Create patrol route map
            fig = go.Figure()
            
            # Add numbered markers for each priority location
            for priority_num, (idx, row) in enumerate(hotspot_df.iterrows(), start=1):
                marker_color = '#ff0844' if priority_num <= 3 else '#ff6b6b' if priority_num <= 5 else '#ffa502'

                fig.add_trace(go.Scattermapbox(
                    lat=[row['center_lat']],
                    lon=[row['center_lon']],
                    mode='markers+text',
                    marker=dict(
                        size=row['crime_count'] * 2,
                        color=marker_color,
                        opacity=0.8
                    ),
                    text=[f"#{priority_num}"],
                    textposition="middle center",
                    textfont=dict(size=12, color='white', family='Arial Black'),
                    hovertemplate=f'<b>Priority #{priority_num}</b><br>' +
                                  f'Area: {row["area_name"]}<br>' +
                                  f'Crimes: {row["crime_count"]}<br>' +
                                  f'<extra></extra>',
                    name=f'Priority {priority_num}',
                    showlegend=False
                ))
            
            # Add patrol route lines
            if len(hotspot_df) > 1:
                fig.add_trace(go.Scattermapbox(
                    lat=hotspot_df['center_lat'].tolist(),
                    lon=hotspot_df['center_lon'].tolist(),
                    mode='lines',
                    line=dict(width=3, color='blue'),
                    opacity=0.5,
                    hoverinfo='skip',
                    name='Suggested Route',
                    showlegend=True
                ))
            
            # Calculate center
            center_lat = hotspot_df['center_lat'].mean()
            center_lon = hotspot_df['center_lon'].mean()
            
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=11
                ),
                height=500,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Route summary
            st.info(f"📍 **Patrol Route**: {len(hotspot_df)} priority locations | 🔴 High Priority: First 3 stops")

# ============================================
# PAGE: AREA DETAILS
# ============================================
elif page == "📍 Area Details":
    st.header("📍 Area-wise Crime Analysis")
    
    # Area selection
    area_name = st.text_input("🔍 Enter Area Name", placeholder="e.g., Connaught Place, Rohini")
    
    if st.button("🔎 Search Area", use_container_width=True):
        if area_name:
            with st.spinner(f"Fetching details for {area_name}..."):
                area_data = fetch_api_data(f"area/{area_name}")
                
                if area_data and 'error' not in area_data:
                    st.success(f"✅ Found data for **{area_data.get('area_name', area_name)}**")
                    
                    # Real-time status banner
                    realtime_data = area_data.get('realtime_data', {})
                    realtime_status = realtime_data.get('status', 'Inactive')
                    new_crimes = realtime_data.get('new_crimes_today', 0)
                    
                    if realtime_status == 'Active':
                        st.warning(f"🔴 **ACTIVE AREA** - {new_crimes} new crimes reported today")
                    else:
                        st.info(f"🟢 Status: {realtime_status}")
                    
                    st.markdown("---")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        hist_crimes = area_data.get('historical_data', {}).get('total_crimes', 0)
                        st.metric("Historical Crimes", hist_crimes)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("New Today", new_crimes)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        pop_info = area_data.get('population_info', {})
                        if pop_info:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Population", f"{pop_info.get('population', 0):,}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        risk_pred = area_data.get('risk_prediction', {})
                        risk_cat = str(risk_pred.get('risk_category', 'N/A'))
                        risk_class = {
                            'Very High': 'risk-very-high',
                            'High': 'risk-high',
                            'Medium': 'risk-medium',
                            'Low': 'risk-low'
                        }.get(risk_cat, 'risk-medium')

                        st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
                        st.metric("Risk Level", risk_cat)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Detailed analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Historical Crime Analysis")
                        
                        # Crime types
                        hist_data = area_data.get('historical_data', {})
                        crime_types = hist_data.get('crime_types', {})
                        if crime_types:
                            st.markdown("**Crime Type Distribution:**")
                            crime_types_df = pd.DataFrame(
                                list(crime_types.items()),
                                columns=['Crime Type', 'Count']
                            ).sort_values('Count', ascending=False)
                            
                            fig = px.bar(crime_types_df, x='Crime Type', y='Count',
                                        color='Count', color_continuous_scale='Blues')
                            fig.update_layout(showlegend=False, height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Time period distribution
                        time_periods = hist_data.get('time_period_crimes', {})
                        if time_periods:
                            st.markdown("**Crime by Time Period:**")
                            time_df = pd.DataFrame(
                                list(time_periods.items()),
                                columns=['Period', 'Count']
                            )
                            
                            fig = px.pie(time_df, values='Count', names='Period',
                                        color_discrete_sequence=px.colors.sequential.RdBu)
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("🔴 Real-time Activity")
                        
                        # New crime types today
                        new_crime_types = realtime_data.get('new_crime_types', {})
                        if new_crime_types:
                            st.markdown("**New Crimes Today (By Type):**")
                            new_crimes_df = pd.DataFrame(
                                list(new_crime_types.items()),
                                columns=['Crime Type', 'Count']
                            )
                            
                            fig = px.bar(new_crimes_df, x='Crime Type', y='Count',
                                        color='Count', color_continuous_scale='Reds')
                            fig.update_layout(showlegend=False, height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No new crimes reported today")
                        
                        # Last crime details
                        last_crime = realtime_data.get('last_crime')
                        if last_crime:
                            st.markdown("**Most Recent Crime:**")
                            try:
                                crime_time = datetime.fromisoformat(last_crime['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                crime_time = last_crime.get('timestamp', 'Unknown')
                                
                            st.markdown(f"""
                            <div class="alert-box">
                                <p><strong>Type:</strong> {last_crime.get('crime_type', 'Unknown')}</p>
                                <p><strong>Time:</strong> {crime_time}</p>
                                <p><strong>ID:</strong> {last_crime.get('id', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Population and infrastructure info
                    if pop_info:
                        st.markdown("---")
                        st.subheader("🏛️ Area Infrastructure")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Population", f"{pop_info.get('population', 0):,}")
                        with col2:
                            st.metric("Area (sq km)", f"{pop_info.get('area_sq_km', 0):.2f}")
                        with col3:
                            st.metric("Police Stations", pop_info.get('police_stations', 0))
                        with col4:
                            st.metric("Response Time", f"{pop_info.get('avg_response_time', 0)} min")
                    
                    # Risk prediction details
                    if risk_pred:
                        st.markdown("---")
                        st.subheader("⚠️ AI Risk Assessment")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            risk_score = float(risk_pred.get('risk_score', 0))
                            st.metric("Risk Score", f"{risk_score:.1f}/100")
                        with col2:
                            pred_crimes = int(risk_pred.get('predicted_crimes', 0))
                            st.metric("Predicted Crimes", f"{pred_crimes}")
                        with col3:
                            crime_rate = float(risk_pred.get('crime_rate_per_1000', 0))
                            st.metric("Crime Rate", f"{crime_rate:.2f}/1000")
                        
                        # Risk gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=risk_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Risk Score"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, 25], 'color': "#43e97b"},
                                    {'range': [25, 50], 'color': "#4facfe"},
                                    {'range': [50, 75], 'color': "#f5576c"},
                                    {'range': [75, 100], 'color': "#ff0844"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 75
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"Last updated: {area_data.get('last_update', 'Unknown')}")
                    
                else:
                    st.error(f"❌ No data found for area: {area_name}")
        else:
            st.warning("⚠️ Please enter an area name")
    
    # Show example areas
    st.markdown("---")
    st.subheader("💡 Example Areas")
    st.info("Try searching: Connaught Place, Rohini, Dwarka, Saket, Karol Bagh")

# ============================================
# AUTO-REFRESH MECHANISM
# ============================================
if auto_refresh:
    time.sleep(30)
    st.rerun()

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>🚓 Smart Patrol AI</strong> - Real-time Crime Prediction & Patrol Optimization System</p>
    <p>Powered by Machine Learning | Last Update: {}</p>
    <p style='font-size: 0.9rem;'>⚠️ For demonstration purposes only</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
