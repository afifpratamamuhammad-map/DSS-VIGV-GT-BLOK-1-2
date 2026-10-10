import math
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import requests
import streamlit as st

# ==============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT & MODERN CLEAN UI STYLING
# ==============================================================================
st.set_page_config(
    page_title="DECISION SUPPORT SYSTEM GT BLOK 1-2 UBP PRIOK",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #F8FAFC; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    
    /* Breadcrumb Header Navigasi */
    .breadcrumb-nav { font-size: 13px; color: #64748B; font-weight: 500; margin-bottom: 2px; }
    .breadcrumb-nav span { color: #2563EB; font-weight: 600; }
    .app-title { font-size: 24px; font-weight: 800; color: #0F172A; margin-bottom: 18px; letter-spacing: -0.5px; }
    
    /* Clean UI Custom Cards */
    .card-safe { background-color: #F0FDF4; border-left: 6px solid #16A34A; padding: 16px; border-radius: 8px; margin-bottom: 12px; }
    .card-warn { background-color: #FEFCE8; border-left: 6px solid #CA8A04; padding: 16px; border-radius: 8px; margin-bottom: 12px; }
    .card-danger { background-color: #FEF2F2; border-left: 6px solid #DC2626; padding: 16px; border-radius: 8px; margin-bottom: 12px; }
    .weather-card { background-color: #EFF6FF; border: 1px solid #BFDBFE; padding: 16px; border-radius: 8px; margin-bottom: 16px; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    </style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. LOAD MODEL PROPHET & FUNGSI PREDIKSI AI WEATHER
# ==============================================================================
@st.cache_resource
def load_prophet_weather_models():
  try:
    m_temp = joblib.load('model_temp_prophet.pkl')
    m_rh = joblib.load('model_rh_prophet.pkl')
    return m_temp, m_rh
  except Exception:
    return None, None


m_temp_prophet, m_rh_prophet = load_prophet_weather_models()


def predict_weather_24h_prophet(m_temp, m_rh, anchor_temp, anchor_rh):
  if m_temp is None or m_rh is None:
    return None

  now = pd.Timestamp.now().floor('h')
  future_dates = pd.date_range(start=now, periods=24, freq='h')

  base_dates = pd.date_range(
      start='2023-06-01 00:00:00', periods=24, freq='h'
  )
  df_base = pd.DataFrame({'ds': base_dates})

  pred_temp_base = m_temp.predict(df_base)
  pred_rh_base = m_rh.predict(df_base)

  temp_daily_pattern = (
      pred_temp_base['weekly'] + pred_temp_base['daily']
      if 'weekly' in pred_temp_base
      else pred_temp_base['daily']
  )
  rh_daily_pattern = (
      pred_rh_base['weekly'] + pred_rh_base['daily']
      if 'weekly' in pred_rh_base
      else pred_rh_base['daily']
  )

  temp_vals = []
  rh_vals = []

  for dt in future_dates:
    hour_idx = dt.hour
    dev_temp = temp_daily_pattern.iloc[hour_idx] - temp_daily_pattern.mean()
    dev_rh = rh_daily_pattern.iloc[hour_idx] - rh_daily_pattern.mean()

    calc_t = round(float(anchor_temp) + dev_temp, 1)
    calc_rh = round(np.clip(float(anchor_rh) + dev_rh, 30.0, 99.0), 0)

    temp_vals.append(calc_t)
    rh_vals.append(calc_rh)

  df_forecast = pd.DataFrame({
      'Waktu': future_dates,
      'Prediksi_Temp_°C': temp_vals,
      'Prediksi_RH_%': rh_vals,
  })

  return df_forecast


# ==============================================================================
# 3. FUNGSI PERHITUNGAN FISIKA UDARA BASAH & VIGV TRIM
# ==============================================================================
def calculate_moist_air_density(temp_c, rh_pct, press_mbar):
  """Kalkulasi Kerapatan Udara Basah (Moist Air Density - kg/m³)"""
  if temp_c is None or rh_pct is None or press_mbar is None:
    return 1.165, 15.0

  p_sat = 6.1078 * math.pow(10, (7.5 * temp_c) / (237.3 + temp_c))
  p_v = (rh_pct / 100.0) * p_sat
  p_d = press_mbar - p_v

  t_k = temp_c + 273.15
  r_d = 287.058
  r_v = 461.495

  rho_d = (p_d * 100.0) / (r_d * t_k)
  rho_v = (p_v * 100.0) / (r_v * t_k)

  rho_moist = rho_d + rho_v
  return round(rho_moist, 4), round(p_v, 2)


def calculate_vigv_trim(
    t_intake, load_mw, act_vigv=62.7, rho_moist=1.165, mode_simulasi=False
):
  """Kalkulasi VIGV Trim Otomatis 2-Arah Berbasis Koreksi Proporsional Dinamis."""
  if load_mw <= 0:
    return 0.0, 0.0, 0.0, 0.0, 'UNIT OFFLINE'

  if mode_simulasi:
    base_trim_temp = (t_intake - 20.0) * 0.1
    rho_iso = 1.2041
    density_corr = (
        max((rho_iso - rho_moist) / rho_iso, 0.0) * 0.8
        if rho_moist < rho_iso
        else 0.0
    )
    trim_target = float(np.clip(base_trim_temp + density_corr, 0.2, 2.0))
    status_msg = 'NORMAL ADVISORY'
  else:
    base_ideal_angle = 45.0 + (load_mw / 120.0) * 20.0
    rho_iso = 1.2041
    density_corr = max((rho_iso - rho_moist) / rho_iso, 0.0) * 0.8
    temp_corr = (t_intake - 20.0) * 0.1

    target_angle_ideal = base_ideal_angle + temp_corr + density_corr
    delta_angle = target_angle_ideal - act_vigv
    trim_calc = delta_angle * 0.6
    trim_target = float(np.clip(trim_calc, -2.0, 2.0))

    if trim_target > 0.1:
      status_msg = 'UNDER-OPENED (Tercekik) ➔ Rekomendasi Trim NAIK (+)'
    elif trim_target < -0.1:
      status_msg = 'OVER-OPENED (Kelebihan) ➔ Rekomendasi Trim TURUN (-)'
    else:
      status_msg = 'OPTIMAL ➔ Bukaan Sudut Sudah Pas'

  fuel_saved_h = base_saving_rate * (abs(trim_target) / 2.0)
  fin_saved_h = fuel_saved_h * price_per_unit
  co2_red_h = fuel_saved_h * co2_factor

  return (
      round(trim_target, 2),
      round(fuel_saved_h, 3),
      round(fin_saved_h, 0),
      round(co2_red_h, 1),
      status_msg,
  )


def predict_intake_temp(t_ambient_base, hour_slot):
  if '10:00' in hour_slot:
    return round(t_ambient_base + 1.8, 1)
  elif '17:00' in hour_slot:
    return round(t_ambient_base + 1.2, 1)
  elif '00:00' in hour_slot:
    return round(t_ambient_base + 0.3, 1)
  return round(t_ambient_base + 1.0, 1)


# ==============================================================================
# 4. INTEGRASI API CUACA OPEN-METEO
# ==============================================================================
LATITUDE = -6.1102
LONGITUDE = 106.8671


@st.cache_data(ttl=300)
def fetch_priok_weather_api():
  """Mengambil data cuaca real-time & prediksi dari Open-Meteo API khusus lokasi PLTGU Priok."""
  try:
    url = (
        f'https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}'
        '&current=temperature_2m,relative_humidity_2m,surface_pressure'
        '&hourly=temperature_2m,relative_humidity_2m,surface_pressure'
        '&timezone=Asia%2FJakarta'
    )
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers, timeout=5)

    if response.status_code == 200:
      data = response.json()
      if 'current' in data:
        return data
  except Exception:
    return None
  return None


weather_json = fetch_priok_weather_api()
api_available = weather_json is not None

if api_available:
  curr_weather = weather_json.get('current', {})
  live_temp = curr_weather.get('temperature_2m', 31.5)
  live_rh = curr_weather.get('relative_humidity_2m', 78.0)
  live_press = curr_weather.get('surface_pressure') or 1011.0
else:
  live_temp, live_rh, live_press = 31.5, 78.0, 1011.0

# ==============================================================================
# 5. SIDEBAR - NAVIGASI CLEAN UI & CONFIG
# ==============================================================================
st.sidebar.markdown('### 🏢 Unit Selection')
selected_unit = st.sidebar.selectbox(
    'Pilih Unit Turbin Gas:',
    ['GT 1.1', 'GT 1.2', 'GT 1.3', 'GT 2.1', 'GT 2.2', 'GT 2.3'],
)

active_page = st.sidebar.radio(
    '📍 Navigasi Modul:',
    [
        '📊 Live Weather & Prediksi Shift',
        '🎛️ Live Verification & DCS Interlock',
        '🔮 What-If Analysis & Dynamic Simulation',
        '🛡️ Manfaat & Prediksi Filter RUL',
    ],
)

st.sidebar.markdown('---')
st.sidebar.markdown('### ⛽ Parameter Pembangkit')
fuel_mode = st.sidebar.radio(
    'Mode Bahan Bakar :',
    ('Gas Alam (Natural Gas)', 'HSD (High Speed Diesel / Solar)'),
)

st.sidebar.markdown('---')
st.sidebar.markdown('### 🌐 Telemetri Stasiun Cuaca')
use_live_api = st.sidebar.toggle(
    'Gunakan Live API Cuaca Tanjung Priok', value=api_available
)

if use_live_api and api_available:
  st.sidebar.success(f'🟢 API Terhubung (Lat: {LATITUDE}, Lon: {LONGITUDE})')
  input_temp_amb = live_temp
  input_rh_amb = live_rh
  input_pamb_mbar = live_press
else:
  if not api_available and use_live_api:
    st.sidebar.warning('⚠️ Gagal koneksi API. Menggunakan Mode Manual.')
  input_temp_amb = st.sidebar.number_input(
      'Temp Ambeien Manual (°C):', value=31.5, step=0.1
  )
  input_rh_amb = st.sidebar.number_input(
      'Kelembapan Relative RH Manual (%):', value=78.0, step=1.0
  )
  input_pamb_mbar = st.sidebar.number_input(
      'Ambient Press Manual (mbar):', value=1011.0, step=1.0
  )

weather_condition = st.sidebar.selectbox(
    'Kondisi Cuaca Lapangan:',
    ['Cerah / Normal', 'Mendung / Gerimis', 'Hujan Deras (Extreme Drop)'],
)

if fuel_mode == 'Gas Alam (Natural Gas)':
  lhv_fuel = 8800.0 * 1000.0
  co2_factor = 1.98 * 1000.0
  price_per_unit = 4025.0 * 1000.0
  fuel_unit = 'kscm/h'
  fuel_short = 'Gas Alam'
  base_saving_rate = 0.247
  default_fuel_cons = 35.30
else:
  lhv_fuel = 8600.0 * 1000.0
  co2_factor = 2.68 * 1000.0
  price_per_unit = 13500.0 * 1000.0
  fuel_unit = 'cbm/h'
  fuel_short = 'HSD (Solar)'
  base_saving_rate = 0.220
  default_fuel_cons = 31.20

moist_density, vapor_press = calculate_moist_air_density(
    input_temp_amb, input_rh_amb, input_pamb_mbar
)

# RENDER HEADER BREADCRUMB
page_name_clean = active_page.split(' ', 1)[1]
st.markdown(
    f'<div class="breadcrumb-nav">PLTGU Priok / {selected_unit} /'
    f' <span>{page_name_clean}</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="app-title">DSS OPTIMIZATION — {selected_unit}</div>',
    unsafe_allow_html=True,
)


# ==============================================================================
# 6. FUNGSI GENERATOR GRAFIK PLOTLY INTERAKTIF
# ==============================================================================
def create_vigv_trend_chart(
    act_vigv, target_vigv, dcs_tat, pred_tat, dcs_tit, pred_tit
):
  fig = make_subplots(
      rows=1,
      cols=2,
      subplot_titles=(
          'Perbandingan Bukaan VIGV Angle (°)',
          'Perubahan Suhu TAT & TIT (°C)',
      ),
      horizontal_spacing=0.15,
  )

  fig.add_trace(
      go.Bar(
          x=['Actual VIGV (DCS)', 'Target VIGV (DSS)'],
          y=[act_vigv, target_vigv],
          text=[f'{act_vigv:.1f}°', f'{target_vigv:.2f}°'],
          textposition='auto',
          marker_color=[
              '#64748B',
              '#1E3A8A' if target_vigv >= act_vigv else '#D97706',
          ],
          name='VIGV Angle',
      ),
      row=1,
      col=1,
  )

  fig.add_trace(
      go.Bar(
          x=['TAT (Exhaust)', 'TIT (Inlet)'],
          y=[dcs_tat, dcs_tit],
          text=[f'{dcs_tat:.1f}°C', f'{dcs_tit:.1f}°C'],
          textposition='auto',
          marker_color='#EF4444',
          name='Before Trim',
      ),
      row=1,
      col=2,
  )

  fig.add_trace(
      go.Bar(
          x=['TAT (Exhaust)', 'TIT (Inlet)'],
          y=[pred_tat, pred_tit],
          text=[f'{pred_tat:.1f}°C', f'{pred_tit:.1f}°C'],
          textposition='auto',
          marker_color='#10B981',
          name='After Trim',
      ),
      row=1,
      col=2,
  )

  fig.update_layout(
      height=380,
      showlegend=True,
      barmode='group',
      margin=dict(l=20, r=20, t=40, b=20),
      template='plotly_white',
  )
  fig.update_yaxes(title_text='Sudut (°)', row=1, col=1)
  fig.update_yaxes(title_text='Temperatur (°C)', row=1, col=2)

  return fig


def create_performance_map(load_mw, pcd_bar, target_vigv):
  fig = go.Figure()

  loads = np.linspace(40, 150, 50)
  pcd_low = (loads / 150.0) * 11.5 + 2.0
  pcd_opt = (loads / 150.0) * 13.0 + 1.8
  pcd_high = (loads / 150.0) * 14.2 + 1.5

  fig.add_trace(
      go.Scatter(
          x=np.concatenate([loads, loads[::-1]]),
          y=np.concatenate([pcd_high, pcd_low[::-1]]),
          fill='toself',
          fillcolor='rgba(219, 234, 254, 0.5)',
          line=dict(color='rgba(255,255,255,0)'),
          hoverinfo='skip',
          name='Operating Envelope',
      )
  )

  fig.add_trace(
      go.Scatter(
          x=loads,
          y=pcd_opt,
          mode='lines',
          line=dict(color='#2563EB', width=2, dash='dash'),
          name='Optimum VIGV Line',
      )
  )

  fig.add_trace(
      go.Scatter(
          x=[load_mw],
          y=[pcd_bar],
          mode='markers+text',
          marker=dict(
              size=16,
              color='#DC2626',
              symbol='cross-dot',
              line=dict(width=2, color='black'),
          ),
          text=[
              f'  Current Operating Point ({load_mw:.1f} MW, {pcd_bar:.1f}'
              ' bar)'
          ],
          textposition='top right',
          name='Aktual Unit GT',
      )
  )

  fig.update_layout(
      title='Peta Kinerja & Operating Envelope GT13E1 (Load vs Pcd)',
      xaxis_title='Beban Generator (MW)',
      yaxis_title='Pressure Discharge Compressor / Pcd (bar)',
      height=400,
      margin=dict(l=20, r=20, t=50, b=20),
      template='plotly_white',
      legend=dict(
          orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1
      ),
  )

  return fig


# ==============================================================================
# 7. ROUTING HALAMAN DENGAN FITUR LENGKAP
# ==============================================================================

# ------------------------------------------------------------------------------
# HALAMAN 1: LIVE WEATHER & PREDIKSI SHIFT
# ------------------------------------------------------------------------------
if active_page == '📊 Live Weather & Prediksi Shift':
  st.subheader('🌐 Monitoring Live Telemetri Cuaca & Prediksi VIGV Trim')

  st.markdown('<div class="weather-card">', unsafe_allow_html=True)
  wc1, wc2, wc3, wc4 = st.columns(4)
  wc1.metric(
      'Suhu Ambeien (Tamb)',
      f'{input_temp_amb:.1f} °C',
      delta='Live API Open-Meteo' if use_live_api else 'Manual',
  )
  wc2.metric(
      'Kelembapan Relatif (RH)',
      f'{input_rh_amb:.0f} %',
      delta=f'Uap Air: {vapor_press} hPa',
  )
  wc3.metric('Tekanan Udara (Pamb)', f'{input_pamb_mbar:.1f} mbar')

  iso_density = 1.2041
  density_diff_pct = ((moist_density - iso_density) / iso_density) * 100.0
  wc4.metric(
      'Kerapatan Udara Basah (ρ)',
      f'{moist_density:.4f} kg/m³',
      delta=f'{density_diff_pct:.2f}% vs ISO Standard',
  )
  st.markdown('</div>', unsafe_allow_html=True)

  st.markdown('---')
  st.subheader('Perencanaan Harian VIGV Trim (3 Jam Transisi Utama Shift)')
  col1, col2 = st.columns(2)
  t_amb_input = col1.number_input(
      'Prakiraan Suhu Ambeien Luar (°C)',
      value=float(input_temp_amb),
      step=0.5,
  )
  load_mw_plan = col2.number_input(
      'Rencana Set Point Beban (MW)', value=104.0, step=1.0
  )

  slots = ['10:00 WIB (Pagi)', '17:00 WIB (Siang)', '00:00 WIB (Malam)']
  cols_slot = st.columns(3)

  for idx, slot_name in enumerate(slots):
    with cols_slot[idx]:
      if weather_condition == 'Hujan Deras (Extreme Drop)':
        t_pred = 24.5
      else:
        base_adj = t_amb_input + (
            1.5 if idx == 0 else (0.5 if idx == 1 else -2.0)
        )
        t_pred = predict_intake_temp(base_adj, slot_name)

      rho_slot, _ = calculate_moist_air_density(
          t_pred, input_rh_amb, input_pamb_mbar
      )
      trim_p, f_sav_p, rp_sav_p, co2_p, _ = calculate_vigv_trim(
          t_pred, load_mw_plan, rho_moist=rho_slot, mode_simulasi=True
      )

      st.markdown(f'### 🕒 Slot {slot_name.split()[0]}')
      st.metric('Prediksi Temp Intake', f'{t_pred} °C')
      st.metric('Rekomendasi VIGV Trim', f'+{trim_p:.2f} °')
      st.caption(
          f'Est. Hemat: {f_sav_p:.3f} {fuel_unit} (Rp {rp_sav_p:,.0f}/jam)'
      )

  if api_available and use_live_api:
    st.markdown('---')
    st.subheader(
        '📊 Tabel Prediksi Cuaca Jam-Jaman PLTGU Priok (6 Jam Ke Depan)'
    )
    hourly_data = weather_json.get('hourly', {})
    df_hourly = pd.DataFrame({
        'Waktu': hourly_data.get('time', [])[:6],
        'Temp_Ambeien_°C': hourly_data.get('temperature_2m', [])[:6],
        'Humidity_RH_%': hourly_data.get('relative_humidity_2m', [])[:6],
        'Pressure_mbar': hourly_data.get('surface_pressure', [])[:6],
    })

    df_hourly['Est_Temp_Intake_°C'] = df_hourly['Temp_Ambeien_°C'] + 1.2

    def get_trim_from_row(row):
      rho_row, _ = calculate_moist_air_density(
          row['Temp_Ambeien_°C'], row['Humidity_RH_%'], row['Pressure_mbar']
      )
      return calculate_vigv_trim(
          row['Est_Temp_Intake_°C'],
          load_mw_plan,
          rho_moist=rho_row,
          mode_simulasi=True,
      )[0]

    df_hourly['Target_VIGV_Trim_°'] = df_hourly.apply(
        get_trim_from_row, axis=1
    )
    st.dataframe(df_hourly, use_container_width=True)

  # AI Prophet Forecasting
  st.markdown('---')
  st.subheader('🤖 Local AI Weather Forecasting (Prophet Model 24 Jam)')

  df_ai_forecast = predict_weather_24h_prophet(
      m_temp_prophet, m_rh_prophet, input_temp_amb, input_rh_amb
  )

  if df_ai_forecast is not None:

    def calc_trim_row_ai(row):
      rho_row, _ = calculate_moist_air_density(
          row['Prediksi_Temp_°C'], row['Prediksi_RH_%'], input_pamb_mbar
      )
      return calculate_vigv_trim(
          row['Prediksi_Temp_°C'] + 1.2,
          load_mw_plan,
          rho_moist=rho_row,
          mode_simulasi=True,
      )[0]

    df_ai_forecast['Target_VIGV_Trim_°'] = df_ai_forecast.apply(
        calc_trim_row_ai, axis=1
    )

    fig_ai = go.Figure()
    fig_ai.add_trace(
        go.Scatter(
            x=df_ai_forecast['Waktu'],
            y=df_ai_forecast['Prediksi_Temp_°C'],
            name='Prediksi Suhu Ambeien (°C)',
            line=dict(color='#EF4444', width=2),
        )
    )
    fig_ai.add_trace(
        go.Scatter(
            x=df_ai_forecast['Waktu'],
            y=df_ai_forecast['Target_VIGV_Trim_°'],
            name='Rekomendasi VIGV Trim (°)',
            line=dict(color='#2563EB', width=2, dash='dash'),
            yaxis='y2',
        )
    )

    fig_ai.update_layout(
        title='Prediksi Fluktuasi Suhu Pesisir & Rekomendasi VIGV Trim (24 Jam)',
        xaxis_title='Waktu (Jam)',
        yaxis=dict(title='Suhu (°C)'),
        yaxis2=dict(
            title='VIGV Trim (°)',
            overlaying='y',
            side='right',
            range=[0, 2.5],
        ),
        height=380,
        template='plotly_white',
    )

    st.plotly_chart(fig_ai, use_container_width=True)

    with st.expander('🔍 Lihat Detail Tabel Prediksi Cuaca 24 Jam ML'):
      st.dataframe(df_ai_forecast, use_container_width=True)
  else:
    st.info(
        '💡 Model `.pkl` lokal belum diunggah ke repositori GitHub. Tampilan saat'
        ' ini menggunakan Live API standar.'
    )

# ------------------------------------------------------------------------------
# HALAMAN 2: LIVE VERIFICATION & DCS SAFETY
# ------------------------------------------------------------------------------
elif active_page == '🎛️ Live Verification & DCS Interlock':
  st.subheader('DCS Inputs & Protective Limit Monitor')

  col_d1, col_d2, col_d3 = st.columns([1.2, 1.2, 1.6])

  with col_d1:
    st.markdown('**1. Parameter Beban & Operasi GT (DCS):**')
    dcs_set_point_load = st.number_input(
        'Set Point Load (MW)', value=104.0, step=0.5
    )
    dcs_act_vigv = st.number_input(
        'Actual VIGV Angle Saat Ini (°)', value=62.7, step=0.1
    )
    dcs_fuel_cons = st.number_input(
        f'Fuel Consumption DCS ({fuel_unit})',
        value=default_fuel_cons,
        step=0.1,
        format='%.2f',
    )
    dcs_tat = st.number_input('Exhaust Temp / TAT (°C)', value=498.0, step=0.1)
    dcs_tit = st.number_input(
        'Turbine Inlet Temp / TIT (°C)', value=1050.0, step=1.0
    )

  with col_d2:
    st.markdown('**2. Parameter Tekanan & Lingkungan DCS:**')
    dcs_temp_intake = st.number_input(
        'Temp Intake DCS (°C)', value=round(input_temp_amb + 1.2, 1), step=0.1
    )
    dcs_pcd = st.number_input(
        'Comp Discharge Press / Pcd (bar)', value=11.2, step=0.1
    )
    dcs_pamb_mbar = st.number_input(
        'Ambient Pressure / Pamb (mbar)',
        value=float(input_pamb_mbar),
        step=1.0,
    )
    dcs_rh_pct = st.number_input(
        'Ambient Humidity / RH (%)', value=float(input_rh_amb), step=1.0
    )

  rho_live_dcs, _ = calculate_moist_air_density(
      dcs_temp_intake, dcs_rh_pct, dcs_pamb_mbar
  )

  trim_live, f_sav_live, rp_sav_live, co2_live, status_msg_live = (
      calculate_vigv_trim(
          dcs_temp_intake,
          dcs_set_point_load,
          act_vigv=dcs_act_vigv,
          rho_moist=rho_live_dcs,
          mode_simulasi=False,
      )
  )
  target_vigv_live = dcs_act_vigv + trim_live

  tat_drop_pred = 2.5 * (trim_live / 2.0)
  tit_drop_pred = 4.0 * (trim_live / 2.0)
  pred_tat_after = dcs_tat - tat_drop_pred
  pred_tit_after = dcs_tit - tit_drop_pred

  with col_d3:
    card_style = 'card-safe' if trim_live >= 0 else 'card-warn'
    st.markdown(f'<div class="{card_style}">', unsafe_allow_html=True)
    st.markdown('**💡 REKOMENDASI INPUT HMI DCS:**')

    trim_sign = '+' if trim_live > 0 else ''
    st.markdown(
        f'<h2 style="color: #1E3A8A; margin:0;">Target Bias:'
        f' {trim_sign}{trim_live:.2f}°</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(f'**Status Evaluasi:** `{status_msg_live}`')
    st.markdown(
        f'**Target Bukaan VIGV Akhir:** `{target_vigv_live:.2f}°` *(Saat Ini:'
        f' {dcs_act_vigv:.1f}°)*'
    )
    st.markdown('---')
    st.markdown('**📈 PREDIKSI PARAMETER PASCA-TRIM (AFTER):**')
    st.markdown(
        f'• **Prediksi TAT After Trim:** `{pred_tat_after:.1f} °C` *(Estimasi'
        f' Perubahan: {-tat_drop_pred:+.1f}°C)*'
    )
    st.markdown(
        f'• **Prediksi TIT After Trim:** `{pred_tit_after:.1f} °C` *(Estimasi'
        f' Perubahan: {-tit_drop_pred:+.1f}°C)*'
    )
    st.markdown('</div>', unsafe_allow_html=True)

  st.markdown('---')
  st.subheader('Batas Keamanan Gas Turbine Real-Time')

  e_col1, e_col2, e_col3, e_col4 = st.columns(4)

  with e_col1:
    st.markdown('**Status VIGV Angle**')
    if target_vigv_live < 45.0:
      st.error(f'🚨 **TRIP CRITICAL!** VIGV < 45° ({target_vigv_live:.1f}°)')
    elif dcs_set_point_load <= 70.0 and (47.0 <= target_vigv_live <= 48.0):
      st.info(f'ℹ️ Standby Mode (0-70 MW): VIGV {target_vigv_live:.1f}°')
    elif dcs_set_point_load > 120.0 and target_vigv_live >= 72.0:
      st.warning('⚠️ Base Load (>120 MW): Max Limit 72° REACHED')
    else:
      st.success(f'✅ Normal Angle: {target_vigv_live:.1f}° (Safe)')

  with e_col2:
    st.markdown('**Status TAT Exhaust**')
    if dcs_tat >= 575.0:
      st.error(f'🚨 **TRIP TAT!** ({dcs_tat:.1f}°C ≥ 575°C)')
    elif dcs_tat >= 565.0:
      st.warning(f'⚠️ **ALARM TAT!** ({dcs_tat:.1f}°C ≥ 565°C)')
    else:
      st.success(f'✅ Safe Margin: +{575.0 - pred_tat_after:.1f}°C to Trip')

  with e_col3:
    st.markdown('**Status TIT Inlet**')
    if dcs_tit >= 1130.0:
      st.error(f'🚨 **TRIP TIT!** ({dcs_tit:.1f}°C ≥ 1130°C)')
    elif dcs_tit >= 1120.0:
      st.warning(f'⚠️ **ALARM TIT!** ({dcs_tit:.1f}°C ≥ 1120°C)')
    else:
      st.success(f'✅ Safe Margin: +{1130.0 - pred_tit_after:.1f}°C to Trip')

  with e_col4:
    st.markdown('**Status Press Discharge Compressor**')
    if dcs_pcd >= 13.5:
      st.warning(f'⚠️ **ALARM SURGE Pcd!** ({dcs_pcd:.1f} bar ≥ 13.5 bar)')
    else:
      st.success(f'✅ Pcd Normal ({dcs_pcd:.1f} bar)')

  st.markdown('---')
  st.subheader('📊 Visualisasi Trend & Peta Efisiensi GT13E1 Real-Time')

  chart_tab1, chart_tab2 = st.tabs([
      '📈 Trend VIGV & Respon Suhu',
      '🗺️ Peta Efisiensi Operasi (Performance Map)',
  ])

  with chart_tab1:
    fig_trend = create_vigv_trend_chart(
        dcs_act_vigv,
        target_vigv_live,
        dcs_tat,
        pred_tat_after,
        dcs_tit,
        pred_tit_after,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

  with chart_tab2:
    fig_map = create_performance_map(
        dcs_set_point_load, dcs_pcd, target_vigv_live
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ------------------------------------------------------------------------------
# HALAMAN 3: WHAT-IF ANALYSIS & DYNAMIC SIMULATION
# ------------------------------------------------------------------------------
elif active_page == '🔮 What-If Analysis & Dynamic Simulation':
  st.subheader('🔮 Visual What-If Analysis & Dynamic Simulation')
  st.caption(
      'Geser slider di bawah ini untuk mensimulasikan dampak fluktuasi beban dan'
      ' suhu ambeien secara dinamis.'
  )

  sim_col1, sim_col2 = st.columns(2)
  sim_mw = sim_col1.slider('Simulasi Beban Generator (MW)', 40.0, 140.0, 104.0)
  sim_temp = sim_col2.slider(
      'Simulasi Perubahan Suhu Intake (°C)', 20.0, 40.0, 32.5
  )

  sim_rho, _ = calculate_moist_air_density(sim_temp, 75.0, 1011.0)
  sim_trim, sim_fsav, sim_rpsav, sim_co2, _ = calculate_vigv_trim(
      sim_temp, sim_mw, rho_moist=sim_rho, mode_simulasi=True
  )

  sim_pcd = (sim_mw / 150.0) * 13.0 + 1.8 + (sim_trim * 0.1)
  fig_sim = create_performance_map(sim_mw, sim_pcd, sim_trim)

  res1, res2 = st.columns([1.5, 1])
  with res1:
    st.plotly_chart(fig_sim, use_container_width=True)
  with res2:
    st.markdown('<div class="card-safe">', unsafe_allow_html=True)
    st.markdown('#### Hasil Evaluasi Dynamic What-If')
    st.metric('Rekomendasi VIGV Trim', f'+{sim_trim:.2f} °')
    st.metric('Estimasi Hemat BBM', f'{sim_fsav:.3f} {fuel_unit}')
    st.metric('Estimasi Hemat Biaya', f'Rp {sim_rpsav:,.0f} / jam')
    st.metric('Reduksi Emisi CO2', f'{sim_co2:.1f} kg CO2 / jam')
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# HALAMAN 4: MANFAAT & PREDIKSI FILTER RUL
# ------------------------------------------------------------------------------
elif active_page == '🛡️ Manfaat & Prediksi Filter RUL':
  # Tetap hitung variabel yang dibutuhkan dari DCS Inputs
  dcs_set_point_load = 104.0
  dcs_act_vigv = 62.7
  dcs_fuel_cons = default_fuel_cons
  dcs_tat = 498.0
  dcs_temp_intake = round(input_temp_amb + 1.2, 1)
  dcs_pcd = 11.2
  dcs_pamb_mbar = float(input_pamb_mbar)
  dcs_rh_pct = float(input_rh_amb)

  rho_live_dcs, _ = calculate_moist_air_density(
      dcs_temp_intake, dcs_rh_pct, dcs_pamb_mbar
  )
  trim_live, f_sav_live, rp_sav_live, co2_live, status_msg_live = (
      calculate_vigv_trim(
          dcs_temp_intake,
          dcs_set_point_load,
          act_vigv=dcs_act_vigv,
          rho_moist=rho_live_dcs,
          mode_simulasi=False,
      )
  )

  st.subheader(f'Ringkasan Manfaat & Prediksi ({fuel_mode})')

  dcs_pamb_bar = dcs_pamb_mbar / 1000.0

  pcd_expected_clean = (dcs_pamb_bar * 11.15) * (
      dcs_set_point_load / 104.0
  ) ** 0.1
  pcd_deficit = max(pcd_expected_clean - dcs_pcd, 0.0)

  virtual_dp_filter_bar = 0.006 + (pcd_deficit * 0.02)
  virtual_dp_mbar = virtual_dp_filter_bar * 1000.0

  dp_warning_limit_bar = 0.015
  dp_replace_limit_bar = 0.019

  rate_dp_per_day_bar = 0.00007 if dcs_temp_intake >= 26.0 else 0.00012
  margin_to_replace_bar = max(
      dp_replace_limit_bar - virtual_dp_filter_bar, 0.0
  )
  estimated_rul_days = int(margin_to_replace_bar / rate_dp_per_day_bar)

  estimated_fuel_cons_after_trim = max(dcs_fuel_cons - f_sav_live, 0.0)
  total_energy_saved_kkal = f_sav_live * lhv_fuel

  if dcs_set_point_load > 0:
    heat_rate_gain_kkal = total_energy_saved_kkal / (
        dcs_set_point_load * 1000.0
    )
  else:
    heat_rate_gain_kkal = 0.0

  heat_rate_gain_kj = heat_rate_gain_kkal * 4.184

  p_in_virtual = dcs_pamb_bar - virtual_dp_filter_bar
  r_p = dcs_pcd / p_in_virtual
  t_in_k = dcs_temp_intake + 273.15
  eta_c_base = (
      (t_in_k * (r_p**0.286 - 1)) / (dcs_tat + 273.15 - t_in_k) * 100
  )
  eta_c_gain = 0.65 if abs(trim_live) > 0 else 0.0
  eta_c_final = min(eta_c_base + eta_c_gain, 88.5)

  sfc_actual = (
      (dcs_fuel_cons / dcs_set_point_load) if dcs_set_point_load > 0 else 0.0
  )
  sfc_saving = (
      (f_sav_live / dcs_set_point_load) if dcs_set_point_load > 0 else 0.0
  )

  st.markdown('### 1. Dampak Hemat Bahan Bakar & Dekarbonisasi')
  m1, m2, m3, m4 = st.columns(4)

  m1.metric(
      'Est. Konsumsi BB After Trim',
      f'{estimated_fuel_cons_after_trim:.2f} {fuel_unit}',
      delta=f'-{f_sav_live:.3f} {fuel_unit}',
  )
  m2.metric(
      'Laju Penghematan BB / Jam',
      f'{f_sav_live:.3f} {fuel_unit}',
      delta=f'-{sfc_saving:.4f} {fuel_unit}/MWh',
  )
  m3.metric('Penghematan Finansial / Jam', f'Rp {rp_sav_live:,.0f}')
  m4.metric('Reduksi Emisi CO2 / Jam', f'{co2_live:.1f} kg CO2/jam')

  st.markdown('---')
  st.markdown(
      '### 2. Status Tingkat Kekotoran & Prediksi Penggantian Filter Air Intake'
  )

  f_col1, f_col2, f_col3 = st.columns(3)

  f_col1.metric(
      'Virtual Estimasi Beda Tekanan (DP)',
      f'{virtual_dp_filter_bar:.3f} bar',
      delta=f'{virtual_dp_mbar:.1f} mbar (Soft-Sensor)',
  )

  f_col2.metric(
      'Prediksi Sisa Umur Pakai (RUL)',
      f'{estimated_rul_days} Hari Lagi',
      delta='Condition-Based Maintenance',
  )

  with f_col3:
    st.markdown('**Evaluasi & Rekomendasi Pemeliharaan:**')
    if virtual_dp_filter_bar >= dp_replace_limit_bar:
      st.error(
          '🚨 **CRITICAL REPLACE!** Estimated DP ≥ 0,019 bar'
          f' ({virtual_dp_mbar:.1f} mbar). Filter kotor, jadwatchkan'
          ' penggantian segera!'
      )
    elif virtual_dp_filter_bar >= dp_warning_limit_bar:
      st.warning(
          '⚠️ **WARNING:** Estimated DP ≥ 0,015 bar'
          f' ({virtual_dp_mbar:.1f} mbar). Penumpukan debu meningkat, siapkan'
          ' stok filter.'
      )
    else:
      st.success(
          '🟢 **FILTER CLEAN:** Estimated DP < 0,015 bar'
          f' ({virtual_dp_mbar:.1f} mbar). Kondisi aliran intake bersih &'
          ' optimal.'
      )

  st.markdown('---')
  st.markdown('### 3. Performa Termal & Efisiensi Kompresor GT')
  e1, e2, e3 = st.columns(3)
  e1.metric('Specific Fuel Cons (SFC)', f'{sfc_actual:.4f} {fuel_unit}/MWh')

  e2.metric(
      'Perbaikan Heat Rate (ΔHR)',
      f'-{heat_rate_gain_kkal:.2f} kkal/kWh',
      delta=f'-{heat_rate_gain_kj:.2f} kJ/kWh',
  )
  e3.metric(
      'Efisiensi Isentropik (ηc)',
      f'{eta_c_final:.2f} %',
      delta=f'+{eta_c_gain:.2f} %',
  )
