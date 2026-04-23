"""
Demo Académica — Interamericana Norte · Chery Tiggo 2
Sistema de Predicción de Demanda (ISDI · Troncal)
Datos de ejemplo pre-cargados. Sin autenticación. URL pública.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Demo · TIGGO 2 · Interamericana Norte",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700&display=swap');
:root {
  --c-bg:      #04080F;
  --c-surface: #070C18;
  --c-raised:  #0A1020;
  --c-border:  rgba(0,224,255,0.13);
  --c-cyan:    #00E0FF;
  --c-gold:    #FFC107;
  --c-red:     #FF3A5C;
  --c-green:   #00F5A0;
  --c-purple:  #A78BFA;
  --c-text:    #C9D8E6;
  --c-muted:   #3F5060;
}
html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
  font-family: 'Rajdhani', sans-serif !important;
  background-color: var(--c-bg) !important;
}
h1 { font-family:'Rajdhani',sans-serif!important; font-weight:700!important; letter-spacing:.06em!important; text-transform:uppercase!important; }
h2,h3 { font-family:'Rajdhani',sans-serif!important; font-weight:600!important; letter-spacing:.04em!important; }
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family:'Rajdhani',sans-serif!important; font-weight:600!important;
  font-size:.84rem!important; letter-spacing:.07em!important; text-transform:uppercase!important;
}
[data-testid="metric-container"] {
  border-radius:5px!important; border:1px solid var(--c-border)!important;
  padding:10px!important; background:var(--c-surface)!important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-family:'Rajdhani',sans-serif!important; font-weight:700!important;
  letter-spacing:.08em!important; text-transform:uppercase!important;
  font-size:.74rem!important; color:var(--c-muted)!important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family:'JetBrains Mono',monospace!important; color:var(--c-cyan)!important;
}
[data-testid="stSidebar"] {
  border-right:1px solid rgba(0,224,255,0.07)!important;
  background-color:var(--c-surface)!important;
}
[data-testid="stSidebar"] * { font-family:'Rajdhani',sans-serif!important; }
[data-testid="stDataFrame"] { font-family:'JetBrains Mono',monospace!important; font-size:.8rem!important; }
code,pre { font-family:'JetBrains Mono',monospace!important; }

.kpi-card {
  background:linear-gradient(150deg,var(--c-surface) 0%,var(--c-raised) 100%);
  border:1px solid var(--c-border); border-radius:5px;
  padding:20px 16px 16px; text-align:center; position:relative; overflow:hidden;
  box-shadow:0 2px 18px rgba(0,0,0,.55); margin-bottom:6px;
}
.kpi-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--c-cyan),rgba(0,224,255,.15));
  box-shadow:0 1px 8px rgba(0,224,255,.25);
}
.kpi-icon { font-size:1.15rem; margin-bottom:10px; opacity:.55; }
.kpi-label {
  font-size:.63rem; letter-spacing:.16em; text-transform:uppercase;
  color:var(--c-muted); margin-bottom:8px;
  font-family:'Rajdhani',sans-serif; font-weight:700;
}
.kpi-label::before{content:'[ ';opacity:.5;} .kpi-label::after{content:' ]';opacity:.5;}
.kpi-value {
  font-size:2.25rem; font-weight:400;
  font-family:'JetBrains Mono',monospace;
  color:var(--c-cyan); line-height:1.1;
  text-shadow:0 0 22px rgba(0,224,255,.28);
}
.kpi-value.amber { color:var(--c-gold); text-shadow:0 0 22px rgba(255,193,7,.24); }
.kpi-value.blue  { color:#38BDF8;        text-shadow:0 0 22px rgba(56,189,248,.22); }
.kpi-value.green { color:var(--c-green); text-shadow:0 0 22px rgba(0,245,160,.22); }
.kpi-value.red   { color:var(--c-red);   text-shadow:0 0 22px rgba(255,58,92,.24); }
.kpi-sub { font-size:.68rem; color:var(--c-muted); margin-top:6px; font-family:'JetBrains Mono',monospace; }

.section-header {
  display:flex; align-items:center; gap:10px;
  margin:26px 0 14px; padding-bottom:10px;
  border-bottom:1px solid rgba(0,224,255,0.06);
}
.section-header-bar {
  width:3px; height:15px; flex-shrink:0;
  background:linear-gradient(180deg,var(--c-cyan),rgba(0,224,255,.2));
  border-radius:2px; box-shadow:0 0 10px rgba(0,224,255,.35);
}
.section-header-text {
  font-family:'Rajdhani',sans-serif; font-size:.85rem; font-weight:700;
  color:var(--c-text); letter-spacing:.12em; text-transform:uppercase;
}
.winner-box {
  background:rgba(255,193,7,.06); border-left:4px solid var(--c-gold);
  border-radius:0 5px 5px 0; padding:14px 18px; margin:12px 0;
  font-size:.98rem; font-weight:700; color:var(--c-gold);
  font-family:'Rajdhani',sans-serif; letter-spacing:.04em; text-transform:uppercase;
}
.success-box {
  background:rgba(0,245,160,.04); border-left:3px solid var(--c-green);
  border-radius:0 4px 4px 0; padding:12px 16px; margin:10px 0;
  font-family:'Rajdhani',sans-serif; font-size:.9rem; color:var(--c-text);
}
.info-box {
  background:rgba(0,224,255,.04); border-left:3px solid var(--c-cyan);
  border-radius:0 4px 4px 0; padding:12px 16px; margin:10px 0;
  font-family:'Rajdhani',sans-serif; font-size:.9rem; color:var(--c-text);
}
.app-footer {
  text-align:center; padding:20px 0 10px;
  font-size:.67rem; color:#1A2838;
  font-family:'JetBrains Mono',monospace;
  border-top:1px solid rgba(0,224,255,.05); margin-top:36px; letter-spacing:.07em;
}
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
[data-testid="stDecoration"]{display:none!important;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

COLORS = {
    'primary': '#00E0FF',
    'accent':  '#FFC107',
    'red':     '#FF3A5C',
    'green':   '#00F5A0',
    'series':  ['#00E0FF', '#FFC107', '#00F5A0', '#FF3A5C', '#A78BFA'],
}

def kpi(label, value, icon='', color=''):
    sub = ''
    val_cls = f'kpi-value {color}' if color else 'kpi-value'
    icon_h  = f'<div class="kpi-icon">{icon}</div>' if icon else ''
    return f"""<div class="kpi-card">{icon_h}
<div class="kpi-label">{label}</div>
<div class="{val_cls}">{value}</div>{sub}</div>"""

def sec(text, icon=''):
    i = f'<span style="margin-right:6px;opacity:.55">{icon}</span>' if icon else ''
    return f"""<div class="section-header">
<div class="section-header-bar"></div>
<span class="section-header-text">{i}{text}</span></div>"""

def theme(fig, h=None, title=None):
    layout = dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(4,8,15,0.92)',
        font=dict(family='Rajdhani, sans-serif', color='#7A95A8', size=12),
        xaxis=dict(gridcolor='rgba(0,224,255,0.07)', showgrid=True, zeroline=False,
                   tickfont=dict(family='JetBrains Mono, monospace', color='#3F5060', size=11)),
        yaxis=dict(gridcolor='rgba(0,224,255,0.07)', showgrid=True, zeroline=False,
                   tickfont=dict(family='JetBrains Mono, monospace', color='#3F5060', size=11)),
        hoverlabel=dict(bgcolor='#070C18', font=dict(family='Rajdhani, sans-serif', color='#C9D8E6', size=14),
                        bordercolor='rgba(0,224,255,0.3)'),
        legend=dict(bgcolor='rgba(4,8,15,0.85)', bordercolor='rgba(0,224,255,0.12)',
                    borderwidth=1, font=dict(family='Rajdhani, sans-serif', color='#7A95A8', size=12)),
        margin=dict(l=20, r=20, t=50, b=30),
    )
    if h:     layout['height'] = h
    if title: layout['title']  = dict(text=title,
                                       font=dict(family='Rajdhani, sans-serif', color='#C9D8E6', size=16),
                                       x=0, xanchor='left', pad=dict(l=4))
    fig.update_layout(**layout)
    return fig

# ── Datos pre-cargados ────────────────────────────────────────────────────────

# Histórico mensual: Jan 2021 – Dic 2024 (48 meses)
FECHAS_HIST = pd.date_range('2021-01-01', periods=48, freq='MS')
VENTAS_HIST = [
    16, 19, 22, 18, 17, 20, 19, 21, 23, 19, 22, 28,  # 2021
    20, 23, 27, 22, 21, 25, 23, 26, 28, 23, 27, 34,  # 2022
    23, 27, 31, 25, 24, 28, 26, 29, 31, 26, 30, 38,  # 2023
    25, 28, 32, 27, 25, 29, 27, 30, 32, 27, 31, 37,  # 2024
]

hist = pd.Series(VENTAS_HIST, index=FECHAS_HIST, name='Ventas')

# Predicciones: Jan 2025 – Jun 2025
FECHAS_PRED = pd.date_range('2025-01-01', periods=6, freq='MS')
PRED        = [28, 32, 35, 29, 27, 33]
IC_INF      = [22, 25, 28, 23, 21, 26]
IC_SUP      = [34, 39, 42, 35, 33, 40]

pred = pd.DataFrame({
    'Fecha':       FECHAS_PRED,
    'Mes':         [d.strftime('%b %Y') for d in FECHAS_PRED],
    'Predicción':  PRED,
    'IC_Inferior': IC_INF,
    'IC_Superior': IC_SUP,
})

# Walk-forward validation: Ene 2024 – Dic 2024
FECHAS_WF  = pd.date_range('2024-01-01', periods=12, freq='MS')
WF_REAL    = [25, 28, 32, 27, 25, 29, 27, 30, 32, 27, 31, 37]
WF_PRED    = [23.1, 26.5, 29.8, 25.6, 23.9, 27.4, 25.2, 28.3, 30.1, 28.7, 29.2, 34.6]

wf = pd.DataFrame({
    'fecha':      FECHAS_WF,
    'real':       WF_REAL,
    'prediccion': WF_PRED,
})
wf['error_abs'] = abs(wf['real'] - wf['prediccion'])
wf['error_pct'] = wf['error_abs'] / wf['real'] * 100
MAPE = wf['error_pct'].mean()

# Comparativa de modelos ML
MODELOS = ['SARIMA', 'Prophet', 'XGBoost', 'Random Forest', 'Reg. Lineal']
MAPE_ML = [MAPE,      11.2,      14.5,      16.3,            22.1]
MAE_ML  = [1.7,        2.8,       3.7,        4.1,             5.6]
RMSE_ML = [2.3,        3.5,       4.6,        5.1,             6.9]

df_modelos = pd.DataFrame({
    'Modelo': MODELOS,
    'MAPE %': MAPE_ML,
    'MAE':    MAE_ML,
    'RMSE':   RMSE_ML,
}).sort_values('MAPE %')

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div style="padding:18px 0 10px;border-bottom:1px solid rgba(0,224,255,0.08);margin-bottom:16px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3F5060;letter-spacing:.12em;margin-bottom:4px;">
    DEMO ACADÉMICA · ISDI
  </div>
  <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.1rem;color:#C9D8E6;letter-spacing:.08em;text-transform:uppercase;">
    Interamericana Norte
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#3F5060;margin-top:3px;">
    Chery Tiggo 2 · Predicción de demanda
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="font-family:'Rajdhani',sans-serif;font-size:.75rem;color:#3F5060;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">
  Parámetros del Modelo
</div>""", unsafe_allow_html=True)
    st.code("SARIMA(1,1,1)(1,1,1)[12]\nAIC: -148.34\nBIC: -137.21", language=None)

    st.markdown("""
<div style="font-family:'Rajdhani',sans-serif;font-size:.75rem;color:#3F5060;letter-spacing:.1em;text-transform:uppercase;margin:16px 0 8px;">
  Cobertura de datos
</div>""", unsafe_allow_html=True)
    st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#7A95A8;line-height:2;">
  Histórico: Ene 2021 – Dic 2024<br>
  Meses: 48<br>
  Horizonte: 6 meses<br>
  Validación: Walk-forward<br>
  Optimización: Optuna TPE
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Datos de ejemplo para demostración académica. No representan cifras reales de la empresa.")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;gap:16px;padding:12px 0 18px;
            margin-bottom:18px;border-bottom:1px solid rgba(0,224,255,0.07);">
  <div>
    <h1 style="font-size:1.35rem!important;margin:0!important;padding:0!important;
               color:#C9D8E6;letter-spacing:.08em;">
      Sistema de Predicción de Demanda — Tiggo 2
    </h1>
    <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#3F5060;
                margin-top:5px;letter-spacing:.04em;">
      Interamericana Norte &nbsp;·&nbsp; ISDI Troncal &nbsp;·&nbsp; Demo Académica
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── KPIs globales ─────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("Total Ventas",      f"{sum(VENTAS_HIST):,}",          "📦"),        unsafe_allow_html=True)
c2.markdown(kpi("Meses de datos",    "48",                             "📅", "blue"), unsafe_allow_html=True)
c3.markdown(kpi("MAPE Walk-Forward", f"{MAPE:.1f}%",                  "🎯", "green"),unsafe_allow_html=True)
c4.markdown(kpi("Próximo mes",       f"{PRED[0]} uds",                 "🔮", "amber"),unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["📊 Histórico", "🔮 Predicción", "🔄 Validación del Modelo", "🏆 Comparativa ML", "📋 Resumen del Proyecto"])

# ══ Tab 1: Histórico ══════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown(sec("Serie Temporal — Ventas Mensuales Tiggo 2", "📊"), unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)
    h1.metric("Promedio mensual",  f"{hist.mean():.1f} uds")
    h2.metric("Máximo histórico",  f"{hist.max():.0f} uds")
    h3.metric("Mínimo histórico",  f"{hist.min():.0f} uds")

    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode='lines+markers', name='Ventas Mensuales',
        line=dict(color=COLORS['primary'], width=2.5),
        marker=dict(size=5, color=COLORS['primary']),
        fill='tozeroy', fillcolor='rgba(0,224,255,0.05)',
    ))
    fig_h.add_hline(
        y=hist.mean(), line_dash='dot', line_color=COLORS['accent'],
        annotation_text=f"Media: {hist.mean():.1f}",
        annotation_position="top right",
        annotation_font_color=COLORS['accent'],
    )
    theme(fig_h, h=480, title='Ventas Mensuales — Tiggo 2 · Interamericana Norte')
    fig_h.update_layout(hovermode='x unified', xaxis_title='Fecha', yaxis_title='Unidades')
    st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar': False})

    with st.expander("📊 Estadísticas descriptivas por año"):
        hist_df = hist.to_frame('Ventas')
        hist_df['Año'] = hist_df.index.year
        resumen_anual = hist_df.groupby('Año')['Ventas'].agg(
            Total='sum', Promedio='mean', Máximo='max', Mínimo='min'
        ).round(1)
        st.dataframe(
            resumen_anual.style.background_gradient(subset=['Total'], cmap='Blues'),
            use_container_width=True
        )

# ══ Tab 2: Predicción ════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown(sec("Predicción Enero – Junio 2025", "🔮"), unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    p1.markdown(kpi("Próximo mes",     f"{PRED[0]} uds",                    "🔮"),        unsafe_allow_html=True)
    p2.markdown(kpi("Total horizonte", f"{sum(PRED)} uds",                  "📦", "blue"), unsafe_allow_html=True)
    p3.markdown(kpi("Promedio mensual",f"{sum(PRED)/len(PRED):.1f} uds",    "📊", "amber"),unsafe_allow_html=True)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode='lines', name='Histórico',
        line=dict(color=COLORS['primary'], width=2),
    ))
    fig_p.add_trace(go.Scatter(
        x=pred['Fecha'], y=pred['Predicción'],
        mode='lines+markers', name='Predicción SARIMA',
        line=dict(color=COLORS['accent'], width=2.5),
        marker=dict(size=9, symbol='circle', color=COLORS['accent'],
                    line=dict(color='#080D18', width=1.5)),
    ))
    fig_p.add_trace(go.Scatter(
        x=pred['Fecha'].tolist() + pred['Fecha'].tolist()[::-1],
        y=pred['IC_Superior'].tolist() + pred['IC_Inferior'].tolist()[::-1],
        fill='toself', fillcolor='rgba(255,193,7,0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='IC 95%',
    ))
    fig_p.add_shape(
        type='line',
        x0=hist.index[-1], x1=hist.index[-1], y0=0, y1=1, yref='paper',
        line=dict(color='rgba(100,116,139,0.6)', width=1.5, dash='dot'),
    )
    theme(fig_p, h=560, title='Histórico + Predicción — Tiggo 2 · Interamericana Norte')
    fig_p.update_layout(hovermode='x unified', xaxis_title='Fecha', yaxis_title='Unidades')
    st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': False})

    st.subheader("📋 Tabla de predicciones")
    st.dataframe(
        pred[['Mes', 'Predicción', 'IC_Inferior', 'IC_Superior']].style
            .background_gradient(subset=['Predicción'], cmap='Blues')
            .format({'Predicción': '{:.0f}', 'IC_Inferior': '{:.0f}', 'IC_Superior': '{:.0f}'}),
        use_container_width=True, hide_index=True,
    )

    # Recomendación de compra
    proximo = PRED[0]
    ic_inf_p, ic_sup_p = IC_INF[0], IC_SUP[0]
    prom_hist = hist.mean()
    tendencia = ((hist.iloc[-3:].mean() - prom_hist) / prom_hist) * 100

    st.markdown(sec("Recomendación de Compra", "💼"), unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f"""<div class="info-box">
<strong>Estrategia Conservadora</strong><br>
Comprar: <strong>{int(ic_sup_p * 1.05)} unidades</strong><br>
Basado en IC superior + 5% · Minimiza riesgo de rotura de stock
</div>""", unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""<div class="success-box">
<strong>Estrategia Recomendada</strong><br>
Comprar: <strong>{int(proximo * 1.10)} unidades</strong><br>
Predicción central + 10% buffer · Equilibrio demanda / inventario
</div>""", unsafe_allow_html=True)

    if tendencia > 5:
        st.success(f"Tendencia CRECIENTE en los últimos 3 meses: +{tendencia:.1f}% vs. promedio histórico.")
    elif tendencia < -5:
        st.warning(f"Tendencia DECRECIENTE en los últimos 3 meses: {tendencia:.1f}% vs. promedio histórico.")
    else:
        st.info(f"Tendencia ESTABLE: {tendencia:+.1f}% vs. promedio histórico de {prom_hist:.1f} uds/mes.")

# ══ Tab 3: Validación ════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown(sec("Walk-Forward Validation — 2024", "🔄"), unsafe_allow_html=True)

    v1, v2, v3, v4 = st.columns(4)
    v1.markdown(kpi("MAPE Promedio",  f"{MAPE:.1f}%",                  "📊", "green"), unsafe_allow_html=True)
    v2.markdown(kpi("Mejor mes",      f"{wf['error_pct'].min():.1f}%", "✅"),           unsafe_allow_html=True)
    v3.markdown(kpi("Peor mes",       f"{wf['error_pct'].max():.1f}%", "⚠️", "amber"),  unsafe_allow_html=True)
    v4.markdown(kpi("Meses evaluados","12",                             "📅", "blue"),  unsafe_allow_html=True)

    fig_wf = go.Figure()
    fig_wf.add_trace(go.Scatter(
        x=wf['fecha'], y=wf['real'],
        mode='lines+markers', name='Real',
        line=dict(color=COLORS['primary'], width=2.5),
        marker=dict(size=7, color=COLORS['primary']),
    ))
    fig_wf.add_trace(go.Scatter(
        x=wf['fecha'], y=wf['prediccion'],
        mode='lines+markers', name='Predicción (walk-forward)',
        line=dict(color=COLORS['accent'], width=2.5, dash='dot'),
        marker=dict(size=7, color=COLORS['accent'], symbol='diamond'),
    ))
    theme(fig_wf, h=480, title='Walk-Forward Validation — Real vs. Predicción (2024)')
    fig_wf.update_layout(hovermode='x unified', xaxis_title='Mes', yaxis_title='Unidades')
    st.plotly_chart(fig_wf, use_container_width=True, config={'displayModeBar': False})

    wf_show = wf.copy()
    wf_show['fecha'] = wf_show['fecha'].dt.strftime('%B %Y')
    wf_show.columns = ['Mes', 'Real', 'Predicción', 'Error Abs.', 'Error %']
    st.dataframe(
        wf_show.style
               .background_gradient(subset=['Error %'], cmap='RdYlGn_r')
               .format({'Real': '{:.0f}', 'Predicción': '{:.1f}',
                        'Error Abs.': '{:.2f}', 'Error %': '{:.2f}%'}),
        use_container_width=True, hide_index=True,
    )

    if MAPE <= 10:
        st.markdown(f"""<div class="success-box">
MAPE = {MAPE:.1f}% — Modelo de <strong>alta fiabilidad</strong>. Error medio inferior al 10% sobre ventas reales.
Apto para planificación de pedidos y compromisos de inventario.
</div>""", unsafe_allow_html=True)

# ══ Tab 4: Comparativa ML ════════════════════════════════════════════════════

with tabs[3]:
    st.markdown(sec("Comparativa de 5 Modelos — Mismo Histórico", "🏆"), unsafe_allow_html=True)

    st.markdown(f"""<div class="winner-box">
    SARIMA seleccionado — Menor MAPE: {MAPE:.1f}%
    (optimizado con Optuna TPE · 150 combinaciones evaluadas)
</div>""", unsafe_allow_html=True)

    # Gráfico de barras MAPE
    colors_bar = [COLORS['accent'] if m == 'SARIMA' else COLORS['series'][2]
                  for m in df_modelos['Modelo']]
    fig_cmp = go.Figure(go.Bar(
        x=df_modelos['Modelo'], y=df_modelos['MAPE %'],
        marker_color=colors_bar,
        text=[f"{v:.1f}%" for v in df_modelos['MAPE %']],
        textposition='outside',
        textfont=dict(family='JetBrains Mono, monospace', color='#94A3B8', size=11),
    ))
    theme(fig_cmp, h=420, title='MAPE por Modelo — menor es mejor')
    fig_cmp.update_layout(xaxis_title='', yaxis_title='MAPE (%)',
                          showlegend=False, yaxis=dict(range=[0, 28]))
    st.plotly_chart(fig_cmp, use_container_width=True, config={'displayModeBar': False})

    # Tabla completa
    st.subheader("📋 Tabla de métricas")
    st.dataframe(
        df_modelos.style
                  .background_gradient(subset=['MAPE %'], cmap='RdYlGn_r')
                  .format({'MAPE %': '{:.1f}%', 'MAE': '{:.2f}', 'RMSE': '{:.2f}'})
                  .set_properties(**{'font-family': 'JetBrains Mono, monospace'}),
        use_container_width=True, hide_index=True,
    )

    # Radar
    categorias = ['MAPE', 'MAE', 'RMSE']
    # Normalizar: 1 = mejor (min), 0 = peor (max)
    def normalizar(vals):
        mn, mx = min(vals), max(vals)
        return [1 - (v - mn) / (mx - mn) for v in vals]

    n_mape = normalizar(MAPE_ML)
    n_mae  = normalizar(MAE_ML)
    n_rmse = normalizar(RMSE_ML)

    fig_r = go.Figure()
    for i, mod in enumerate(MODELOS):
        vals = [n_mape[i], n_mae[i], n_rmse[i]]
        fig_r.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categorias + [categorias[0]],
            fill='toself' if mod == 'SARIMA' else 'none',
            fillcolor='rgba(255,193,7,0.1)' if mod == 'SARIMA' else 'rgba(0,0,0,0)',
            name=mod,
            line=dict(color=COLORS['series'][i % len(COLORS['series'])],
                      width=3 if mod == 'SARIMA' else 1.5),
        ))
    theme(fig_r, h=420, title='Desempeño relativo por modelo (mayor = mejor)')
    fig_r.update_layout(polar=dict(
        bgcolor='rgba(4,8,15,0.9)',
        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                        gridcolor='rgba(0,224,255,0.08)'),
        angularaxis=dict(gridcolor='rgba(0,224,255,0.08)',
                         tickfont=dict(family='Rajdhani, sans-serif', color='#7A95A8', size=13)),
    ))
    st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar': False})

# ══ Tab 5: Resumen del Proyecto ═══════════════════════════════════════════════

with tabs[4]:
    st.markdown(sec("El Problema de Negocio", "🎯"), unsafe_allow_html=True)
    st.markdown("""
Interamericana Norte necesita **anticipar la demanda mensual del Chery Tiggo 2** para:

- Planificar órdenes de compra al fabricante con 2–3 meses de anticipación
- Reducir el costo de inmovilización de inventario
- Evitar roturas de stock que impactan la satisfacción del cliente
- Asignar unidades entre concesionarios de forma eficiente

Sin un sistema predictivo, las decisiones se basaban en criterio subjetivo del equipo comercial, con errores de estimación superiores al 20%.
""")

    st.markdown(sec("La Solución Desarrollada", "💡"), unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown("""
**Modelo SARIMA**
- Serie temporal con estacionalidad
- Parámetros optimizados con Optuna TPE
- 150 combinaciones evaluadas
- Validación walk-forward 12 meses
""")
    with col_s2:
        st.markdown("""
**Sistema completo**
- Interfaz web para usuarios no técnicos
- Roles: Admin · Analista · Gerente
- Historial de entrenamientos
- Audit log automático
""")
    with col_s3:
        st.markdown("""
**Tecnología**
- Streamlit (interfaz)
- statsmodels (SARIMA)
- Supabase (datos + auth)
- Google Gemini (asistente IA)
""")

    st.markdown(sec("Resultados Clave", "📈"), unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(kpi("MAPE obtenido",    f"{MAPE:.1f}%",  "🎯", "green"), unsafe_allow_html=True)
    r2.markdown(kpi("Reducción error",  "~60%",          "📉", "amber"), unsafe_allow_html=True)
    r3.markdown(kpi("Mejor modelo",     "SARIMA",        "🏆"),          unsafe_allow_html=True)
    r4.markdown(kpi("Horizonte pred.",  "6 meses",       "🔮", "blue"),  unsafe_allow_html=True)

    st.markdown(sec("Flujo de Trabajo", "🔄"), unsafe_allow_html=True)
    st.markdown("""
```
[Excel ventas] → [Limpieza + validación] → [Test ADF estacionariedad]
       ↓
[Búsqueda Optuna: 150 combinaciones SARIMA]
       ↓
[Walk-forward validation: 12 meses]
       ↓
[Aprobación del modelo → Publicación en Dashboard]
       ↓
[Gerente consulta predicciones + Asistente IA]
```
""")

    st.markdown(sec("Stack Tecnológico", "⚙️"), unsafe_allow_html=True)
    tech_cols = st.columns(5)
    techs = [
        ("Streamlit", "Interfaz web"),
        ("statsmodels", "Modelo SARIMA"),
        ("Optuna", "Optimización"),
        ("Supabase", "DB + Auth + Storage"),
        ("Gemini", "Asistente IA"),
    ]
    for i, (name, desc) in enumerate(techs):
        with tech_cols[i]:
            st.markdown(f"""
<div style="background:rgba(0,224,255,.04);border:1px solid rgba(0,224,255,.1);
            border-radius:5px;padding:14px;text-align:center;">
  <div style="font-family:'Rajdhani',sans-serif;font-weight:700;color:#C9D8E6;
              font-size:.95rem;letter-spacing:.06em;text-transform:uppercase;">
    {name}
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#3F5060;
              margin-top:4px;">
    {desc}
  </div>
</div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="app-footer">'
    'Sistema TIGGO 2 &nbsp;·&nbsp; Interamericana Norte &nbsp;·&nbsp; '
    'ISDI Troncal &nbsp;·&nbsp; Demo Académica &nbsp;·&nbsp; Datos de ejemplo'
    '</div>',
    unsafe_allow_html=True,
)
