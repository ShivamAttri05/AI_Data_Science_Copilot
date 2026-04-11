"""
Explainable ML Pipeline Analyzer - Main Streamlit Application

A comprehensive web application for transparent ML workflows with critical
failure detection, reasoned model recommendations, and production deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime
import pickle
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import DataLoader, validate_dataset
from modules.data_quality import DataQualityChecker
from modules.eda_engine import EDAEngine
from modules.ai_insights import AIInsightsGenerator
from modules.automl_engine import AutoMLEngine
from modules.experiment_analysis import ExperimentAnalyzer
from modules.model_deployment import ModelDeployer

from utils.helpers import create_directory_structure
from modules.data_loader import load_sample_dataset

st.set_page_config(
    page_title="ML Pipeline Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
#  Palette  : indigo primary · slate neutrals · semantic greens/ambers/reds
#  Typeface : DM Sans (UI body) + DM Mono (code)
#  Aesthetic: Refined data-tool — dark sidebar, crisp white canvas,
#             strong type hierarchy, purposeful accent colour.
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

/* ── TOKENS ───────────────────────────────────────────────────────────────── */
:root {
    --p:       #4f46e5;
    --p-lt:    #eef2ff;
    --p-dk:    #3730a3;
    --p-mid:   #6366f1;

    --ok:      #059669;
    --ok-lt:   #ecfdf5;
    --warn:    #d97706;
    --warn-lt: #fffbeb;
    --err:     #dc2626;
    --err-lt:  #fef2f2;
    --inf:     #0284c7;
    --inf-lt:  #f0f9ff;

    --s50:  #f8fafc;
    --s100: #f1f5f9;
    --s200: #e2e8f0;
    --s300: #cbd5e1;
    --s400: #94a3b8;
    --s500: #64748b;
    --s600: #475569;
    --s700: #334155;
    --s800: #1e293b;
    --s900: #0f172a;

    --r-sm: 6px;
    --r-md: 10px;
    --r-lg: 14px;
    --r-xl: 20px;

    --f-ui:   'DM Sans', sans-serif;
    --f-mono: 'DM Mono', monospace;
    --t: 0.18s ease;
}

/* ── GLOBAL ───────────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: var(--f-ui) !important; }
.block-container { padding-top:1.25rem !important; padding-bottom:4rem !important; max-width:1400px; }
.stApp > header { display:none; }

/* ── SIDEBAR ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--s900) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: var(--s300) !important; font-family: var(--f-ui) !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {
    color:#fff !important; font-size:0.68rem !important; font-weight:700 !important;
    letter-spacing:0.13em !important; text-transform:uppercase !important;
    margin:1.25rem 0 0.4rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-size:0.875rem !important; font-weight:400 !important; color:var(--s400) !important;
    padding:0.42rem 0.7rem !important; border-radius:var(--r-sm) !important;
    cursor:pointer; transition:all var(--t); display:flex; align-items:center; gap:0.5rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background:rgba(255,255,255,0.06) !important; color:#fff !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background:rgba(99,102,241,0.18) !important; color:#a5b4fc !important; font-weight:500 !important;
}
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.07) !important; margin:0.9rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stCaption"] { font-size:0.73rem !important; color:var(--s500) !important; }

/* ── TYPOGRAPHY CLASSES ───────────────────────────────────────────────────── */
.wordmark { font-size:1.45rem; font-weight:700; color:var(--s900); letter-spacing:-0.03em; line-height:1; }
.tagline  { font-size:0.82rem; color:var(--s500); margin-top:3px; }

.sec-over  { font-size:0.67rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:var(--p); display:block; margin-bottom:2px; }
.sec-title { font-size:1.3rem; font-weight:700; color:var(--s900); letter-spacing:-0.02em; line-height:1.2; }
.sec-rule  { width:28px; height:3px; background:var(--p); border-radius:2px; margin:6px 0 14px; }
.sec-body  { font-size:0.875rem; color:var(--s500); line-height:1.55; margin-bottom:1.4rem; }

/* ── PAGE HERO BAND ───────────────────────────────────────────────────────── */
.hero {
    background: var(--s900);
    border-radius: var(--r-xl);
    padding: 1.4rem 1.75rem;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
}
.hero-icon  { font-size:1.3rem; flex-shrink:0; }
.hero-label { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.14em; color:#6366f1; margin-bottom:2px; }
.hero-title { font-size:1.05rem; font-weight:700; color:#fff; letter-spacing:-0.02em; }
.hero-sub   { font-size:0.78rem; color:var(--s400); margin-top:2px; }

/* ── STAT CHIP ────────────────────────────────────────────────────────────── */
.chip {
    display:inline-flex; align-items:center; gap:6px;
    background:var(--s100); border:1px solid var(--s200); border-radius:999px;
    padding:3px 11px; font-size:0.78rem; font-weight:500; color:var(--s700);
}
.chip-dot { width:6px; height:6px; border-radius:50%; background:var(--ok); flex-shrink:0; }

/* ── BADGES ───────────────────────────────────────────────────────────────── */
.bdg { display:inline-flex; align-items:center; padding:3px 10px; border-radius:999px; font-size:0.71rem; font-weight:600; letter-spacing:0.02em; border:1px solid transparent; }
.bdg-i { background:#eef2ff; color:#3730a3; border-color:#c7d2fe; }
.bdg-v { background:#f5f3ff; color:#4c1d95; border-color:#ddd6fe; }
.bdg-g { background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }

/* ── FEATURE CARDS ────────────────────────────────────────────────────────── */
.feat {
    background:#fff; border:1px solid var(--s200); border-radius:var(--r-lg);
    padding:1.2rem 1rem 1rem; transition:border-color var(--t), transform var(--t);
}
.feat:hover { border-color:var(--p-mid); transform:translateY(-2px); }
.feat-ico   { width:38px; height:38px; border-radius:var(--r-md); background:var(--p-lt); display:flex; align-items:center; justify-content:center; font-size:1.1rem; margin-bottom:0.7rem; }
.feat-name  { font-size:0.875rem; font-weight:600; color:var(--s800); margin-bottom:2px; }
.feat-desc  { font-size:0.77rem; color:var(--s500); line-height:1.4; }

/* ── WORKFLOW STEPS ───────────────────────────────────────────────────────── */
.step {
    display:flex; align-items:flex-start; gap:0.85rem;
    padding:0.75rem 1rem; border-radius:var(--r-md); margin-bottom:0.38rem;
    border:1px solid var(--s100); background:#fff; transition:border-color var(--t);
}
.step:hover { border-color:var(--s300); }
.step-num   { min-width:25px; height:25px; border-radius:50%; background:var(--p); color:#fff; font-size:0.68rem; font-weight:700; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px; }
.step-t     { font-weight:600; font-size:0.875rem; color:var(--s800); }
.step-d     { font-size:0.77rem; color:var(--s500); margin-top:1px; line-height:1.35; }

/* ── SIDEBAR PROGRESS ─────────────────────────────────────────────────────── */
.pb-track { height:3px; background:rgba(255,255,255,0.08); border-radius:2px; margin:5px 0 12px; overflow:hidden; }
.pb-fill  { height:100%; background:linear-gradient(90deg,#6366f1,#8b5cf6); border-radius:2px; transition:width 0.5s ease; }
.pi { display:flex; align-items:center; gap:8px; padding:3px 0; font-size:0.79rem; }
.pi-done { color:#6ee7b7 !important; }
.pi-todo { color:var(--s600) !important; }
.pi-c    { width:15px; height:15px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.58rem; flex-shrink:0; }
.pi-c-y  { background:#059669; color:#fff; }
.pi-c-n  { background:var(--s700); }

/* ── WINNER BANNER ────────────────────────────────────────────────────────── */
.win {
    background:var(--s900); border:1px solid rgba(99,102,241,0.28);
    border-radius:var(--r-lg); padding:1.25rem 1.5rem;
    display:flex; align-items:center; gap:1rem; margin-bottom:1.2rem;
}
.win-trophy { font-size:1.7rem; flex-shrink:0; }
.win-lbl  { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.13em; color:#6366f1; margin-bottom:2px; }
.win-name { font-size:1.05rem; font-weight:700; color:#fff; letter-spacing:-0.02em; }
.win-sub  { font-size:0.77rem; color:var(--s400); margin-top:2px; }

/* ── CONFIDENCE ───────────────────────────────────────────────────────────── */
.conf-wrap  { text-align:right; }
.conf-lbl   { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.11em; color:var(--s400); }
.conf-val   { font-size:1.55rem; font-weight:700; letter-spacing:-0.03em; line-height:1; }
.conf-note  { font-size:0.71rem; color:var(--s400); margin-top:2px; }
.c-hi { color:#10b981; }
.c-md { color:#f59e0b; }
.c-lo { color:#f87171; }

/* ── HEALTH CHIPS ─────────────────────────────────────────────────────────── */
.hc { display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:5px 14px; font-size:0.82rem; font-weight:600; border:1px solid transparent; }
.hc-ok   { background:var(--ok-lt);   color:#065f46; border-color:#a7f3d0; }
.hc-mild { background:var(--warn-lt); color:#78350f; border-color:#fde68a; }
.hc-bad  { background:var(--err-lt);  color:#7f1d1d; border-color:#fecaca; }

/* ── STREAMLIT COMPONENT OVERRIDES ───────────────────────────────────────── */

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap:2px !important; border-bottom:2px solid var(--s100) !important;
    background:transparent !important; padding-bottom:0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family:var(--f-ui) !important; font-size:0.82rem !important; font-weight:500 !important;
    color:var(--s500) !important; padding:0.5rem 0.85rem !important;
    border-radius:var(--r-sm) var(--r-sm) 0 0 !important;
    background:transparent !important; border:none !important;
    border-bottom:2px solid transparent !important; margin-bottom:-2px !important;
    transition:all var(--t) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color:var(--p) !important; background:var(--p-lt) !important; }
[data-testid="stTabs"] [aria-selected="true"] { color:var(--p) !important; font-weight:600 !important; border-bottom:2px solid var(--p) !important; background:transparent !important; }
div[data-testid="stTabPanel"] { padding-top:1.2rem !important; }

/* Metrics */
[data-testid="stMetric"] { background:var(--s50) !important; border:1px solid var(--s200) !important; border-radius:var(--r-md) !important; padding:1rem 1.1rem !important; }
[data-testid="stMetric"] label { font-family:var(--f-ui) !important; font-size:0.7rem !important; font-weight:700 !important; text-transform:uppercase !important; letter-spacing:0.1em !important; color:var(--s500) !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family:var(--f-ui) !important; font-size:1.45rem !important; font-weight:700 !important; color:var(--s900) !important; letter-spacing:-0.03em !important; }
[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size:0.71rem !important; }

/* Buttons */
[data-testid="stButton"] > button { font-family:var(--f-ui) !important; font-size:0.875rem !important; font-weight:600 !important; border-radius:var(--r-md) !important; transition:all var(--t) !important; }
[data-testid="stButton"] > button[kind="primary"] { background:var(--p) !important; border:none !important; color:#fff !important; letter-spacing:0.01em !important; }
[data-testid="stButton"] > button[kind="primary"]:hover { background:var(--p-dk) !important; transform:translateY(-1px) !important; }
[data-testid="stButton"] > button[kind="secondary"] { background:transparent !important; border:1px solid var(--s300) !important; color:var(--s700) !important; }
[data-testid="stButton"] > button[kind="secondary"]:hover { border-color:var(--p) !important; color:var(--p) !important; }

/* Inputs */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div,
[data-testid="stNumberInput"] > div > div {
    border-radius:var(--r-md) !important; border:1px solid var(--s200) !important;
    font-family:var(--f-ui) !important; font-size:0.875rem !important;
    transition:border-color var(--t) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextInput"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div:focus-within {
    border-color:var(--p) !important; box-shadow:0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border:2px dashed var(--s200) !important; border-radius:var(--r-lg) !important;
    background:var(--s50) !important; transition:all var(--t) !important;
}
[data-testid="stFileUploader"]:hover { border-color:var(--p-mid) !important; background:var(--p-lt) !important; }

/* Expander */
[data-testid="stExpander"] { border:1px solid var(--s200) !important; border-radius:var(--r-md) !important; background:#fff !important; overflow:hidden !important; }
[data-testid="stExpander"] summary { font-family:var(--f-ui) !important; font-size:0.875rem !important; font-weight:500 !important; color:var(--s700) !important; padding:0.7rem 1rem !important; background:var(--s50) !important; border-bottom:1px solid var(--s100) !important; }
[data-testid="stExpander"] summary:hover { background:var(--p-lt) !important; color:var(--p) !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border:1px solid var(--s200) !important; border-radius:var(--r-md) !important; overflow:hidden !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius:var(--r-md) !important; border-width:1px !important; font-family:var(--f-ui) !important; font-size:0.875rem !important; }

/* Download button */
[data-testid="stDownloadButton"] > button { font-family:var(--f-ui) !important; font-size:0.875rem !important; font-weight:600 !important; border-radius:var(--r-md) !important; background:var(--ok) !important; color:#fff !important; border:none !important; }
[data-testid="stDownloadButton"] > button:hover { background:#047857 !important; transform:translateY(-1px) !important; }

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background:var(--p) !important; border-color:var(--p) !important; }

/* Code */
code { font-family:var(--f-mono) !important; font-size:0.81rem !important; background:var(--s100) !important; border:1px solid var(--s200) !important; border-radius:var(--r-sm) !important; padding:1px 6px !important; color:var(--p-dk) !important; }
pre code { background:var(--s900) !important; border:none !important; color:#e2e8f0 !important; padding:0 !important; }
pre { background:var(--s900) !important; border-radius:var(--r-md) !important; padding:1rem 1.25rem !important; border:1px solid rgba(255,255,255,0.06) !important; }

/* Checkboxes */
[data-testid="stCheckbox"] label { font-family:var(--f-ui) !important; font-size:0.875rem !important; color:var(--s700) !important; }

/* Spinner */
[data-testid="stSpinner"] p { font-family:var(--f-ui) !important; font-size:0.875rem !important; color:var(--p) !important; }

/* Caption */
[data-testid="stCaption"] { font-family:var(--f-ui) !important; font-size:0.77rem !important; color:var(--s400) !important; }

/* JSON */
[data-testid="stJson"] { border-radius:var(--r-md) !important; border:1px solid var(--s200) !important; font-family:var(--f-mono) !important; font-size:0.79rem !important; }

/* Plotly container */
[data-testid="stPlotlyChart"] { border:1px solid var(--s100) !important; border-radius:var(--r-md) !important; overflow:hidden !important; background:#fff !important; }

/* Divider */
hr { border:none !important; border-top:1px solid var(--s100) !important; margin:1.25rem 0 !important; }

/* Form submit */
[data-testid="stForm"] [data-testid="stButton"] > button[kind="primary"] { background:var(--p) !important; width:100% !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────

_PL = dict(
    font_family="DM Sans, sans-serif",
    font_color="#334155",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=44, b=24, l=12, r=12),
    title_font=dict(size=13, color="#1e293b", family="DM Sans"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
    xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(size=11), title_font=dict(size=11)),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(size=11), title_font=dict(size=11)),
    colorway=["#4f46e5","#0284c7","#059669","#d97706","#dc2626","#7c3aed","#db2777"],
)

def _fig(f):
    f.update_layout(**_PL)
    return f


# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in {
    'df':None,'data_loaded':False,'quality_report':None,'eda_results':None,
    'ai_insights':None,'automl_results':None,'target_col':None,'problem_type':None,
    'best_model':None,'analyzer':None,'predictions':None,'prediction_engine':None,
    'prediction_model_path':None,'feature_names':None,'last_prediction':None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── UI helpers ────────────────────────────────────────────────────────────────

def _pipeline_progress():
    return [
        ("Dataset upload",     st.session_state.data_loaded),
        ("Data quality check", st.session_state.quality_report is not None),
        ("EDA analysis",       st.session_state.eda_results    is not None),
        ("AI insights",        st.session_state.ai_insights    is not None),
        ("AutoML training",    st.session_state.automl_results is not None),
    ]


def _problem_badge(ptype: str) -> str:
    cfg = {
        'binary_classification':    ('bdg-i', 'Binary classification'),
        'multiclass_classification': ('bdg-v', 'Multiclass classification'),
        'regression':               ('bdg-g', 'Regression'),
    }
    cls, lbl = cfg.get(ptype, ('bdg-i', ptype.replace('_',' ').title()))
    return f'<span class="bdg {cls}">{lbl}</span>'


def _hero(icon: str, title: str, sub: str):
    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-icon">{icon}</div>'
        f'<div><div class="hero-label">Section</div>'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-sub">{sub}</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )


def _section(over: str, title: str):
    st.markdown(
        f'<span class="sec-over">{over}</span>'
        f'<div class="sec-title">{title}</div>'
        f'<div class="sec-rule"></div>',
        unsafe_allow_html=True
    )


# ── render_header ─────────────────────────────────────────────────────────────

def render_header():
    """Render the application header."""
    c_brand, c_stat = st.columns([4, 1])

    with c_brand:
        st.markdown(
            '<div class="wordmark">🔬 ML Pipeline Analyzer</div>'
            '<div class="tagline">Transparent AutoML · Failure detection · Reasoned recommendations</div>',
            unsafe_allow_html=True
        )

    with c_stat:
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            st.markdown(
                f'<div style="text-align:right;padding-top:0.3rem;">'
                f'<span class="chip"><span class="chip-dot"></span>{df.shape[0]:,} rows · {df.shape[1]} cols</span>',
                unsafe_allow_html=True
            )
            if st.session_state.problem_type:
                st.markdown(
                    f'<div style="text-align:right;margin-top:6px;">'
                    f'{_problem_badge(st.session_state.problem_type)}</div>',
                    unsafe_allow_html=True
                )

    st.markdown('<hr style="margin:0.9rem 0 1.2rem;">', unsafe_allow_html=True)


# ── render_sidebar ────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown("## Navigation")

        page = st.radio(
            "Go to",
            ["🏠 Home","📁 Dataset Upload","🔍 Data Quality","📈 EDA Dashboard",
             "🧠 AI Insights","🤖 AutoML Training","📊 Model Evaluation","🔮 Predictions","🚀 Deployment"],
            label_visibility="collapsed"
        )

        st.divider()

        steps    = _pipeline_progress()
        done_cnt = sum(d for _, d in steps)
        pct      = int(done_cnt / len(steps) * 100)

        st.markdown(
            f'<div style="font-size:0.67rem;font-weight:700;letter-spacing:0.13em;'
            f'text-transform:uppercase;color:#94a3b8;margin-bottom:4px;">'
            f'Progress · {done_cnt}/{len(steps)}</div>'
            f'<div class="pb-track"><div class="pb-fill" style="width:{pct}%"></div></div>',
            unsafe_allow_html=True
        )

        for label, done in steps:
            c, t, m = ("pi-c-y","pi-done","✓") if done else ("pi-c-n","pi-todo","")
            st.markdown(
                f'<div class="pi"><div class="pi-c {c}">{m}</div>'
                f'<span class="{t}">{label}</span></div>',
                unsafe_allow_html=True
            )

        if st.session_state.data_loaded and st.session_state.df is not None:
            st.divider()
            df = st.session_state.df
            st.markdown(
                '<span style="font-size:0.67rem;font-weight:700;letter-spacing:0.13em;'
                'text-transform:uppercase;color:#94a3b8;">Dataset</span>',
                unsafe_allow_html=True
            )
            st.caption(f"Rows: **{df.shape[0]:,}** · Cols: **{df.shape[1]}**")
            if st.session_state.target_col:
                st.caption(f"Target: **{st.session_state.target_col}**")
            if st.session_state.automl_results:
                bm = st.session_state.automl_results.get('best_model','—')
                st.caption(f"Best model: **{bm}**")

        st.divider()
        st.caption("Explainable AutoML · Built with Streamlit")

        return page


# ── render_home ───────────────────────────────────────────────────────────────

def render_home():
    """Render the home page."""
    _section("Get started", "Welcome to ML Pipeline Analyzer")
    st.markdown(
        '<p class="sec-body">An end-to-end, transparent ML workspace — from raw data to deployed API, '
        'with every decision explained.</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    for icon, name, desc, col in [
        ("📁","Upload data","CSV & Excel with auto-profiling",c1),
        ("🔍","Audit quality","Missing values, outliers & constants",c2),
        ("🤖","AutoML","Multi-model CV training",c3),
        ("🚀","Deploy","Export .pkl or FastAPI",c4),
    ]:
        with col:
            st.markdown(
                f'<div class="feat"><div class="feat-ico">{icon}</div>'
                f'<div class="feat-name">{name}</div>'
                f'<div class="feat-desc">{desc}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown('<br>', unsafe_allow_html=True)
    _section("How it works", "Seven-step pipeline")

    for i, (title, desc) in enumerate([
        ("📁 Dataset Upload",          "Load CSV / Excel or pick a built-in sample"),
        ("🔍 Data Quality Check",      "Detect missing values, duplicates & outliers"),
        ("📈 Exploratory Analysis",    "Distributions, correlations & categorical breakdowns"),
        ("🧠 AI Dataset Investigation","Gemini-powered insights & preprocessing guidance"),
        ("🤖 AutoML Training",         "Train & compare models with overfit diagnostics"),
        ("📊 Model Evaluation",        "CV stability, bias–variance & feature importance"),
        ("🚀 Deployment",              "Export .pkl or generate a FastAPI service"),
    ], 1):
        st.markdown(
            f'<div class="step"><div class="step-num">{i}</div>'
            f'<div><div class="step-t">{title}</div>'
            f'<div class="step-d">{desc}</div></div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<br>', unsafe_allow_html=True)
    if not st.session_state.data_loaded:
        st.info("👈 Start by uploading your dataset in **Dataset Upload**.")
    else:
        done = sum(d for _, d in _pipeline_progress())
        st.success(f"✅ {done}/5 pipeline steps completed — keep going!")


# ── render_dataset_upload ─────────────────────────────────────────────────────

def render_dataset_upload():
    """Render the dataset upload page."""
    _hero("📁", "Dataset Upload", "Load your data or explore a built-in sample")

    uploaded_file = st.file_uploader(
        "Drop a CSV or Excel file here",
        type=['csv','xlsx','xls'],
        help="Supports .csv, .xlsx, .xls"
    )

    st.markdown(
        '<div style="font-size:0.77rem;font-weight:700;color:var(--s500);'
        'text-transform:uppercase;letter-spacing:0.09em;margin:1rem 0 0.5rem;">Or try a sample</div>',
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    sample_dataset = None
    with col1:
        if st.button("🌸 Iris — classification", use_container_width=True):   sample_dataset = 'iris'
    with col2:
        if st.button("🍷 Wine — classification", use_container_width=True):   sample_dataset = 'wine'
    with col3:
        if st.button("🏠 California — regression", use_container_width=True): sample_dataset = 'california'

    if uploaded_file is not None or sample_dataset:
        try:
            loader = DataLoader()
            if uploaded_file:
                df = loader.load_from_upload(uploaded_file)
                src = uploaded_file.name
            else:
                df  = load_sample_dataset(sample_dataset)
                src = f"{sample_dataset} (sample)"

            is_valid, issues = validate_dataset(df)
            if not is_valid:
                st.warning("⚠️ Validation issues:")
                for issue in issues:
                    st.markdown(f"- {issue}")

            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"✅ Loaded **{src}** — {df.shape[0]:,} rows · {df.shape[1]} columns")

            _section("Overview", "Dataset snapshot")

            numeric_cols     = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            missing_total    = int(df.isnull().sum().sum())
            missing_pct      = round(missing_total / df.size * 100, 1)

            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Rows",        f"{df.shape[0]:,}")
            m2.metric("Columns",     df.shape[1])
            m3.metric("Numeric",     len(numeric_cols))
            m4.metric("Categorical", len(categorical_cols))
            m5.metric("Missing",     f"{missing_pct}%",
                      delta=None if missing_pct==0 else f"{missing_total} cells",
                      delta_color="inverse")

            t1, t2 = st.tabs(["Data preview","Column details"])
            with t1:
                st.dataframe(df.head(10), use_container_width=True)
            with t2:
                md = df.isnull().sum()
                mp = (md / len(df) * 100).round(2)
                ci = pd.DataFrame({'Column':df.columns,'Dtype':[str(d) for d in df.dtypes],
                                   'Unique':[df[c].nunique() for c in df.columns],
                                   'Missing':md.values,'Missing %':mp.values})
                st.dataframe(ci.style.background_gradient(subset=['Missing %'],cmap='Reds',vmin=0,vmax=100),
                             use_container_width=True, hide_index=True)

            _section("Configure","Select target column")

            cs, cb = st.columns([2,1])
            with cs:
                target_col = st.selectbox(
                    "Column to predict (optional)",
                    ['— none —'] + list(df.columns),
                    help="Leave as 'none' for EDA-only workflows."
                )

            if target_col != '— none —':
                st.session_state.target_col = target_col
                n_u = df[target_col].nunique()
                if df[target_col].dtype in ['object','category'] or n_u < 10:
                    ptype = 'binary_classification' if n_u == 2 else 'multiclass_classification'
                else:
                    ptype = 'regression'
                st.session_state.problem_type = ptype

                with cb:
                    st.markdown(
                        f'<div style="background:var(--inf-lt);border:1px solid #bae6fd;'
                        f'border-radius:var(--r-md);padding:0.8rem 1rem;margin-top:1.55rem;">'
                        f'<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
                        f'letter-spacing:0.1em;color:#0284c7;margin-bottom:4px;">Detected type</div>'
                        f'{_problem_badge(ptype)}'
                        f'<div style="font-size:0.72rem;color:#0c4a6e;margin-top:5px;">'
                        f'{n_u} unique values</div></div>',
                        unsafe_allow_html=True
                    )

                with st.expander("📊 Target distribution"):
                    if ptype == 'regression':
                        f = _fig(px.histogram(df, x=target_col, nbins=30,
                                              title=f"Distribution — {target_col}",
                                              color_discrete_sequence=["#4f46e5"]))
                        f.update_layout(showlegend=False)
                        st.plotly_chart(f, use_container_width=True)
                    else:
                        vc = df[target_col].value_counts().reset_index()
                        vc.columns = [target_col, 'count']
                        f = _fig(px.bar(vc, x=target_col, y='count',
                                        title=f"Class distribution — {target_col}",
                                        color_discrete_sequence=["#4f46e5"]))
                        f.update_layout(showlegend=False)
                        st.plotly_chart(f, use_container_width=True)
            else:
                st.session_state.target_col   = None
                st.session_state.problem_type = None

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
            st.info("Check that the file is a valid CSV or Excel workbook.")


# ── render_data_quality ───────────────────────────────────────────────────────

def render_data_quality():
    """Render the data quality page."""
    _hero("🔍","Data Quality Report","Detect issues before they derail training")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Upload a dataset first.")
        return

    df = st.session_state.df
    cb, ch = st.columns([1,3])
    with cb:
        run_check = st.button("🔍 Run quality check", type="primary", use_container_width=True)
    with ch:
        st.caption("Checks for missing values, duplicates, outliers, and constant columns.")

    if run_check:
        with st.spinner("Analysing data quality…"):
            checker = DataQualityChecker(df)
            report  = checker.run_all_checks(st.session_state.target_col)
            st.session_state.quality_report = report

    if not st.session_state.quality_report:
        st.info("Click **Run quality check** to audit your dataset.")
        return

    report   = st.session_state.quality_report
    overview = report['overview']
    ni, nw   = len(report['issues']), len(report['warnings'])

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Rows",           f"{overview['n_rows']:,}")
    m2.metric("Columns",        overview['n_columns'])
    m3.metric("Memory (MB)",    overview['memory_usage_mb'])
    m4.metric("Critical issues", ni, delta=None if ni==0 else "action needed", delta_color="inverse")
    m5.metric("Warnings",       nw, delta=None if nw==0 else "review advised", delta_color="off")

    if report['issues']:
        for issue in report['issues']: st.error(f"🚨 {issue}")
    elif not report['warnings']:
        st.success("✅ No critical issues found.")

    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs(["Missing values","Duplicates","Outliers","Constants","Warnings","Recommendations"])

    with tabs[0]:
        missing  = report['missing_values']
        st.markdown(f"**{missing['total_missing']:,}** missing cells ({missing['total_missing_percentage']}% of dataset)")
        if missing['columns_with_missing'] > 0:
            mdf = pd.DataFrame([{'Column':c,'Count':d['count'],'Percentage':d['percentage']}
                                 for c,d in missing['missing_by_column'].items()]).sort_values('Percentage',ascending=False)
            f = _fig(px.bar(mdf.head(20), x='Column', y='Percentage',
                            title='Missing values per column (%)',
                            color='Percentage', color_continuous_scale=['#e0e7ff','#4f46e5'], text_auto='.1f'))
            f.update_layout(coloraxis_showscale=False)
            st.plotly_chart(f, use_container_width=True)
            st.dataframe(mdf, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No missing values.")

    with tabs[1]:
        dup = report['duplicates']
        if dup['n_duplicate_rows'] > 0:
            st.warning(f"⚠️ **{dup['n_duplicate_rows']:,}** duplicate rows ({dup['percentage']}%).")
        else:
            st.success("✅ No duplicate rows.")

    with tabs[2]:
        out = report['outliers']
        st.caption(f"Detection method: **{out['method'].upper()}**")
        if out['outliers_by_column']:
            odf = pd.DataFrame([{'Column':c,'Count':d['count'],'Percentage':d['percentage']}
                                 for c,d in out['outliers_by_column'].items()]).sort_values('Percentage',ascending=False)
            f = _fig(px.bar(odf.head(15), x='Column', y='Percentage',
                            title='Outlier rate per column (%)',
                            color='Percentage', color_continuous_scale=['#fef3c7','#d97706'], text_auto='.1f'))
            f.update_layout(coloraxis_showscale=False)
            st.plotly_chart(f, use_container_width=True)
            st.dataframe(odf, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No significant outliers.")

    with tabs[3]:
        con = report['constant_columns']
        if con['n_constant'] > 0:
            st.error(f"🚨 **{con['n_constant']}** constant column(s) — will cause training errors.")
            for cd in con['constant_columns']: st.code(cd['column'])
        else:
            st.success("✅ No constant columns.")
        if con['n_near_constant'] > 0:
            st.warning(f"⚠️ **{con['n_near_constant']}** near-constant column(s).")

    with tabs[4]:
        if report['warnings']:
            for w in report['warnings']: st.warning(w)
        else:
            st.success("✅ No warnings.")

    with tabs[5]:
        for rec in report['recommendations']: st.info(f"💡 {rec}")


# ── render_eda ────────────────────────────────────────────────────────────────

def render_eda():
    """Render the EDA dashboard page."""
    _hero("📈","EDA Dashboard","Distributions, correlations and categorical profiles")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Upload a dataset first.")
        return

    df = st.session_state.df
    cb, ch = st.columns([1,3])
    with cb:
        run_eda = st.button("📊 Run EDA", type="primary", use_container_width=True)
    with ch:
        st.caption("Generates summary statistics, correlations, and distribution plots.")

    if run_eda:
        with st.spinner("Running EDA…"):
            engine  = EDAEngine(df)
            results = engine.run_full_analysis(st.session_state.target_col)
            st.session_state.eda_results = results
            st.session_state.eda_engine  = engine

    if not st.session_state.eda_results:
        st.info("Click **Run EDA** to explore your dataset.")
        return

    results = st.session_state.eda_results
    engine  = st.session_state.eda_engine
    tabs    = st.tabs(["Overview","Statistics","Correlations","Distributions","Categorical"])

    with tabs[0]:
        ov = results['overview']
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows",        f"{ov['shape'][0]:,}")
        c2.metric("Columns",     ov['shape'][1])
        c3.metric("Missing %",   ov['missing_percentage'])
        c4.metric("Numeric",     ov['n_numeric'])

        tdf = pd.DataFrame({'Type':['Numeric','Categorical','Datetime'],
                            'Count':[ov['n_numeric'],ov['n_categorical'],ov['n_datetime']]})
        f   = _fig(px.pie(tdf, names='Type', values='Count',
                          title='Column type breakdown',
                          color_discrete_sequence=['#4f46e5','#7c3aed','#0284c7']))
        f.update_traces(textfont_size=12)
        st.plotly_chart(f, use_container_width=True)

    with tabs[1]:
        stats = results['summary_statistics']
        if 'numeric' in stats:
            st.markdown("**Numeric summary**")
            st.dataframe(stats['numeric'].style.format(precision=3), use_container_width=True)
        if 'categorical' in stats:
            st.markdown("**Categorical summary**")
            st.dataframe(stats['categorical'], use_container_width=True)

    with tabs[2]:
        corr = results['correlation_analysis']
        if 'pearson_correlation' in corr:
            hf = engine.generate_correlation_heatmap(use_plotly=True)
            if hf:
                hf.update_layout(**_PL)
                st.plotly_chart(hf, use_container_width=True)
        if corr.get('high_correlation_pairs'):
            st.markdown("**Highly correlated pairs**")
            st.dataframe(pd.DataFrame(corr['high_correlation_pairs']),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No high-correlation pairs above threshold.")

    with tabs[3]:
        nc = df.select_dtypes(include=[np.number]).columns
        if len(nc) == 0:
            st.info("No numeric columns available.")
        else:
            sc = st.selectbox("Select column", nc, key="eda_dist")
            fs = engine.generate_distribution_plots([sc], use_plotly=True)
            if fs:
                fs[0][1].update_layout(**_PL)
                st.plotly_chart(fs[0][1], use_container_width=True)
            ds = results['distributions']['distribution_stats'].get(sc, {})
            if ds:
                d1,d2,d3,d4 = st.columns(4)
                d1.metric("Mean",     f"{ds.get('mean',0):.3f}")
                d2.metric("Std dev",  f"{ds.get('std',0):.3f}")
                d3.metric("Skewness",f"{ds.get('skewness',0):.3f}")
                d4.metric("Normal?",  "Yes" if ds.get('is_normal') else "No")

    with tabs[4]:
        cc = df.select_dtypes(include=['object','category']).columns
        if len(cc) == 0:
            st.info("No categorical columns available.")
        else:
            sc2 = st.selectbox("Select column", cc, key="eda_cat")
            vc  = df[sc2].value_counts().head(20)
            f2  = _fig(px.bar(x=vc.values, y=vc.index, orientation='h',
                              title=f"Top categories — {sc2}",
                              color=vc.values, color_continuous_scale=['#e0e7ff','#4f46e5'],
                              text_auto=True))
            f2.update_layout(coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
            st.plotly_chart(f2, use_container_width=True)


# ── render_ai_insights ────────────────────────────────────────────────────────

def render_ai_insights():
    """Render the AI insights page."""
    _hero("🧠","AI Dataset Investigation",
          "Gemini-powered pattern detection, preprocessing advice & model suggestions")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Upload a dataset first.")
        return

    df = st.session_state.df
    with st.expander("🔑 Gemini API key (optional)"):
        api_key = st.text_input("API key", type="password",
                                help="Leave blank to use GEMINI_API_KEY env var.")

    cb, ch = st.columns([1,3])
    with cb:
        run_ai = st.button("🧠 Generate insights", type="primary", use_container_width=True)
    with ch:
        st.caption("Surfaces patterns, preprocessing advice, and model recommendations.")

    if run_ai:
        with st.spinner("Analysing with AI…"):
            try:
                gen = AIInsightsGenerator(api_key=api_key if api_key else None)
                st.session_state.ai_insights = gen.analyze_dataset(
                    df, st.session_state.target_col, st.session_state.problem_type
                )
            except Exception as e:
                st.error(f"Error: {e}")

    if not st.session_state.ai_insights:
        st.info("Click **Generate insights** to get AI-powered analysis.")
        return

    ins = st.session_state.ai_insights
    if 'note' in ins: st.info(ins['note'])

    ca, cb2 = st.columns(2)
    with ca:
        if 'dataset_overview' in ins:
            st.markdown("#### Dataset overview")
            ov = ins['dataset_overview']
            st.markdown(f"**Description:** {ov.get('description','—')}")
            st.markdown(f"**Size assessment:** {ov.get('size_assessment','—')}")
        if 'key_insights' in ins:
            st.markdown("#### Key insights")
            for i in ins['key_insights']: st.markdown(f"- {i}")
        if 'ml_problem_type' in ins:
            st.markdown("#### ML problem type")
            ml = ins['ml_problem_type']
            st.markdown(f"**Suggested:** {ml.get('suggested_type','—')}")
            st.markdown(f"**Confidence:** {ml.get('confidence','—')}")
            st.markdown(f"**Reasoning:** {ml.get('reasoning','—')}")

    with cb2:
        if ins.get('data_quality_issues'):
            st.markdown("#### Data quality issues")
            for issue in ins['data_quality_issues']:
                sev = issue.get('severity','low')
                msg = f"**{issue.get('issue','Issue')}** — {issue.get('recommendation','')}"
                (st.error if sev=='high' else st.warning if sev=='medium' else st.info)(msg)
        if ins.get('recommended_preprocessing'):
            st.markdown("#### Preprocessing recommendations")
            for s in ins['recommended_preprocessing']:
                with st.expander(f"{s.get('step','Step')} (priority: {s.get('priority','—')})"):
                    st.write(s.get('reason','—'))

    if ins.get('important_features'):
        st.markdown("#### Important features")
        st.dataframe(pd.DataFrame(ins['important_features']), use_container_width=True, hide_index=True)
    if ins.get('model_recommendations'):
        st.markdown("#### Recommended models")
        st.dataframe(pd.DataFrame(ins['model_recommendations']), use_container_width=True, hide_index=True)


# ── render_automl ─────────────────────────────────────────────────────────────

def render_automl():
    """Render the AutoML training page."""
    _hero("🤖","AutoML Training",
          "Train multiple models with penalised CV selection & bias-variance diagnostics")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Upload a dataset first."); return
    if not st.session_state.target_col:
        st.warning("⚠️ Select a target column in **Dataset Upload** first."); return

    df, target_col = st.session_state.df, st.session_state.target_col

    with st.expander("⚙️ Training settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1: test_size = st.slider("Test split",0.10,0.40,0.20,0.05, format="%.0f%%")
        with c2: cv_folds  = st.slider("CV folds",3,10,5)

    st.markdown(
        '<div style="font-size:0.77rem;font-weight:700;color:var(--s500);'
        'text-transform:uppercase;letter-spacing:0.09em;margin:1rem 0 0.5rem;">Select models</div>',
        unsafe_allow_html=True
    )

    is_clf = st.session_state.problem_type in ['binary_classification','multiclass_classification']
    avm    = ({'Logistic Regression':'logistic_regression','Random Forest':'random_forest',
               'Gradient Boosting':'gradient_boosting','XGBoost':'xgboost'}
              if is_clf else
              {'Linear Regression':'linear_regression','Ridge':'ridge',
               'Random Forest':'random_forest','Gradient Boosting':'gradient_boosting','XGBoost':'xgboost'})

    sel, mc = [], st.columns(len(avm))
    for i,(name,key) in enumerate(avm.items()):
        with mc[i]:
            if st.checkbox(name, value=True, key=f"m_{key}"): sel.append(key)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Start training", type="primary", use_container_width=True):
        if not sel: st.error("Select at least one model."); return
        with st.spinner(f"Training {len(sel)} model(s)…"):
            try:
                engine  = AutoMLEngine(test_size=test_size, cv_folds=cv_folds)
                results = engine.auto_train(df, target_col, problem_type=st.session_state.problem_type, models_to_train=sel)
                st.session_state.automl_results = results
                st.session_state.best_model     = engine.best_model
                st.session_state.automl_engine  = engine
                st.success("✅ Training complete!")
            except Exception as e:
                st.error(f"❌ Training failed: {e}"); return

    if not st.session_state.automl_results: return

    results    = st.session_state.automl_results
    reasoning  = results['reasoning']
    confidence = reasoning['confidence']
    cc         = "c-hi" if confidence>=80 else "c-md" if confidence>=50 else "c-lo"

    st.divider()
    st.markdown(
        f'<div class="win">'
        f'<div class="win-trophy">🏆</div>'
        f'<div style="flex:1;"><div class="win-lbl">Best model</div>'
        f'<div class="win-name">{results["best_model"]}</div>'
        f'<div class="win-sub">Selected via penalised cross-validation</div></div>'
        f'<div class="conf-wrap">'
        f'<div class="conf-lbl">Pipeline confidence</div>'
        f'<div class="conf-val {cc}">{confidence}'
        f'<span style="font-size:0.85rem;font-weight:400;">/100</span></div>'
        f'<div class="conf-note">{reasoning["confidence_explanation"]}</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("##### ✅ Why it won")
        st.info(reasoning['why_it_won'])
        st.markdown("**Strengths:**")
        for s in reasoning['strengths'][:3]: st.markdown(f"- {s}")
    with r2:
        st.markdown("##### ⚠️ Failure modes")
        st.warning("\n".join(f"- {f}" for f in reasoning['when_it_fails']))

    _section("Analysis","Model comparison & tradeoffs")
    with st.expander("Full tradeoff table", expanded=True):
        tdf = pd.DataFrame(reasoning['tradeoff_analysis'])
        st.dataframe(tdf[['model','cv_mean','test_score','overfit_status','interpretability','tradeoff_summary']],
                     use_container_width=True, hide_index=True)

    mk  = 'accuracy' if is_clf else 'r2'
    f   = _fig(px.bar(results['comparison'], x='model', y=mk,
                      color='overfit_status',
                      color_discrete_map={'none':'#059669','mild':'#d97706','severe':'#dc2626'},
                      title=f"Performance ({mk.upper()}) · coloured by overfit risk",
                      text_auto='.3f'))
    f.update_layout(legend_title_text='Overfit risk')
    st.plotly_chart(f, use_container_width=True)

    if results.get('feature_importance') is not None:
        _section("Explainability","Feature importance (top 10)")
        fi = results['feature_importance'].head(10)
        ff = _fig(px.bar(fi, x='importance', y='feature', orientation='h',
                         title='What drives the model?',
                         color='importance', color_continuous_scale=['#e0e7ff','#4f46e5'],
                         text_auto='.3f'))
        ff.update_layout(coloraxis_showscale=False,
                         yaxis=dict(autorange='reversed', categoryorder='total ascending'))
        st.plotly_chart(ff, use_container_width=True)


# ── render_model_evaluation ───────────────────────────────────────────────────

def render_model_evaluation():
    """Render the model evaluation page."""
    _hero("📊","Model Diagnostics & Evaluation",
          "Health checks, test-set metrics, and CV stability analysis")

    if not st.session_state.automl_results:
        st.warning("⚠️ Train models first in **AutoML Training**."); return

    results     = st.session_state.automl_results
    model_names = list(results['model_results'].keys())
    default_ix  = model_names.index(results['best_model']) if results['best_model'] in model_names else 0

    sel = st.selectbox("Select model to evaluate", model_names, index=default_ix,
                       help="Defaults to the winning model.")
    if not sel: return

    mr = results['model_results'][sel]
    if 'error' in mr: st.error(f"Error: {mr['error']}"); return

    tabs = st.tabs(["🩺 Health check","📈 Metrics","🔁 CV stability","🗂️ Raw report"])

    with tabs[0]:
        of = mr['overfitting']
        bv = mr['bias_variance']
        hc_map = {'none':'hc-ok','mild':'hc-mild','severe':'hc-bad'}

        st.markdown(
            f'Overfitting status: <span class="hc {hc_map.get(of["status"],"hc-ok")}">'
            f'{of["status"].title()}</span>',
            unsafe_allow_html=True
        )
        st.caption(of['details'])
        st.divider()
        st.markdown("#### Bias–variance decomposition")
        b1,b2,b3 = st.columns(3)
        b1.metric("Bias level",     bv['bias_level'].title())
        b2.metric("Variance level", bv['variance_level'].title())
        b3.metric("Dominant error", bv['dominant_error'].title())
        st.info(f"**Interpretation:** {bv['interpretation']}")
        st.success(f"**Recommendation:** {bv['recommendation']}")

    with tabs[1]:
        metrics = mr['metrics']
        is_clf  = st.session_state.problem_type in ['binary_classification','multiclass_classification']
        if is_clf:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Accuracy",  f"{metrics.get('accuracy',0):.4f}")
            c2.metric("Precision", f"{metrics.get('precision',0):.4f}")
            c3.metric("Recall",    f"{metrics.get('recall',0):.4f}")
            c4.metric("F1",        f"{metrics.get('f1',0):.4f}")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("RMSE", f"{metrics.get('rmse',0):.4f}")
            c2.metric("MAE",  f"{metrics.get('mae',0):.4f}")
            c3.metric("R²",   f"{metrics.get('r2',0):.4f}")
            c4.metric("MAPE", f"{metrics.get('mape',0):.2f}%")

    with tabs[2]:
        cv   = mr['cv_scores']
        mu   = np.mean(cv)
        sd   = np.std(cv)
        st.markdown(f"CV mean: **{mu:.4f}** · std dev: **{sd:.4f}**")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1,len(cv)+1)), y=cv, mode='lines+markers', name='Score',
            line=dict(color='#4f46e5',width=2),
            marker=dict(size=8,color='#4f46e5',line=dict(width=2,color='#fff'))
        ))
        fig.add_hline(y=mu, line_dash='dash', line_color='#dc2626',
                      annotation_text=f"Mean {mu:.4f}", annotation_font_color='#dc2626')
        fig.update_layout(**_PL, xaxis_title='CV fold', yaxis_title='Score',
                          title='Score stability across folds')
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.json({"metrics":mr['metrics'],"train_metrics":mr['train_metrics'],
                 "overfitting":mr['overfitting'],"bias_variance":mr['bias_variance']})


# ── render_predictions ────────────────────────────────────────────────────────

def render_predictions():
    """Render the predictions page with custom new data inputs."""
    _hero("🔮","Predictions","Single or batch inference on new data")

    if 'automl_engine' not in st.session_state or st.session_state.automl_engine is None:
        st.warning("⚠️ Complete **AutoML Training** first."); return

    engine, problem_type = st.session_state.automl_engine, st.session_state.problem_type

    def get_feature_names_from_pipeline(pipeline):
        """Extract feature names from pipeline with better error handling."""
        fn = []
        if hasattr(pipeline,'named_steps'):
            pre = pipeline.named_steps.get('preprocessor')
            if pre is not None:
                try:
                    if hasattr(pre,'get_feature_names_out'):   fn = pre.get_feature_names_out().tolist()
                    elif hasattr(pre,'feature_names_in_'):     fn = pre.feature_names_in_.tolist()
                    elif hasattr(pre,'transformers_'):
                        for nm,tr,cols in pre.transformers_:
                            if tr in ('drop',None): continue
                            try:
                                fn.extend(tr.get_feature_names_out(cols).tolist() if hasattr(tr,'get_feature_names_out')
                                          else tr.feature_names_in_.tolist() if hasattr(tr,'feature_names_in_') else cols)
                            except Exception: fn.extend(cols)
                except Exception as e: st.warning(f"Could not extract feature names: {e}")
        if not fn and hasattr(engine,'_preprocessed_df'): fn = engine._preprocessed_df.columns.tolist()
        if not fn and hasattr(engine,'_X_train_ref'):     fn = engine._X_train_ref.columns.tolist()
        if not fn: fn = [f"feature_{i}" for i in range(10)]
        return fn

    try:
        feature_names = get_feature_names_from_pipeline(engine.best_model)
        st.session_state.feature_names = feature_names
        c1,c2,c3 = st.columns(3)
        c1.metric("Model",    st.session_state.automl_results.get("best_model", "—") if st.session_state.get("automl_results") else "—")
        c2.metric("Features", len(feature_names))
        c3.metric("Type",     (problem_type or "—").replace('_',' ').title())

        with st.expander("📋 All features"):
            cpr = 4
            for i in range(0, len(feature_names), cpr):
                row = feature_names[i:i+cpr]
                rc  = st.columns(cpr)
                for j,f in enumerate(row): rc[j].code(f)

    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        feature_names = (engine._preprocessed_df.columns.tolist()
                         if hasattr(engine,'_preprocessed_df')
                         else [f"feature_{i}" for i in range(10)])
        st.session_state.feature_names = feature_names

    t1,t2,t3 = st.tabs(["🎯 Single prediction","📊 Batch prediction","💾 Load saved model"])

    with t1:
        st.markdown("Enter values for each feature:")
        with st.form("prediction_form"):
            input_data, cpr2 = {}, 3
            for i in range(0, len(feature_names), cpr2):
                rc = st.columns(cpr2)
                for j in range(cpr2):
                    idx = i+j
                    if idx >= len(feature_names): break
                    fn  = feature_names[idx]
                    ft  = ("categorical"
                           if hasattr(engine,'_preprocessed_df') and fn in engine._preprocessed_df.columns
                           and engine._preprocessed_df[fn].dtype in ['object','category']
                           else "numeric")
                    with rc[j]:
                        if ft == "categorical":
                            cats = engine._preprocessed_df[fn].unique().tolist()
                            input_data[fn] = st.selectbox(fn, cats, key=f"s_{idx}")
                        else:
                            input_data[fn] = st.number_input(fn, value=0.0, step=0.1, format="%.4f", key=f"s_{idx}")

            if st.form_submit_button("🔮 Predict", type="primary", use_container_width=True):
                try:
                    X_new = pd.DataFrame([input_data])[feature_names]
                    pred  = engine.predict(X_new)[0]
                    cp, cb3 = st.columns(2)
                    with cp:
                        st.metric("Prediction", f"{pred:.4f}" if problem_type=='regression' else str(pred))
                    if problem_type != 'regression' and hasattr(engine.best_model,'predict_proba'):
                        try:
                            proba = engine.best_model.predict_proba(engine.preprocessor.transform(X_new))[0]
                            with cb3:
                                st.metric("Confidence", f"{max(proba):.1%}")
                                if len(proba)<=10:
                                    f = _fig(px.bar(pd.DataFrame({'Class':range(len(proba)),'Probability':proba}),
                                                    x='Class',y='Probability',color_discrete_sequence=['#4f46e5']))
                                    st.plotly_chart(f, use_container_width=True)
                        except Exception as e: st.warning(f"Could not compute probabilities: {e}")
                    st.success("✅ Done!")
                    st.session_state.last_prediction = {'input':input_data,'prediction':pred}
                except Exception as e: st.error(f"Prediction error: {e}")

    with t2:
        bc = ", ".join(feature_names[:5]) + ("…" if len(feature_names)>5 else "")
        st.caption(f"Required columns: {bc}")
        uf = st.file_uploader("Choose CSV", type=['csv'], key="batch_upload")
        if uf:
            try:
                df_new = pd.read_csv(uf)
                mc2    = set(feature_names) - set(df_new.columns)
                ec     = set(df_new.columns) - set(feature_names)
                if mc2:
                    st.error(f"Missing columns: {', '.join(sorted(mc2))}")
                else:
                    if ec:
                        st.warning(f"Ignoring: {', '.join(sorted(ec))}")
                        df_new = df_new[feature_names]
                    preds      = engine.predict(df_new)
                    df_r       = df_new.copy()
                    df_r['prediction'] = preds
                    if problem_type != 'regression' and hasattr(engine.best_model,'predict_proba'):
                        try:
                            probs = engine.best_model.predict_proba(engine.preprocessor.transform(df_new))
                            df_r['confidence'] = np.max(probs,axis=1).round(4)
                        except Exception: pass
                    st.success(f"✅ {len(preds):,} predictions generated.")
                    st.dataframe(df_r, use_container_width=True, hide_index=True)
                    st.download_button("📥 Download predictions (CSV)",
                                       df_r.to_csv(index=False).encode('utf-8'),
                                       "predictions.csv","text/csv",use_container_width=True)
                    if problem_type=='regression' and len(preds)>1:
                        f = _fig(px.histogram(preds, nbins=30, title='Prediction distribution',
                                              color_discrete_sequence=['#4f46e5']))
                        f.update_layout(showlegend=False)
                        st.plotly_chart(f, use_container_width=True)
            except Exception as e: st.error(f"Batch error: {e}")

    with t3:
        st.markdown("Load a previously exported `.pkl` model without retraining.")
        mf = st.file_uploader("Upload model file", type=['pkl','pickle'], key="model_load")
        if mf:
            try:
                md = pickle.load(mf)
                if isinstance(md,dict):
                    st.session_state.prediction_engine = md
                    st.success("✅ Model loaded.")
                    if 'metadata' in md:
                        meta = md['metadata']
                        mc3,mc4 = st.columns(2)
                        mc3.info(f"**Model:** {meta.get('model_name','—')}")
                        mc3.info(f"**Problem type:** {meta.get('problem_type','—')}")
                        mc4.info(f"**Training date:** {meta.get('training_date','—')}")
                        if 'best_cv_score' in meta: mc4.info(f"**CV score:** {meta['best_cv_score']:.4f}")
                    if 'metadata' in md and 'feature_names' in md['metadata']:
                        st.session_state.feature_names = md['metadata']['feature_names']
                        st.caption(f"Features: {len(st.session_state.feature_names)}")
                else:
                    st.error("Invalid model format.")
            except Exception as e: st.error(f"Error loading model: {e}")


# ── render_deployment ─────────────────────────────────────────────────────────

def render_deployment():
    """Render the deployment page."""
    _hero("🚀","Model Deployment","Export as .pkl or spin up a FastAPI service")

    if not st.session_state.best_model:
        st.warning("⚠️ Train a model first in **AutoML Training**."); return

    engine = st.session_state.automl_engine
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 💾 Save model")
        st.caption("Exports the full sklearn pipeline as a `.pkl` file.")
        model_name = st.text_input("Model name","trained_model",key="save_name")

        if st.button("💾 Export model", type="primary", use_container_width=True):
            with st.spinner("Saving…"):
                try:
                    deployer = ModelDeployer()
                    fi       = engine.get_feature_importance()
                    meta     = {
                        'model_name':model_name,'problem_type':st.session_state.problem_type,
                        'target_column':st.session_state.target_col,
                        'training_date':datetime.now().isoformat(),
                        'best_cv_score':float(engine.results[engine.best_model_name]['cv_mean']),
                        'feature_names':st.session_state.feature_names or [],
                    }
                    fp = deployer.save_model(model=engine.best_model,model_name=model_name,
                                             preprocessor=engine.preprocessor,metadata=meta)
                    st.success(f"✅ Saved: `{fp}`")
                    with open(fp,'rb') as fh:
                        st.download_button("📥 Download model file", fh.read(),
                                           file_name=os.path.basename(fp),
                                           mime='application/octet-stream',use_container_width=True)
                except Exception as e: st.error(f"Save failed: {e}")

    with c2:
        st.markdown("#### 🌐 FastAPI service")
        st.caption("Generates a ready-to-run REST API with Swagger docs.")
        api_port = st.number_input("Port",8000,9000,8000)

        if st.button("🌐 Generate API", type="primary", use_container_width=True):
            with st.spinner("Generating API…"):
                try:
                    deployer = ModelDeployer()
                    mp  = deployer.save_model(model=engine.best_model,model_name="api_model",
                                              preprocessor=engine.preprocessor,
                                              metadata={'problem_type':st.session_state.problem_type,
                                                        'target_column':st.session_state.target_col,
                                                        'feature_names':st.session_state.feature_names or []})
                    adir = deployer.generate_fastapi_service(model_path=mp,
                                                             feature_names=st.session_state.feature_names or [],
                                                             port=api_port)
                    st.success(f"✅ API generated in: `{adir}`")
                    st.code(f"cd api\npip install -r requirements.txt\nuvicorn main:app --reload --port {api_port}",
                            language="bash")
                    st.info(f"Swagger UI → `http://localhost:{api_port}/docs`  \nReDoc → `http://localhost:{api_port}/redoc`")
                except Exception as e: st.error(f"API generation failed: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    """Main application function."""
    render_header()
    page = render_sidebar()

    if   page == "🏠 Home":             render_home()
    elif page == "📁 Dataset Upload":   render_dataset_upload()
    elif page == "🔍 Data Quality":     render_data_quality()
    elif page == "📈 EDA Dashboard":    render_eda()
    elif page == "🧠 AI Insights":      render_ai_insights()
    elif page == "🤖 AutoML Training":  render_automl()
    elif page == "📊 Model Evaluation": render_model_evaluation()
    elif page == "🔮 Predictions":      render_predictions()
    elif page == "🚀 Deployment":       render_deployment()


if __name__ == "__main__":
    main()