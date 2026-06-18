import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import datetime
from autogluon.tabular import TabularPredictor
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Fatigue & Productivity Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# LOAD MODEL & DATA ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    model_path = "models/fatigue_prediction_best_model"
    if not os.path.exists(model_path):
        st.error(f"Model path '{model_path}' not found! Please run the training notebook first.")
        st.stop()
    return TabularPredictor.load(model_path)

@st.cache_data
def load_data():
    dataset_path = "employee_fatigue_productivity_dataset.csv"
    if not os.path.exists(dataset_path):
        st.error(f"Dataset '{dataset_path}' not found! Please run the dataset generator first.")
        st.stop()
        
    df = pd.read_csv(dataset_path)
    
    # Split the dataset dynamically to match the test set split in training (random_state=42, 20% test size)
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        stratify=df['Fatigue_Level']
    )
    return df, test_data

try:
    predictor = load_predictor()
    dataset, test_data = load_data()
except Exception as e:
    st.error(f"Error loading application artifacts: {e}")
    st.stop()

# ─────────────────────────────────────────────
# CUSTOM CSS FOR SLICK UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding-top: 1rem; padding-bottom: 0rem; }
.main            { background-color: #f5f7fb; }

/* ── Typography ── */
h1               { color: #0b1b5e; font-weight: 800; }
h2, h3           { color: #0b1b5e; }

/* ── Primary Button ── */
.stButton > button {
    background-color : #1565ff;
    color            : white;
    border-radius    : 10px;
    height           : 52px;
    width            : 100%;
    font-size        : 17px;
    font-weight      : bold;
    border           : none;
    margin-top       : 4px;
    transition       : background 0.2s;
}
.stButton > button:hover { background-color: #0d4ed8; }

/* ── KPI Metric Cards ── */
[data-testid="stMetric"] {
    background    : white;
    padding       : 20px 22px;
    border-radius : 18px;
    border-top    : 5px solid #1565ff;
    box-shadow    : 0px 2px 10px rgba(0,0,0,0.07);
    text-align    : left;
}
[data-testid="stMetricValue"] { font-size: 34px; font-weight: 700; color: #111827; }
[data-testid="stMetricLabel"] { font-size: 14px; color: #6B7280; }

/* ── Prediction Result Cards ── */
.pred-card-green {
    background    : linear-gradient(135deg, #dcfce7, #f0fdf4);
    padding       : 28px 32px;
    border-radius : 14px;
    border-left   : 6px solid #10b981;
    margin-bottom : 16px;
}
.pred-card-yellow {
    background    : linear-gradient(135deg, #fef9c3, #fefce8);
    padding       : 28px 32px;
    border-radius : 14px;
    border-left   : 6px solid #f59e0b;
    margin-bottom : 16px;
}
.pred-card-red {
    background    : linear-gradient(135deg, #fee2e2, #fff1f2);
    padding       : 28px 32px;
    border-radius : 14px;
    border-left   : 6px solid #ef4444;
    margin-bottom : 16px;
}
.pred-title {
    font-size      : 13px;
    font-weight    : 600;
    color          : #6B7280;
    text-transform : uppercase;
    letter-spacing : 0.07em;
    margin-bottom  : 6px;
}
.pred-value {
    font-size   : 46px;
    font-weight : 800;
    color       : #111827;
    line-height : 1.1;
}
.pred-sub  { font-size: 18px; color: #374151; margin-top: 6px; }
.pred-risk { font-size: 20px; font-weight: 700; margin-top: 10px; }

/* ── Section Headers ── */
.section-header {
    font-size      : 18px;
    font-weight    : 700;
    color          : #0b1b5e;
    border-bottom  : 2px solid #1565ff;
    padding-bottom : 6px;
    margin-bottom  : 14px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    height        : 46px;
    padding       : 0 24px;
    border-radius : 8px 8px 0 0;
    font-size     : 15px;
    font-weight   : 600;
}
.stTabs [aria-selected="true"] { background-color: #1565ff; color: white; }

/* ── Table ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Inputs ── */
.stSelectbox div[data-baseweb="select"] > div { min-height: 42px; }
.stTextInput input { height: 42px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:38px; color:#1d2340; font-weight:700; margin-bottom:2px;'>
Employee Fatigue & Productivity Intelligence Platform
</h1>
<p style='color:#6B7280; font-size:16px; margin-bottom:0;'>
Enterprise Workforce Scheduling & Performance Analytics
</p>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# TABS (Excluding EDA tab as requested)
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Fatigue Prediction",
    "Analytics Dashboard",
    "Bulk Fatigue Analysis"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — FATIGUE LEVEL PREDICTION
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Select Shift Record to Evaluate</div>", unsafe_allow_html=True)

    # Search and select employee shift record
    record_list = test_data.apply(lambda r: f"{r['Employee_ID']} on {r['Shift_Date']}", axis=1).unique()
    selected_record = st.selectbox(
        "Employee Shift Record",
        record_list,
        help="Select an Employee ID and Date — all feature values auto-populate below"
    )

    # Extract ID and date
    emp_id, shift_date = selected_record.split(" on ")
    row = test_data[(test_data["Employee_ID"] == emp_id) & (test_data["Shift_Date"] == shift_date)].iloc[0]

    # Display auto-populated features
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("**Employee Profile & Scheduling Information**")
        emp_id_val = st.text_input("Employee ID", value=str(row["Employee_ID"]))
        shift_date_val = st.text_input("Shift Date", value=str(row["Shift_Date"]))
        
        dept_options = ['Operations', 'Engineering', 'Customer Support', 'Sales', 'Logistics']
        department_val = st.selectbox("Department", options=dept_options, index=dept_options.index(row["Department"]) if row["Department"] in dept_options else 0)
        
        role_options = ['Supervisor', 'Associate', 'Software Engineer', 'QA Engineer', 'Support Agent', 'Team Lead', 'Account Executive', 'Sales Specialist', 'Coordinator', 'Dispatcher']
        job_role_val = st.selectbox("Job Role", options=role_options, index=role_options.index(row["Job_Role"]) if row["Job_Role"] in role_options else 0)
        
        c1, c2 = st.columns(2)
        with c1:
            age_val = st.number_input("Age", value=int(row["Age"]), min_value=18, max_value=100)
        with c2:
            exp_val = st.number_input("Experience (Years)", value=int(row["Experience_Years"]), min_value=0, max_value=60)

        c3, c4 = st.columns(2)
        with c3:
            shift_options = ['Morning', 'Evening', 'Night']
            shift_type_val = st.selectbox("Shift Type", options=shift_options, index=shift_options.index(row["Shift_Type"]) if row["Shift_Type"] in shift_options else 0)
        with c4:
            shift_duration_val = st.number_input("Shift Duration (Hours)", value=float(row["Shift_Duration_Hours"]), min_value=0.0)
            
        c5, c6 = st.columns(2)
        with c5:
            consec_days_val = st.number_input("Consecutive Work Days", value=int(row["Consecutive_Work_Days"]), min_value=0)
        with c6:
            night_shifts_val = st.number_input("Night Shifts (Last 30 Days)", value=int(row["Night_Shifts_Last_30_Days"]), min_value=0)

        c7, c8 = st.columns(2)
        with c7:
            overtime_hours_val = st.number_input("Overtime Hours", value=float(row["Overtime_Hours"]), min_value=0.0)
        with c8:
            monthly_ot_val = st.number_input("Monthly Overtime Hours", value=float(row["Monthly_Overtime_Hours"]), min_value=0.0)
            
        weekend_options = [0, 1]
        weekend_shift_val = st.selectbox("Weekend Shift", options=weekend_options, format_func=lambda x: "Yes" if x == 1 else "No", index=weekend_options.index(row["Weekend_Shift"]) if "Weekend_Shift" in row else 0)

    with right_col:
        st.markdown("**Workload, Environment & Health Information**")
        
        c9, c10 = st.columns(2)
        with c9:
            sleep_hours_val = st.number_input("Sleep Duration Previous Night (Hours)", value=float(row["Sleep_Hours_Previous_Night"]) if not pd.isna(row["Sleep_Hours_Previous_Night"]) else 7.0, min_value=0.0, max_value=24.0)
        with c10:
            break_time_val = st.number_input("Break Time (Minutes)", value=float(row["Break_Time_Minutes"]) if not pd.isna(row["Break_Time_Minutes"]) else 30.0, min_value=0.0)

        c11, c12 = st.columns(2)
        with c11:
            days_absent_val = st.number_input("Days Absent (Last 90 Days)", value=int(row["Days_Absent_Last_90_Days"]), min_value=0)
        with c12:
            late_arrivals_val = st.number_input("Late Arrivals (Last 30 Days)", value=int(row["Late_Arrivals_Last_30_Days"]), min_value=0)

        c13, c14 = st.columns(2)
        with c13:
            workload_options = ['Low', 'Medium', 'High']
            workload_intensity_val = st.selectbox("Workload Intensity", options=workload_options, index=workload_options.index(row["Workload_Intensity"]) if row["Workload_Intensity"] in workload_options else 0)
        with c14:
            tasks_assigned_val = st.number_input("Tasks Assigned", value=int(row["Tasks_Assigned"]), min_value=0)

        c15, c16 = st.columns(2)
        with c15:
            tasks_completed_val = st.number_input("Tasks Completed", value=int(row["Tasks_Completed"]), min_value=0)
        with c16:
            average_complexity_val = st.number_input("Average Task Complexity", value=float(row["Average_Task_Complexity"]) if not pd.isna(row["Average_Task_Complexity"]) else 3.0, min_value=1.0, max_value=5.0)

        c17, c18, c19 = st.columns(3)
        with c17:
            temperature_val = st.number_input("Workspace Temp (°C)", value=float(row["Temperature_Celsius"]), min_value=-10.0, max_value=60.0)
        with c18:
            humidity_val = st.number_input("Humidity (%)", value=float(row["Humidity_Percentage"]), min_value=0.0, max_value=100.0)
        with c19:
            noise_level_val = st.number_input("Noise Level (dB)", value=float(row["Noise_Level_dB"]) if not pd.isna(row["Noise_Level_dB"]) else 60.0, min_value=0.0)

        c20, c21, c22 = st.columns(3)
        with c20:
            stress_val = st.number_input("Stress Level (1-10)", value=int(row["Stress_Level"]), min_value=1, max_value=10)
        with c21:
            heart_rate_val = st.number_input("Heart Rate Avg (BPM)", value=float(row["Heart_Rate_Avg"]), min_value=0.0)
        with c22:
            hydration_breaks_val = st.number_input("Hydration Breaks", value=int(row["Hydration_Breaks_Count"]), min_value=0)

        c23, c24, c25 = st.columns(3)
        with c23:
            crowding_val = st.number_input("Workspace Crowding (1-10)", value=int(row["Workspace_Crowding_Score"]) if "Workspace_Crowding_Score" in row else 5, min_value=1, max_value=10)
        with c24:
            efficiency_val = st.number_input("Work Efficiency (%)", value=float(row["Work_Efficiency_Percentage"]) if not pd.isna(row["Work_Efficiency_Percentage"]) else 80.0, min_value=0.0, max_value=100.0)
        with c25:
            error_count_val = st.number_input("Error Count", value=int(row["Error_Count"]), min_value=0)

        c26, c27 = st.columns(2)
        with c26:
            response_time_val = st.number_input("Response Time (Min)", value=float(row["Response_Time_Minutes"]), min_value=0.0)
        with c27:
            quality_score_val = st.number_input("Quality Score", value=float(row["Quality_Score"]), min_value=0.0, max_value=100.0)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict Button ────────────────────────────────────
    if st.button("Evaluate Fatigue Risk", key="predict_single"):
        try:
            # Construct input row from the interactive widgets
            input_data = {
                'Employee_ID': emp_id_val,
                'Shift_Date': shift_date_val,
                'Department': department_val,
                'Job_Role': job_role_val,
                'Age': int(age_val),
                'Experience_Years': int(exp_val),
                'Weekend_Shift': int(weekend_shift_val),
                'Shift_Type': shift_type_val,
                'Shift_Duration_Hours': float(shift_duration_val),
                'Consecutive_Work_Days': int(consec_days_val),
                'Night_Shifts_Last_30_Days': int(night_shifts_val),
                'Overtime_Hours': float(overtime_hours_val),
                'Monthly_Overtime_Hours': float(monthly_ot_val),
                'Sleep_Hours_Previous_Night': float(sleep_hours_val),
                'Days_Absent_Last_90_Days': int(days_absent_val),
                'Late_Arrivals_Last_30_Days': int(late_arrivals_val),
                'Workload_Intensity': workload_intensity_val,
                'Tasks_Assigned': int(tasks_assigned_val),
                'Tasks_Completed': int(tasks_completed_val),
                'Average_Task_Complexity': float(average_complexity_val),
                'Break_Time_Minutes': float(break_time_val),
                'Temperature_Celsius': float(temperature_val),
                'Humidity_Percentage': float(humidity_val),
                'Noise_Level_dB': float(noise_level_val),
                'Workspace_Crowding_Score': int(crowding_val),
                'Stress_Level': int(stress_val),
                'Heart_Rate_Avg': float(heart_rate_val),
                'Hydration_Breaks_Count': int(hydration_breaks_val),
                'Work_Efficiency_Percentage': float(efficiency_val),
                'Error_Count': int(error_count_val),
                'Response_Time_Minutes': float(response_time_val),
                'Quality_Score': float(quality_score_val)
            }
            input_df = pd.DataFrame([input_data])

            # Predict using AutoGluon best model
            prediction = str(predictor.predict(input_df).iloc[0])

            # Classify risk card and recommendations
            if prediction == "Low":
                risk_label = "Low Fatigue Risk"
                card_class = "pred-card-green"
                recommendation = "Employee is fit for duty. Keep regular shifts and standard break intervals."
            elif prediction == "Medium":
                risk_label = "Medium Fatigue Risk"
                card_class = "pred-card-yellow"
                recommendation = "Provide extra hydration breaks. Limit shift duration and avoid night shifts next."
            else: # High
                risk_label = "High Fatigue Risk"
                card_class = "pred-card-red"
                recommendation = "Mandatory rest recommended. Relieve from heavy tasks and overtime immediately."

            # Display Results
            result_col, factor_col = st.columns([1, 1])

            with result_col:
                st.markdown("<div class='section-header'>Prediction Result</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='{card_class}'>
                    <div class='pred-title'>Predicted Fatigue Level</div>
                    <div class='pred-value'>{prediction}</div>
                    <div class='pred-sub'>{risk_label}</div>
                    <div style='margin-top:14px; font-size:15px; font-weight:600; color:#374151;'>
                        <b>Action Recommended:</b> {recommendation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with factor_col:
                st.markdown("<div class='section-header'>Key Contributing Factors</div>", unsafe_allow_html=True)
                key_factors = pd.DataFrame({
                    "Workday Feature": [
                        "Sleep Hours Previous Night",
                        "Stress Level (1-10)",
                        "Daily Overtime Hours",
                        "Consecutive Work Days",
                        "Workload Intensity",
                        "Workspace Temperature",
                        "Noise Level (dB)",
                        "Errors Logged"
                    ],
                    "Value": [
                        f"{sleep_hours_val} Hrs",
                        f"{stress_val} / 10",
                        f"{overtime_hours_val} Hrs",
                        f"{consec_days_val} Days",
                        str(workload_intensity_val),
                        f"{temperature_val} °C",
                        f"{noise_level_val} dB",
                        str(int(error_count_val))
                    ]
                })
                st.dataframe(key_factors, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ═══════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Workforce Analytics & Fatigue Trends</div>", unsafe_allow_html=True)

    # Date parse
    dash_data = dataset.copy()
    dash_data["Shift_Date"] = pd.to_datetime(dash_data["Shift_Date"], errors="coerce")
    ds_start = dash_data["Shift_Date"].min()
    ds_end   = dash_data["Shift_Date"].max()

    # ── Filters on Top ────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        from_date = st.date_input(
            "From Date",
            value=ds_start.date(),
            min_value=ds_start.date(),
            max_value=ds_end.date()
        )
    with fc2:
        to_date = st.date_input(
            "To Date",
            value=ds_end.date(),
            min_value=ds_start.date(),
            max_value=ds_end.date()
        )
    with fc3:
        quick_filter = st.selectbox(
            "Quick Filter",
            ["All Time", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last 1 Year"]
        )

    # ── Apply quick filter ────────────────────────────────
    filtered = dash_data.copy()
    today_ts = filtered["Shift_Date"].max()

    if quick_filter == "Last 30 Days":
        from_date = (today_ts - pd.Timedelta(days=30)).date()
    elif quick_filter == "Last 90 Days":
        from_date = (today_ts - pd.Timedelta(days=90)).date()
    elif quick_filter == "Last 6 Months":
        from_date = (today_ts - pd.DateOffset(months=6)).date()
    elif quick_filter == "Last 1 Year":
        from_date = (today_ts - pd.DateOffset(years=1)).date()

    filtered = filtered[
        (filtered["Shift_Date"] >= pd.to_datetime(from_date)) &
        (filtered["Shift_Date"] <= pd.to_datetime(to_date))
    ]

    # ── KPI Cards ─────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    
    high_fatigue_rate = (filtered["Fatigue_Level"] == "High").mean() * 100
    avg_sleep = filtered["Sleep_Hours_Previous_Night"].mean()
    avg_stress = filtered["Stress_Level"].mean()
    total_errors = filtered["Error_Count"].sum()

    with k1:
        st.metric("Total Shifts Evaluated", f"{len(filtered):,}")
    with k2:
        st.metric("High Fatigue Rate", f"{high_fatigue_rate:.1f}%")
    with k3:
        st.metric("Avg Sleep Duration", f"{avg_sleep:.1f} Hrs")
    with k4:
        st.metric("Avg Stress Level", f"{avg_stress:.1f} / 10")
    with k5:
        st.metric("Total Errors Recorded", f"{int(total_errors):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────
    CHART_LAYOUT = dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        title_font=dict(size=18, color="#111827"),
        font=dict(color="#374151")
    )
    
    FATIGUE_COLORS = {
        'Low': '#10b981',     # Emerald Green
        'Medium': '#f59e0b',  # Amber Yellow
        'High': '#ef4444'     # Rose Red
    }

    r1c1, r1c2 = st.columns(2)

    # 1. Monthly Fatigue Incident Rate Trend
    with r1c1:
        monthly_trend = (
            filtered.groupby(filtered["Shift_Date"].dt.to_period("M"))
            .apply(lambda x: (x["Fatigue_Level"] == "High").mean() * 100)
            .reset_index(name="High_Fatigue_Rate")
        )
        monthly_trend["Shift_Date"] = monthly_trend["Shift_Date"].dt.to_timestamp()
        monthly_trend["High_Fatigue_Rate"] = monthly_trend["High_Fatigue_Rate"].round(1)

        fig1 = px.line(
            monthly_trend, x="Shift_Date", y="High_Fatigue_Rate",
            markers=True,
            title="1. High Fatigue Incident Rate Trend (%)"
        )
        fig1.update_traces(
            mode="lines+markers+text",
            text=monthly_trend["High_Fatigue_Rate"],
            textposition="top center",
            line=dict(color="#1565ff", width=3),
            marker=dict(size=8, color="#1565ff"),
            textfont=dict(size=11, color="#111827")
        )
        fig1.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(title="Month", showgrid=False),
            yaxis=dict(title="High Fatigue Rate (%)", gridcolor="#E5E7EB")
        )
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    # 2. Fatigue Levels by Department
    with r1c2:
        dept_fatigue = (
            filtered.groupby(["Department", "Fatigue_Level"])
            .size()
            .reset_index(name="Shifts")
        )
        # Sort values for visual representation
        dept_fatigue["Fatigue_Level"] = pd.Categorical(
            dept_fatigue["Fatigue_Level"], 
            categories=['Low', 'Medium', 'High'], 
            ordered=True
        )
        dept_fatigue = dept_fatigue.sort_values("Fatigue_Level")

        fig2 = px.bar(
            dept_fatigue, x="Department", y="Shifts",
            color="Fatigue_Level",
            title="2. Fatigue Level Counts by Department",
            color_discrete_map=FATIGUE_COLORS,
            barmode="stack"
        )
        fig2.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(title="Department"),
            yaxis=dict(title="Number of Shifts", gridcolor="#E5E7EB"),
            legend=dict(title="Fatigue Level", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── ROW 2 ─────────────────────────────────────────────
    r2c1, r2c2, r2c3 = st.columns(3)

    # 3. Shift Type vs Fatigue Level (Horizontal Bar)
    with r2c1:
        shift_fatigue = (
            filtered.groupby(["Shift_Type", "Fatigue_Level"])
            .size()
            .reset_index(name="Shifts")
        )
        shift_fatigue["Fatigue_Level"] = pd.Categorical(
            shift_fatigue["Fatigue_Level"], 
            categories=['Low', 'Medium', 'High'], 
            ordered=True
        )
        shift_fatigue = shift_fatigue.sort_values("Fatigue_Level")

        fig3 = px.bar(
            shift_fatigue, y="Shift_Type", x="Shifts",
            color="Fatigue_Level",
            orientation="h",
            color_discrete_map=FATIGUE_COLORS,
            title="3. Shift Type vs Fatigue Distribution",
            barmode="stack"
        )
        fig3.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(title="Number of Shifts", gridcolor="#E5E7EB"),
            yaxis=dict(title="Shift Type"),
            legend=dict(title="Fatigue Level", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # 4. Sleep Hours vs Stress Level (Scatter)
    with r2c2:
        # Sample 1000 points to prevent browser lag during plot rendering
        scatter_sample = filtered.dropna(subset=["Sleep_Hours_Previous_Night", "Stress_Level"]).sample(
            min(1000, len(filtered)), 
            random_state=42
        )
        fig4 = px.scatter(
            scatter_sample, x="Sleep_Hours_Previous_Night", y="Stress_Level",
            color="Fatigue_Level",
            color_discrete_map=FATIGUE_COLORS,
            opacity=0.6,
            title="4. Sleep Duration vs Stress Level"
        )
        fig4.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(title="Sleep Hours (Previous Night)", gridcolor="#E5E7EB"),
            yaxis=dict(title="Stress Level (1-10)", gridcolor="#E5E7EB"),
            legend=dict(title="Fatigue Level", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    # 5. Overtime Hours vs Fatigue Level
    with r2c3:
        # Group average overtime by Fatigue level
        ot_fatigue = (
            filtered.groupby("Fatigue_Level")["Overtime_Hours"]
            .mean()
            .reset_index(name="Avg_Overtime")
        )
        fig5 = px.bar(
            ot_fatigue, x="Fatigue_Level", y="Avg_Overtime",
            color="Fatigue_Level",
            color_discrete_map=FATIGUE_COLORS,
            title="5. Average Daily Overtime by Fatigue Level"
        )
        fig5.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(title="Fatigue Level"),
            yaxis=dict(title="Avg Overtime (Hours)", gridcolor="#E5E7EB"),
            showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════
# TAB 3 — BULK FATIGUE ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Bulk Fatigue Level Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280; font-size:14px;'>Run batch predictions on the 10,000-record test set.</p>", unsafe_allow_html=True)

    if st.button("Run Bulk Prediction", key="bulk_predict"):
        try:
            bulk_raw = test_data.copy()
            # Prepare for predictor (removing the label)
            bulk_input = bulk_raw.drop(columns=["Fatigue_Level"], errors="ignore")

            # Batch predict using AutoGluon predictor
            predictions = predictor.predict(bulk_input)

            # Build results dataframe
            results = bulk_raw.copy()
            results["Predicted_Fatigue_Level"] = predictions

            st.session_state["bulk_fatigue_results"] = results
            st.success(f"Bulk prediction completed for {len(results):,} shifts.")

        except Exception as e:
            st.error(f"Bulk prediction failed: {e}")

    # ── Display Bulk Results ──────────────────────────────
    if "bulk_fatigue_results" in st.session_state:
        results = st.session_state["bulk_fatigue_results"]

        # KPI Summary
        bk1, bk2, bk3, bk4 = st.columns(4)

        total_bk = len(results)
        high_bk = (results["Predicted_Fatigue_Level"] == "High").sum()
        med_bk = (results["Predicted_Fatigue_Level"] == "Medium").sum()
        low_bk = (results["Predicted_Fatigue_Level"] == "Low").sum()

        with bk1:
            st.metric("Total Shifts Analyzed", f"{total_bk:,}")
        with bk2:
            st.metric("High Fatigue Risk", f"{high_bk:,} ({100*high_bk/total_bk:.1f}%)")
        with bk3:
            st.metric("Medium Fatigue Risk", f"{med_bk:,} ({100*med_bk/total_bk:.1f}%)")
        with bk4:
            st.metric("Low Fatigue Risk", f"{low_bk:,} ({100*low_bk/total_bk:.1f}%)")

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk level filter
        risk_filter = st.selectbox(
            "Filter Results by Predicted Fatigue Risk",
            ["All", "Low", "Medium", "High"],
            key="fatigue_risk_filter"
        )

        display = (
            results if risk_filter == "All"
            else results[results["Predicted_Fatigue_Level"] == risk_filter]
        )

        st.info(f"Showing **{len(display):,}** records — Risk Category: **{risk_filter}**")

        # Table Display
        display_cols = [
            "Employee_ID", "Shift_Date", "Department", "Job_Role",
            "Shift_Type", "Sleep_Hours_Previous_Night", "Stress_Level",
            "Predicted_Fatigue_Level"
        ]
        # Keep only existing columns
        display_cols = [c for c in display_cols if c in display.columns]

        st.dataframe(
            display[display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        # Download Report CSV
        csv_data = display.to_csv(index=False)
        st.download_button(
            label="Download Fatigue Predictions Report",
            data=csv_data,
            file_name="employee_fatigue_predictions.csv",
            mime="text/csv"
        )
