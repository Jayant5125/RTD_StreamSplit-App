# app2.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson  # Use simpson instead of simps for compatibility
from skopt import gp_minimize
from skopt.space import Real

st.title("RTD Curve Fitter with Outlier Removal")

def rtd_model(t, c, N=100):
    sum_terms = np.zeros_like(t)
    for n in range(N):
        coef = (-1)**n * (2 * n + 1)
        exponent = (n + 0.5)**2 * np.pi**2 * (t - 0.5) * c
        exponent = np.clip(exponent, 0, 700)
        sum_terms += coef * np.exp(-exponent)
    return c * np.pi * sum_terms

def fit_model(df):
    x = df['Time (s)'].values
    y = df['Average'].values
    mask = (x >= 0.51) & (x <= 1.5)
    x_fit = x[mask]
    y_fit = y[mask]
    y_fit = y_fit / simpson(y_fit, x_fit)

    def objective(c_list):
        c = c_list[0]
        if c <= 0:
            return 1e6
        y_pred = rtd_model(x_fit, c)
        area_pred = simpson(y_pred, x_fit)
        if area_pred == 0:
            return 1e6
        y_pred /= area_pred
        return np.mean((y_fit - y_pred)**2)

    result = gp_minimize(objective, [Real(3.0, 4.5)], n_calls=30, random_state=0)
    best_c = result.x[0]

    x_dense = np.linspace(0.501, 1.5, 500)
    y_model = rtd_model(x_dense, best_c)
    y_model /= simpson(y_model, x_dense)

    return x_fit, y_fit, x_dense, y_model, best_c

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Load Excel file
    try:
        df_raw = pd.read_excel(uploaded_file, sheet_name='1')
    except:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except:
            st.error("Could not read the file.")
            st.stop()

    if 'Time (s)' not in df_raw.columns:
        st.error("File must contain 'Time (s)' column.")
        st.stop()

    # Extract columns 1 to 10 (excluding Time column)
    time_col = df_raw['Time (s)']
    value_cols = df_raw.columns[3:13]
    df_curves = df_raw[value_cols]

    # Plot all curves
    st.subheader("All Individual Curves")
    # Mask to t = 0.5 to 1.5
    mask = (time_col >= 0.5) & (time_col <= 1.5)
    masked_time = time_col[mask]
    df_curves_masked = df_curves.loc[mask]

    fig_all, ax_all = plt.subplots()
    for col in df_curves_masked.columns:
        ax_all.plot(masked_time, df_curves_masked[col], label=col)
    ax_all.set_xlabel("Time (s)")
    ax_all.set_ylabel("Signal")
    ax_all.set_title("Raw Curves (Zoomed to t = 0.5 to 1.5)")
    ax_all.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_all)


    # Select outliers
    outliers = st.multiselect("Select curves to exclude from averaging", df_curves.columns.tolist())

    # Filter and compute average
    df_filtered = df_curves.drop(columns=outliers)

    # Apply baseline correction + normalization to each column BEFORE averaging
    time_values = df_raw['Time (s)'].values
    processed_curves = []

    for col in df_filtered.columns:
        signal = df_filtered[col].copy()
        baseline = signal[time_values < 0.5].mean()
        signal = signal - baseline
        area = np.trapz(signal, time_values)
        if area != 0:
            signal = signal / area
        processed_curves.append(signal)
    

    st.subheader("All Individual Curves (Masked: 0.5s – 1.5s)")

# Reuse the same mask as before
    mask = (time_col >= 0.5) & (time_col <= 1.5)
    masked_time = time_col[mask]

    for col in df_curves.columns:
        with st.expander(f"Masked Curve: {col}"):
            fig_single_masked, ax_single_masked = plt.subplots()
            ax_single_masked.plot(masked_time, df_curves.loc[mask, col], label=col)
            ax_single_masked.set_xlabel("Time (s)")
            ax_single_masked.set_ylabel("Signal")
            ax_single_masked.set_title(f"{col} (Zoomed: 0.5 to 1.5 s)")
            ax_single_masked.legend()
            st.pyplot(fig_single_masked)


    # Take mean of processed curves
    df_raw['Average'] = np.mean(processed_curves, axis=0)


    st.subheader("Filtered Average Curve")
    fig_avg, ax_avg = plt.subplots()
    ax_avg.plot(time_col, df_raw['Average'], color='black', label='Filtered Average')
    ax_avg.set_xlabel("Time (s)")
    ax_avg.set_ylabel("Signal")
    ax_avg.legend()
    st.pyplot(fig_avg)

    # Masked filtered average
    st.subheader("Filtered Average Curve (Zoomed: 0.5s - 1.5s)")
    fig_avg_zoom, ax_avg_zoom = plt.subplots()
    ax_avg_zoom.plot(masked_time, df_raw.loc[mask, 'Average'], color='black', label='Filtered Average')
    ax_avg_zoom.set_xlabel("Time (s)")
    ax_avg_zoom.set_ylabel("Average Signal")
    ax_avg_zoom.set_title("Filtered Average (0.5 to 1.5 s)")
    ax_avg_zoom.legend()
    st.pyplot(fig_avg_zoom)


    # Fit and show final RTD model
    x_fit, y_fit, x_dense, y_model, best_c = fit_model(df_raw)

    st.subheader("RTD Curve Fit")
    fig_fit, ax_fit = plt.subplots()
    ax_fit.plot(x_fit, y_fit, '.', markersize=2, label='Experimental Data')
    ax_fit.plot(x_dense, y_model, 'r-', linewidth=2, label=f'Fitted RTD\nc = {best_c:.4f}')
    ax_fit.set_xlabel('Time (s)')
    ax_fit.set_ylabel('Normalized Flow')
    ax_fit.grid(True)
    ax_fit.legend()
    st.pyplot(fig_fit)

    st.success(f"✅ Optimal c value: {best_c:.5f}")
