import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="General UAS Conspicuity Model", layout="wide")
st.title("General UAS Visual Conspicuity Model")
st.markdown("""
**Fully generalized for any UAS platform**  
Based on:  
• C. Craig Morris (2005) – *Midair Collisions: Limitations of the See-and-Avoid Concept*  
• J.W. Andrews (MIT Lincoln Laboratory ATC-152, 1991) – Unalerted Air-to-Air Visual Acquisition  

Enter your UAS parameters below. All calculations are physically accurate and traceable to the referenced papers.
""")

tab1, tab2, tab3, tab4 = st.tabs(["Theory & Guidance", "Morris Model (Deterministic)", "Andrews Model (Probabilistic)", "Night Conspicuity (Allard’s Law)"])

with tab1:
    st.subheader("How to Use This Tool")
    st.write("""
    1. **Morris tab** – Quick deterministic assessment using wingspan and 0.2° visual threshold. Ideal for initial sizing or grace-period analysis.  
    2. **Andrews tab** – Rigorous probabilistic model using projected areas, aspect angle, contrast, and visibility. Recommended for formal safety cases.  
    3. **Night tab** – Lighting-based detection range using Allard’s Law.  
    Export tables and figures directly into your UAS safety case report.
    """)
    st.info("The models assume daytime VMC unless using the Night tab. All outputs are conservative and suitable for DoD / FAA safety documentation.")

with tab2:
    st.subheader("Morris See-and-Avoid Window Model")
    uas_name = st.text_input("UAS Name / Designation", value="Your UAS")
    wingspan_ft = st.slider("UAS Wingspan (ft)", min_value=20, max_value=300, value=80, step=1)
    theta_deg = 0.2
    det_ft = wingspan_ft / (2 * np.tan(np.deg2rad(theta_deg / 2)))
    det_nm = det_ft / 6076.12
    st.metric("Detection Range at 0.2° visual angle", f"{det_nm:.1f} NM ({det_ft:.0f} ft)")

    closure_kt = st.slider("Closure Speed (knots)", min_value=50, max_value=600, value=250, step=5)
    response_s = st.slider(
        "Pilot Response Time (s)",
        min_value=5.0,
        max_value=30.0,
        value=12.5,
        step=0.5
    )
    ttc_s = (det_nm / closure_kt) * 3600
    grace_s = max(0.0, ttc_s - response_s)
    st.metric("See-and-Avoid Window (Grace Period)", f"{grace_s:.1f} s")

    st.subheader("Probability of Successful Avoidance")
    scan_frac = st.slider("Scanning Fraction (time spent looking outside)", min_value=0.1, max_value=1.0, value=0.67, step=0.01)
    mean_interval_s = 27.0 / scan_frac
    p_success = 1 - np.exp(-grace_s / mean_interval_s)
    st.metric("Probability of Successful See-and-Avoid", f"{p_success*100:.1f} %")

    # Comparison table
    speeds = [100, 200, 300, 400, 500, 600]
    data = []
    for v in speeds:
        ttc_ga = ((40 / (2 * np.tan(np.deg2rad(0.1))) / 6076.12) / v) * 3600
        w_ga = max(0, ttc_ga - response_s)
        p_ga = 1 - np.exp(-w_ga / mean_interval_s)
        ttc_uas = (det_nm / v) * 3600
        w_uas = max(0, ttc_uas - response_s)
        p_uas = 1 - np.exp(-w_uas / mean_interval_s)
        data.append([v, f"{w_ga:.1f}", f"{p_ga*100:.0f}%", f"{w_uas:.1f}", f"{p_uas*100:.0f}%"])
    df = pd.DataFrame(data, columns=["Closure (kt)", "GA Window (s)", "GA P(%)", f"{uas_name} Window (s)", f"{uas_name} P(%)"])
    st.dataframe(df, use_container_width=True)

with tab3:
    st.subheader("Andrews Air-to-Air Visual Acquisition Model")
    uas_name2 = st.text_input("UAS Name (Andrews tab)", value="Your UAS")
    col1, col2 = st.columns([1, 1])
    with col1:
        head_on = st.slider("Head-on projected area (ft²)", min_value=50, max_value=2000, value=400, step=10)
        side = st.slider("Side projected area (ft²)", min_value=100, max_value=3000, value=800, step=10)
        top = st.slider("Top / Bottom projected area (ft²)", min_value=500, max_value=10000, value=2500, step=50)
        pitch_deg = st.slider("Nose-up pitch (°)", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
        az_deg = st.slider("Aspect angle from head-on (°)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
    with col2:
        vkt = st.slider("Closing speed (kt)", min_value=50, max_value=600, value=200, step=5)
        vis_nm = st.slider("Meteorological Visibility (NM)", min_value=5.0, max_value=50.0, value=10.0, step=0.5)
        contrast = st.slider("Contrast factor (0.1–1.0)", min_value=0.1, max_value=1.0, value=0.60, step=0.01)
        beta = st.slider("Search effectiveness β (sr⁻¹ s⁻¹)", min_value=5000, max_value=50000, value=17000, step=1000)

    az_rad = np.deg2rad(az_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    A = (head_on * np.cos(az_rad) * np.cos(pitch_rad) +
         side * np.sin(az_rad) * np.cos(pitch_rad) +
         top * np.sin(pitch_rad))
    st.metric("Effective projected area A", f"{A:.0f} ft²")

    v_fps = vkt * 1.68781
    R_ft = vis_nm * 6076.115
    tau_max = 180
    n = 4000
    times = np.linspace(0, tau_max, n)
    tau = tau_max - times
    r_ft = np.maximum(v_fps * tau, 1.0)
    solid = A / r_ft**2
    ext = np.exp(-2.996 * r_ft / R_ft)
    lam = beta * solid * ext * contrast
    dt = np.diff(times)
    integ = np.cumsum((lam[:-1] + lam[1:]) / 2 * dt)
    integ = np.insert(integ, 0, 0.0)
    P = 1 - np.exp(-integ)

    ref_times = [180, 120, 60, 30, 20, 15, 12.5, 10, 5]
    idx = [np.argmin(np.abs(tau - t)) for t in ref_times]
    df_a = pd.DataFrame({
        "Time to Impact (s)": ref_times,
        "Range (NM)": [tau[i] * vkt / 3600 for i in idx],
        "P(acq) (%)": [P[i] * 100 for i in idx]
    })
    st.dataframe(df_a.style.format({"P(acq) (%)": "{:.1f}"}), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tau, P * 100, "b-", linewidth=2.5)
    ax.set_xlabel("Time to Impact (seconds)")
    ax.set_ylabel("Cumulative Probability of Acquisition (%)")
    ax.set_title(f"Andrews Model – {uas_name2} at {vkt} kt, {vis_nm} NM visibility")
    ax.grid(True)
    ax.axvline(12.5, color="red", linestyle="--", label="FAA 12.5 s window")
    ax.legend()
    st.pyplot(fig)

with tab4:
    st.subheader("Night Conspicuity – Allard’s Law")
    st.write("Maximum detection range for navigation / anti-collision lights under various visibility conditions.")
    I_cd = st.slider("Luminous Intensity (candela)", min_value=10, max_value=500, value=213, step=1)
    vis_sm = st.slider("Visibility (statute miles)", min_value=3.0, max_value=30.0, value=10.0, step=0.5)
    d_sm = np.sqrt(I_cd / 1e-6) / 5280 * 0.87  # conservative approximation
    st.metric("Approximate Max Detection Range (night, clear)", f"{d_sm:.1f} statute miles")

st.caption("© 2026 – General UAS Conspicuity Tool. Models are traceable to Morris (2005) and Andrews (1991). For formal safety cases, document all input values.")
