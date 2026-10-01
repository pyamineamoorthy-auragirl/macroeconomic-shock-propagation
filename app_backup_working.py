import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import grangercausalitytests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Macroeconomic Shock Propagation",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "var_model.pkl"
VARIABLE_PATH = BASE_DIR / "models" / "variable_names.pkl"
DATA_PATH = BASE_DIR / "data" / "macroeconomic_processed.csv"


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    st.error(f"VAR model not found:\n{MODEL_PATH}")
    st.stop()

if not VARIABLE_PATH.exists():
    st.error(f"Variable names file not found:\n{VARIABLE_PATH}")
    st.stop()


try:
    var_model = joblib.load(MODEL_PATH)
    variables = joblib.load(VARIABLE_PATH)

except Exception as e:
    st.error(f"Error loading VAR model: {e}")
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    st.error(
        f"""
        Macroeconomic dataset not found.

        Expected:
        {DATA_PATH}
        """
    )
    st.stop()


data = pd.read_csv(DATA_PATH)

data["Date"] = pd.to_datetime(
    data["Date"],
    errors="coerce"
)

data = data.dropna(subset=["Date"])

data = data.sort_values("Date")


# ============================================================
# CREATE MODEL DATA
# ============================================================

model_data = data.copy()

model_data = model_data.set_index("Date")


# Same transformations used during VAR development

model_data["CPI_Inflation"] = (
    model_data["CPI_Inflation"].diff()
)

model_data["Policy_Rate"] = (
    model_data["Policy_Rate"].diff()
)

model_data["USD_INR"] = (
    model_data["USD_INR"].diff()
)

model_data["Crude_Oil_Price"] = (
    model_data["Crude_Oil_Price"].diff()
)

model_data = model_data.dropna()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Multivariate Macroeconomic Shock Propagation Modeling"
)

st.markdown(
    """
    **Advanced VAR-based macroeconomic shock propagation analysis**

    Analyze how shocks propagate across macroeconomic variables
    using VAR, IRF, FEVD and Granger causality.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Analysis Modules")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Overview",
        "📈 Data Analysis",
        "🤖 VAR Model",
        "💥 Impulse Response",
        "📊 FEVD",
        "🔗 Granger Causality",
        "🌐 Shock Propagation"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("🏠 Project Overview")

    st.write(
        """
        This project studies how macroeconomic shocks propagate
        through a multivariate system over time.

        The analysis focuses on:

        • CPI Inflation
        • Policy Rate
        • USD/INR
        • Crude Oil Price
        • Industrial Production
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Observations",
            len(data)
        )

    with col2:
        st.metric(
            "Variables",
            len(variables)
        )

    with col3:
        st.metric(
            "Model",
            "VAR(1)"
        )

    with col4:
        st.metric(
            "Analysis Horizon",
            "12 Months"
        )

    st.subheader("🔬 Methodology")

    methodology = pd.DataFrame({
        "Method": [
            "Data Collection",
            "EDA",
            "ADF / KPSS",
            "VAR(1)",
            "Granger Causality",
            "Impulse Response",
            "FEVD",
            "Shock Propagation"
        ],
        "Status": [
            "✅ Completed",
            "✅ Completed",
            "✅ Completed",
            "✅ Completed",
            "🚀 Integrated",
            "🚀 Integrated",
            "🚀 Integrated",
            "🚀 Integrated"
        ]
    })

    st.dataframe(
        methodology,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📈 Data Analysis":

    st.header("📈 Macroeconomic Data Analysis")

    selected_variable = st.selectbox(
        "Select Variable",
        variables
    )

    st.subheader(
        f"📈 {selected_variable}"
    )

    chart_data = data[
        ["Date", selected_variable]
    ].set_index("Date")

    st.line_chart(
        chart_data,
        use_container_width=True
    )

    st.subheader("📋 Summary Statistics")

    st.dataframe(
        data[variables].describe().round(4),
        use_container_width=True
    )

    st.subheader("🔗 Correlation Matrix")

    correlation = data[variables].corr()

    st.dataframe(
        correlation.round(3),
        use_container_width=True
    )


# ============================================================
# VAR MODEL
# ============================================================

elif page == "🤖 VAR Model":

    st.header("🤖 VAR(1) Model")

    st.success(
        "✅ VAR(1) model loaded successfully!"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "VAR(1)"
        )

    with col2:
        st.metric(
            "Variables",
            len(variables)
        )

    with col3:

        nobs = getattr(
            var_model,
            "nobs",
            "N/A"
        )

        st.metric(
            "Model Observations",
            nobs
        )

    st.subheader("📌 Variables")

    for variable in variables:
        st.write(f"• {variable}")

    st.subheader("🔬 Model Stability")

    try:

        if var_model.is_stable():

            st.success(
                "✅ VAR(1) model is stable."
            )

        else:

            st.warning(
                "⚠️ VAR(1) model is not stable."
            )

    except Exception as e:

        st.warning(
            f"Could not determine stability: {e}"
        )

    with st.expander("📄 View Full VAR Summary"):

        st.text(
            str(var_model.summary())
        )


# ============================================================
# IMPULSE RESPONSE FUNCTION
# ============================================================

elif page == "💥 Impulse Response":

    st.header("💥 Impulse Response Function (IRF)")

    st.write(
        """
        IRF measures how a shock to one variable affects the
        other variables over subsequent months.
        """
    )

    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=24,
        value=12
    )

    shock_variable = st.selectbox(
        "Select Shock Variable",
        variables
    )

    # Calculate IRF

    irf = var_model.irf(horizon)

    irf_values = irf.irfs

    shock_index = variables.index(
        shock_variable
    )

    st.subheader(
        f"📈 Responses to a Shock in {shock_variable}"
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for response_index, response_variable in enumerate(variables):

        if response_variable == shock_variable:
            continue

        response = irf_values[
            :,
            response_index,
            shock_index
        ]

        ax.plot(
            range(len(response)),
            response,
            marker="o",
            label=response_variable
        )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "Months After Shock"
    )

    ax.set_ylabel(
        "Response"
    )

    ax.set_title(
        f"Impulse Responses to {shock_variable} Shock"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    st.pyplot(fig)

    st.subheader("📋 IRF Interpretation")

    st.info(
        """
        A positive response indicates that the affected
        variable increases following the shock.

        A negative response indicates that the affected
        variable decreases following the shock.

        The magnitude and persistence of the response indicate
        the strength and duration of the propagation effect.
        """
    )


# ============================================================
# FEVD
# ============================================================

elif page == "📊 FEVD":

    st.header(
        "📊 Forecast Error Variance Decomposition"
    )

    st.write(
        """
        FEVD shows how much of the forecast uncertainty of
        each variable is explained by shocks to the different
        variables in the VAR system.
        """
    )

    horizon = st.slider(
        "FEVD Horizon",
        min_value=1,
        max_value=24,
        value=12
    )

    selected_variable = st.selectbox(
        "Select Variable",
        variables
    )

    selected_index = variables.index(
        selected_variable
    )

    fevd = var_model.fevd(horizon)

    fevd_values = fevd.decomp[
        selected_index
    ]

    horizons = range(
        1,
        horizon + 1
    )

    fevd_table = pd.DataFrame(
        fevd_values,
        columns=variables,
        index=horizons
    )

    fevd_table.index.name = "Horizon"

    st.subheader(
        f"📊 Variance Decomposition: {selected_variable}"
    )

    st.dataframe(
        fevd_table.round(4),
        use_container_width=True
    )

    st.subheader("📈 FEVD Visualization")

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bottom = np.zeros(
        len(fevd_table)
    )

    for variable in variables:

        values = fevd_table[
            variable
        ].values

        ax.bar(
            fevd_table.index,
            values,
            bottom=bottom,
            label=variable
        )

        bottom += values

    ax.set_xlabel(
        "Forecast Horizon (Months)"
    )

    ax.set_ylabel(
        "Proportion of Forecast Error"
    )

    ax.set_title(
        f"FEVD of {selected_variable}"
    )

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# GRANGER CAUSALITY
# ============================================================

elif page == "🔗 Granger Causality":

    st.header("🔗 Granger Causality Analysis")

    st.write(
        """
        Granger causality tests whether the past values of one
        macroeconomic variable provide useful predictive information
        about another variable.
        """
    )

    from statsmodels.tsa.stattools import grangercausalitytests

    granger_results = []

    for cause in variables:

        for effect in variables:

            # Skip same variable
            if cause == effect:
                continue

            # Data order must be [effect, cause]
            test_data = model_data[
                [effect, cause]
            ].dropna()

            try:

                result = grangercausalitytests(
                    test_data,
                    maxlag=1
                )

                # Extract SSR F-test p-value
                p_value = result[1][0]["ssr_ftest"][1]

                if p_value < 0.05:
                    significance = "Significant"
                else:
                    significance = "Not Significant"

                granger_results.append({
                    "Cause": cause,
                    "Effect": effect,
                    "Lag": 1,
                    "p_value": p_value,
                    "Result": significance
                })

            except Exception as e:

                granger_results.append({
                    "Cause": cause,
                    "Effect": effect,
                    "Lag": 1,
                    "p_value": np.nan,
                    "Result": "Error"
                })

    # Convert results to DataFrame

    granger_df = pd.DataFrame(
        granger_results
    )

    # Sort by p-value

    granger_df = granger_df.sort_values(
        by="p_value",
        na_position="last"
    )

    # Display results

    st.subheader("📊 Granger Causality Results")

    st.dataframe(
        granger_df.round(5),
        use_container_width=True,
        hide_index=True
    )

    # Significant relationships

    significant_df = granger_df[
        granger_df["p_value"] < 0.05
    ]

    st.subheader(
        "🔍 Significant Predictive Relationships"
    )

    if len(significant_df) > 0:

        st.dataframe(
            significant_df.round(5),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No statistically significant Granger relationships were found at the 5% level."
        )

    st.info(
        """
        Interpretation:
        
        p-value < 0.05 → statistically significant predictive relationship.
        
        p-value ≥ 0.05 → insufficient evidence of predictive relationship.
        
        Granger causality indicates predictive usefulness of past values;
        it does not by itself prove real-world causal influence.
        """
    )


# ============================================================
# SHOCK PROPAGATION
# ============================================================

elif page == "🌐 Shock Propagation":

    st.header("🌐 Shock Propagation Analysis")

    st.write(
        """
        This module summarizes how shocks propagate from one
        macroeconomic variable to other variables over time.
        """
    )

    horizon = st.slider(
        "Propagation Horizon (Months)",
        min_value=1,
        max_value=24,
        value=12
    )

    # Calculate IRF
    irf = var_model.irf(horizon)

    irf_values = irf.irfs

    propagation_results = []

    # ----------------------------------------
    # Calculate propagation
    # ----------------------------------------

    for shock_index, shock in enumerate(variables):

        for response_index, response in enumerate(variables):

            # Skip own response
            if shock == response:
                continue

            values = irf_values[
                :,
                response_index,
                shock_index
            ]

            # Find maximum absolute response
            peak_index = int(
                np.argmax(
                    np.abs(values)
                )
            )

            peak_value = float(
                values[peak_index]
            )

            # Determine direction
            if peak_value > 0:
                direction = "Positive"
            elif peak_value < 0:
                direction = "Negative"
            else:
                direction = "Neutral"

            # Store result
            propagation_results.append({
                "Shock": shock,
                "Response": response,
                "Peak_Response": peak_value,
                "Peak_Month": peak_index,
                "Direction": direction
            })

    # ----------------------------------------
    # Create DataFrame
    # ----------------------------------------

    propagation_df = pd.DataFrame(
        propagation_results
    )

    # ----------------------------------------
    # Sort by strongest response
    # ----------------------------------------

    propagation_df["Absolute_Response"] = (
        propagation_df["Peak_Response"].abs()
    )

    propagation_df = propagation_df.sort_values(
        "Absolute_Response",
        ascending=False
    )

    propagation_df = propagation_df.drop(
        columns=["Absolute_Response"]
    )

    # ----------------------------------------
    # Display complete table
    # ----------------------------------------

    st.subheader(
        "📋 Shock Propagation Summary"
    )

    st.dataframe(
        propagation_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------------
    # Select individual shock
    # ----------------------------------------

    st.subheader(
        "🎯 Analyze Individual Shock"
    )

    selected_shock = st.selectbox(
        "Select Shock Variable",
        variables
    )

    selected_results = propagation_df[
        propagation_df["Shock"] == selected_shock
    ]

    st.dataframe(
        selected_results.round(4),
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------------
    # Visualize selected shock
    # ----------------------------------------

    st.subheader(
        f"📈 Propagation of {selected_shock} Shock"
    )

    shock_index = variables.index(
        selected_shock
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for response_index, response in enumerate(variables):

        if response == selected_shock:
            continue

        response_values = irf_values[
            :,
            response_index,
            shock_index
        ]

        ax.plot(
            range(len(response_values)),
            response_values,
            marker="o",
            label=response
        )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "Months After Shock"
    )

    ax.set_ylabel(
        "Response"
    )

    ax.set_title(
        f"Shock Propagation: {selected_shock}"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    # ----------------------------------------
    # Explanation
    # ----------------------------------------

    st.info(
        """
        Peak Response represents the largest absolute response
        observed during the selected forecast horizon.

        Peak Month represents the month in which that response
        occurs.

        Positive indicates an increase in the responding variable,
        while Negative indicates a decrease.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Multivariate Macroeconomic Shock Propagation Modeling | "
    "VAR(1) | IRF | FEVD | Granger Causality"
)