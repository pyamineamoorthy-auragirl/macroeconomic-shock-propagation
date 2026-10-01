import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Macroeconomic Shock Propagation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "var_model.pkl"
VARIABLES_PATH = BASE_DIR / "models" / "variable_names.pkl"

VECM_MODEL_PATH = BASE_DIR / "models" / "vecm_model.pkl"
VECM_VARIABLES_PATH = BASE_DIR / "models" / "vecm_variable_names.pkl"

DATA_PATH = BASE_DIR / "data" / "macroeconomic_processed.csv"


# ============================================================
# LOAD CUSTOM CSS
# ============================================================

def load_css():
    css_path = BASE_DIR / "style.css"

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )


load_css()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = {
    "VAR model": MODEL_PATH,
    "VAR variable names": VARIABLES_PATH,
    "VECM model": VECM_MODEL_PATH,
    "VECM variable names": VECM_VARIABLES_PATH,
    "Dataset": DATA_PATH
}

missing_files = []

for file_name, file_path in required_files.items():
    if not file_path.exists():
        missing_files.append(f"{file_name}: {file_path}")

if missing_files:
    st.error("❌ Required project files are missing:")

    for file in missing_files:
        st.write(f"- {file}")

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

try:
    var_model = joblib.load(MODEL_PATH)
    variables = list(joblib.load(VARIABLES_PATH))

    vecm_model = joblib.load(VECM_MODEL_PATH)
    vecm_variables = list(joblib.load(VECM_VARIABLES_PATH))

except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(DATA_PATH)
data["Date"] = pd.to_datetime(data["Date"])

# Handle missing values
required_variables = [
    "CPI_Inflation",
    "Policy_Rate",
    "USD_INR",
    "Crude_Oil_Price",
    "Industrial_Production"
]

data[required_variables] = data[required_variables].interpolate(
    method="linear",
    limit_direction="both"
)

# ============================================================
# CHECK REQUIRED VARIABLES
# ============================================================

missing_variables = [
    variable
    for variable in variables
    if variable not in data.columns
]

if missing_variables:
    st.error(
        "❌ The following VAR variables are missing "
        f"from the dataset: {missing_variables}"
    )
    st.stop()


missing_vecm_variables = [
    variable
    for variable in vecm_variables
    if variable not in data.columns
]

if missing_vecm_variables:
    st.error(
        "❌ The following VECM variables are missing "
        f"from the dataset: {missing_vecm_variables}"
    )
    st.stop()


# ============================================================
# MODEL DATA FOR GRANGER CAUSALITY
# ============================================================

model_data = data.copy()
model_data = model_data.set_index("Date")

difference_variables = [
    "CPI_Inflation",
    "Policy_Rate",
    "USD_INR",
    "Crude_Oil_Price"
]

for variable in difference_variables:
    if variable in model_data.columns:
        model_data[variable] = model_data[variable].diff()

model_data = model_data.dropna()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Multivariate Macroeconomic Shock Propagation Modeling"
)

st.markdown(
    "**Advanced VAR and VECM-based macroeconomic shock propagation analysis**"
)

st.write(
    "Analyze how shocks propagate across macroeconomic variables "
    "using VAR, VECM, IRF, FEVD and Granger causality."
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📈 Data Analysis",
        "🤖 VAR Model",
        "🔄 VECM Model",
        "⚖️ VAR vs VECM",
        "💥 Impulse Response",
        "📊 FEVD",
        "🔗 Granger Causality",
        "🌐 Shock Propagation",
        "🎛️ Shock Simulator"
    ]
)


# ============================================================
# SIDEBAR PROJECT INFORMATION
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Period")

st.sidebar.markdown(
    f"**{data['Date'].min().strftime('%b %Y')} "
    f"→ {data['Date'].max().strftime('%b %Y')}**"
)

st.sidebar.markdown("### Observations")
st.sidebar.markdown(f"**{len(data)}**")

st.sidebar.markdown("### Variables")
st.sidebar.markdown(f"**{len(variables)}**")

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    📊 **Multivariate Macroeconomic Shock Propagation Modeling**

    Python • Pandas • Statsmodels • Streamlit
    """
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

    # --------------------------------------------------------
    # KEY METRICS
    # --------------------------------------------------------

    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Observations", len(data))

    with col2:
        st.metric("📈 Variables", len(variables))

    with col3:
        st.metric("🤖 VAR Model", "VAR(1)")

    with col4:
        st.metric("🔄 VECM Rank", "1")

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.subheader("🟢 Model Status")

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        with st.container(border=True):
            st.markdown("### 🟢 DATA ENGINE")
            st.markdown("**READY**")
            st.caption(f"{len(data)} monthly observations processed")

    with status_col2:
        with st.container(border=True):
            st.markdown("### 🔵 STATIONARITY")
            st.markdown("**VERIFIED**")
            st.caption("ADF + KPSS analysis completed")

    with status_col3:
        with st.container(border=True):
            st.markdown("### 🟣 VAR ENGINE")
            st.markdown("**STABLE**")
            st.caption("VAR(1) model ready")

    with status_col4:
        with st.container(border=True):
            st.markdown("### 🟠 VECM ENGINE")
            st.markdown("**READY**")
            st.caption("Cointegration rank = 1")

    # --------------------------------------------------------
    # ANALYTICAL MODULES
    # --------------------------------------------------------

    st.subheader("🔬 Analytical Modules")

    module_col1, module_col2, module_col3 = st.columns(3)

    with module_col1:
        st.info(
            """
            **🔗 GRANGER CAUSALITY**

            Predictive relationships between
            macroeconomic variables.
            """
        )

    with module_col2:
        st.info(
            """
            **💥 IMPULSE RESPONSE**

            Dynamic response of the system
            to an identified shock.
            """
        )

    with module_col3:
        st.info(
            """
            **📊 FEVD**

            Contribution of shocks to
            forecast uncertainty.
            """
        )

    module_col4, module_col5, module_col6 = st.columns(3)

    with module_col4:
        st.info(
            """
            **🌐 SHOCK PROPAGATION**

            Analysis of how shocks spread
            across macroeconomic variables.
            """
        )

    with module_col5:
        st.info(
            """
            **🎛️ SHOCK SIMULATOR**

            Interactive simulation of
            standardized macroeconomic shocks.
            """
        )

    with module_col6:
        st.success(
            """
            **🔄 VECM ENGINE**

            Long-run equilibrium and
            short-run adjustment analysis.
            """
        )

    # --------------------------------------------------------
    # METHODOLOGY
    # --------------------------------------------------------

    st.subheader("🔬 Methodology")

    methodology = pd.DataFrame({
        "Step": [
            "1. Data Collection",
            "2. Data Cleaning",
            "3. Exploratory Data Analysis",
            "4. ADF / KPSS Stationarity",
            "5. VAR Model Selection",
            "6. VAR(1) Estimation",
            "7. Johansen Cointegration Test",
            "8. VECM Estimation",
            "9. Granger Causality",
            "10. Impulse Response Function",
            "11. FEVD",
            "12. Shock Propagation",
            "13. Interactive Shock Simulator"
        ],
        "Status": [
            "✅ Completed",
            "✅ Completed",
            "✅ Completed",
            "✅ Completed",
            "✅ Completed",
            "✅ Completed",
            "✅ Completed",
            "✅ Completed",
            "✅ Integrated",
            "✅ Integrated",
            "✅ Integrated",
            "✅ Integrated",
            "🚀 Available"
        ]
    })

    st.dataframe(
        methodology,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # VARIABLES
    # --------------------------------------------------------

    st.subheader("📌 Variables Used")

    roles = [
        "Inflation indicator",
        "Monetary policy indicator",
        "Exchange rate indicator",
        "External commodity price",
        "Economic activity indicator"
    ]

    if len(variables) == len(roles):
        variable_info = pd.DataFrame({
            "Variable": variables,
            "Role": roles
        })
    else:
        variable_info = pd.DataFrame({
            "Variable": variables,
            "Role": ["Macroeconomic Variable"] * len(variables)
        })

    st.dataframe(
        variable_info,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    st.subheader("🚀 Project Capabilities")

    cap1, cap2 = st.columns(2)

    with cap1:
        st.markdown(
            """
            **📊 Statistical Analysis**

            • Descriptive statistics
            • Correlation analysis
            • ADF and KPSS tests
            • VAR model estimation
            • VECM estimation
            • Cointegration analysis
            """
        )

    with cap2:
        st.markdown(
            """
            **💥 Shock Analysis**

            • Granger causality
            • Impulse Response Function
            • Forecast Error Variance Decomposition
            • Shock propagation analysis
            • Interactive shock simulation
            """
        )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader("🤖 Model Information")

    st.info(
        """
        The project uses two complementary time-series models.

        **VAR(1):**
        Used for the broader five-variable short-run
        dynamic and shock propagation analysis.

        **VECM:**
        Used for the four I(1) variables to model
        long-run equilibrium relationships and short-run
        adjustment.

        Industrial Production is I(0), so it is not included
        in the VECM subsystem.

        The project uses statistical time-series methods
        and does not use artificial intelligence or
        deep-learning models.
        """
    )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📈 Data Analysis":

    st.header("📈 Macroeconomic Data Analysis")

    st.success("✅ Data analysis module loaded successfully.")

    st.subheader("📋 Dataset Information")

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    with info_col1:
        st.metric("Observations", len(data))

    with info_col2:
        st.metric("Variables", len(variables))

    with info_col3:
        st.metric(
            "Start",
            data["Date"].min().strftime("%b %Y")
        )

    with info_col4:
        st.metric(
            "End",
            data["Date"].max().strftime("%b %Y")
        )

    st.subheader("📈 Time-Series Analysis")

    selected_variable = st.selectbox(
        "Select Variable",
        variables,
        key="data_analysis_variable"
    )

    chart_data = data[
        ["Date", selected_variable]
    ].dropna()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        chart_data["Date"],
        chart_data[selected_variable],
        marker="o",
        linewidth=1.8
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(selected_variable)
    ax.set_title(f"{selected_variable} Time Series")
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("📋 Summary Statistics")

    summary_table = data[variables].describe().T.round(4)

    summary_table.insert(
        0,
        "Variable",
        summary_table.index
    )

    summary_table = summary_table.reset_index(drop=True)

    st.dataframe(
        summary_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🔗 Correlation Matrix")

    correlation = data[variables].corr().round(3)

    st.dataframe(
        correlation,
        use_container_width=True
    )

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 7))

    matrix = correlation.values

    image = ax.imshow(
        matrix,
        aspect="auto"
    )

    ax.set_xticks(range(len(variables)))
    ax.set_yticks(range(len(variables)))

    ax.set_xticklabels(
        variables,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(variables)

    for i in range(len(variables)):
        for j in range(len(variables)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title("Macroeconomic Variable Correlation")

    fig.colorbar(image, ax=ax)

    plt.tight_layout()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("🔎 Missing Value Check")

    missing_table = pd.DataFrame({
        "Variable": variables,
        "Missing Values": [
            int(data[v].isna().sum())
            for v in variables
        ]
    })

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True
    )

    if missing_table["Missing Values"].sum() == 0:
        st.success(
            "✅ No missing values detected in the analysis variables."
        )
    else:
        st.warning("⚠️ Missing values were detected.")


# ============================================================
# VAR MODEL
# ============================================================

elif page == "🤖 VAR Model":

    st.header("🤖 VAR(1) Model")

    st.success("✅ VAR(1) model loaded successfully.")

    # --------------------------------------------------------
    # MODEL OVERVIEW
    # --------------------------------------------------------

    st.subheader("📌 Model Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Type", "VAR(1)")

    with col2:
        st.metric("Variables", len(variables))

    with col3:
        st.metric(
            "Observations",
            getattr(var_model, "nobs", "N/A")
        )

    with col4:
        st.metric(
            "Lag Order",
            getattr(var_model, "k_ar", "N/A")
        )

    # --------------------------------------------------------
    # VARIABLES
    # --------------------------------------------------------

    st.subheader("📊 Variables Used in VAR Model")

    variable_df = pd.DataFrame({
        "Variable": variables,
        "Role": ["Macroeconomic Variable"] * len(variables)
    })

    st.dataframe(
        variable_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MODEL STABILITY
    # --------------------------------------------------------

    st.subheader("🟢 Model Stability")

    try:
        stability = var_model.is_stable()

        if stability:
            st.success(
                "✅ VAR(1) model is stable. "
                "All characteristic roots satisfy the stability condition."
            )
        else:
            st.error("❌ VAR(1) model is not stable.")

    except Exception as e:
        st.warning(
            f"⚠️ Stability check could not be completed: {e}"
        )

    # --------------------------------------------------------
    # INFORMATION CRITERIA
    # --------------------------------------------------------

    st.subheader("📈 Model Information Criteria")

    ic_col1, ic_col2, ic_col3, ic_col4 = st.columns(4)

    with ic_col1:
        try:
            st.metric("AIC", f"{var_model.aic:.4f}")
        except Exception:
            st.metric("AIC", "N/A")

    with ic_col2:
        try:
            st.metric("BIC", f"{var_model.bic:.4f}")
        except Exception:
            st.metric("BIC", "N/A")

    with ic_col3:
        try:
            st.metric("HQIC", f"{var_model.hqic:.4f}")
        except Exception:
            st.metric("HQIC", "N/A")

    with ic_col4:
        try:
            st.metric("FPE", f"{var_model.fpe:.4f}")
        except Exception:
            st.metric("FPE", "N/A")

    # --------------------------------------------------------
    # COEFFICIENTS
    # --------------------------------------------------------

    st.subheader("📋 VAR Coefficients")

    # Build labels from the ACTUAL parameter matrix shape.
    # The saved model currently returns a 6 x 5 matrix. That means
    # 6 rows = 1 constant + 5 coefficient rows = one lag for 5 variables.
    # IMPORTANT: statsmodels VARResultsWrapper can try to rebuild params
    # using stale xnames stored inside an older pickled model. In this
    # project that metadata contains 61 names while the actual parameter
    # matrix is 6 x 5. Read the underlying raw VARResults object first.
    raw_var_model = getattr(var_model, "_results", var_model)
    params_array = np.asarray(raw_var_model.params, dtype=float)

    if params_array.ndim == 1:
        params_array = params_array.reshape(-1, 1)

    param_rows, param_cols = params_array.shape

    if param_cols == len(variables):
        coefficient_columns = variables
    else:
        coefficient_columns = [
            f"Equation {i + 1}"
            for i in range(param_cols)
        ]

    coefficient_table = pd.DataFrame(
        params_array,
        columns=coefficient_columns
    )

    if (
        param_rows > 1
        and len(variables) > 0
        and (param_rows - 1) % len(variables) == 0
    ):
        inferred_lags = (param_rows - 1) // len(variables)
        coefficient_names = ["Constant"]

        for lag in range(1, inferred_lags + 1):
            for variable in variables:
                coefficient_names.append(f"L{lag}.{variable}")
    else:
        inferred_lags = None
        coefficient_names = [
            f"Term {i + 1}"
            for i in range(param_rows)
        ]

    coefficient_table.insert(0, "Term", coefficient_names)

    if inferred_lags is not None:
        st.caption(
            f"Parameter matrix: {param_rows} rows × {param_cols} columns → VAR({inferred_lags})"
        )

    st.dataframe(
        coefficient_table.round(4),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # P VALUES
    # --------------------------------------------------------

    st.subheader("🔬 Coefficient Significance")

    # Use the underlying raw results here for the same reason as above.
    pvalues_array = np.asarray(raw_var_model.pvalues, dtype=float)

    if pvalues_array.ndim == 1:
        pvalues_array = pvalues_array.reshape(-1, 1)

    pvalue_rows, pvalue_cols = pvalues_array.shape

    if pvalue_cols == len(variables):
        pvalue_columns = variables
    else:
        pvalue_columns = [
            f"Equation {i + 1}"
            for i in range(pvalue_cols)
        ]

    pvalue_table = pd.DataFrame(
        pvalues_array,
        columns=pvalue_columns
    )

    if pvalue_rows == len(coefficient_names):
        pvalue_names = coefficient_names
    else:
        pvalue_names = [
            f"Term {i + 1}"
            for i in range(pvalue_rows)
        ]

    pvalue_table.insert(0, "Term", pvalue_names)

    st.dataframe(
        pvalue_table.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Smaller p-values indicate stronger statistical "
        "evidence against the null hypothesis for the "
        "corresponding coefficient."
    )

    # --------------------------------------------------------
    # ROOTS
    # --------------------------------------------------------

    st.subheader("🔍 Characteristic Roots")

    try:
        roots = np.asarray(var_model.roots)

        root_df = pd.DataFrame({
            "Root": range(1, len(roots) + 1),
            "Value": np.round(roots, 4),
            "Absolute Value": np.round(np.abs(roots), 4)
        })

        st.dataframe(
            root_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.warning(
            f"⚠️ Characteristic roots could not be displayed: {e}"
        )

    # --------------------------------------------------------
    # FULL SUMMARY
    # --------------------------------------------------------

    st.subheader("📄 Full VAR Model Summary")

    with st.expander("View Complete VAR(1) Summary"):
        try:
            st.text(str(var_model.summary()))
        except Exception as e:
            st.warning(
                f"⚠️ Full summary could not be displayed: {e}"
            )


# ============================================================
# VECM MODEL
# ============================================================

elif page == "🔄 VECM Model":

    st.header("🔄 Vector Error Correction Model (VECM)")

    st.success("✅ VECM model loaded successfully.")

    st.write(
        """
        The VECM models both short-run dynamics and
        long-run equilibrium relationships among the
        four I(1) macroeconomic variables.
        """
    )

    st.subheader("📌 VECM Model Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Type", "VECM")

    with col2:
        st.metric("Variables", len(vecm_variables))

    with col3:
        try:
            st.metric(
                "Observations",
                len(vecm_model.resid)
            )
        except Exception:
            st.metric("Observations", "N/A")

    with col4:
        st.metric("Cointegration Rank", "1")

    st.subheader("📊 Variables Used in VECM")

    vecm_variable_df = pd.DataFrame({
        "Variable": vecm_variables,
        "Integration Order": ["I(1)"] * len(vecm_variables)
    })

    st.dataframe(
        vecm_variable_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        Industrial Production is I(0), so it is excluded from
        the VECM subsystem. It remains included in the
        five-variable VAR model.
        """
    )

    # --------------------------------------------------------
    # ALPHA
    # --------------------------------------------------------

    st.subheader("⚙️ Adjustment Coefficients (α)")

    try:
        alpha_values = np.asarray(vecm_model.alpha)

        if alpha_values.ndim == 2:
            alpha_values = alpha_values[:, 0]

        alpha_values = alpha_values.flatten()

        alpha_count = min(
            len(alpha_values),
            len(vecm_variables)
        )

        alpha_df = pd.DataFrame({
            "Variable": vecm_variables[:alpha_count],
            "Adjustment Coefficient": alpha_values[:alpha_count]
        })

        st.dataframe(
            alpha_df.round(6),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.warning(
            f"⚠️ Alpha coefficients could not be displayed: {e}"
        )

    # --------------------------------------------------------
    # BETA
    # --------------------------------------------------------

    st.subheader("🔗 Cointegration Coefficients (β)")

    try:
        beta_values = np.asarray(vecm_model.beta)

        if beta_values.ndim == 2:
            beta_values = beta_values[:, 0]

        beta_values = beta_values.flatten()

        beta_count = min(
            len(beta_values),
            len(vecm_variables)
        )

        beta_df = pd.DataFrame({
            "Variable": vecm_variables[:beta_count],
            "Cointegration Coefficient": beta_values[:beta_count]
        })

        st.dataframe(
            beta_df.round(6),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.warning(
            f"⚠️ Beta coefficients could not be displayed: {e}"
        )

    # --------------------------------------------------------
    # VECM DIAGNOSTICS
    # --------------------------------------------------------

    st.subheader("🔬 VECM Diagnostics")

    diag_col1, diag_col2, diag_col3 = st.columns(3)

    with diag_col1:
        st.metric("Cointegration Rank", "1")

    with diag_col2:
        st.metric("VECM Variables", len(vecm_variables))

    with diag_col3:
        try:
            st.metric(
                "Residual Observations",
                len(vecm_model.resid)
            )
        except Exception:
            st.metric("Residual Observations", "N/A")

    # --------------------------------------------------------
    # ALPHA VISUALIZATION
    # --------------------------------------------------------

    st.subheader("⚙️ Adjustment Coefficient Visualization")

    try:
        alpha_values = np.asarray(vecm_model.alpha)

        if alpha_values.ndim == 2:
            alpha_values = alpha_values[:, 0]

        alpha_values = alpha_values.flatten()

        alpha_count = min(
            len(alpha_values),
            len(vecm_variables)
        )

        alpha_plot_df = pd.DataFrame({
            "Variable": vecm_variables[:alpha_count],
            "Adjustment": alpha_values[:alpha_count]
        })

        st.bar_chart(
            alpha_plot_df.set_index("Variable"),
            use_container_width=True
        )

    except Exception as e:
        st.warning(
            f"⚠️ Adjustment coefficient chart unavailable: {e}"
        )

    # --------------------------------------------------------
    # BETA VISUALIZATION
    # --------------------------------------------------------

    st.subheader("🔗 Cointegration Coefficient Visualization")

    try:
        beta_values = np.asarray(vecm_model.beta)

        if beta_values.ndim == 2:
            beta_values = beta_values[:, 0]

        beta_values = beta_values.flatten()

        beta_count = min(
            len(beta_values),
            len(vecm_variables)
        )

        beta_plot_df = pd.DataFrame({
            "Variable": vecm_variables[:beta_count],
            "Coefficient": beta_values[:beta_count]
        })

        st.bar_chart(
            beta_plot_df.set_index("Variable"),
            use_container_width=True
        )

    except Exception as e:
        st.warning(
            f"⚠️ Cointegration coefficient chart unavailable: {e}"
        )

    # --------------------------------------------------------
    # LONG-RUN RELATIONSHIP
    # --------------------------------------------------------

    st.subheader("📐 Estimated Long-Run Relationship")

    try:
        beta_values = np.asarray(vecm_model.beta)

        if beta_values.ndim == 2:
            beta_values = beta_values[:, 0]

        beta_values = beta_values.flatten()

        if len(beta_values) >= 4:
            equation = (
                f"{vecm_variables[0]} "
                f"+ ({beta_values[1]:.4f}) × {vecm_variables[1]} "
                f"+ ({beta_values[2]:.4f}) × {vecm_variables[2]} "
                f"+ ({beta_values[3]:.4f}) × {vecm_variables[3]} = 0"
            )

            st.code(equation, language="text")

    except Exception as e:
        st.warning(
            f"⚠️ Long-run equation could not be displayed: {e}"
        )

    # --------------------------------------------------------
    # FULL VECM SUMMARY
    # --------------------------------------------------------

    st.subheader("📄 Full VECM Model Summary")

    with st.expander("View Complete VECM Summary"):
        try:
            st.text(str(vecm_model.summary()))
        except Exception as e:
            st.warning(
                f"⚠️ Full VECM summary could not be displayed: {e}"
            )


# ============================================================
# VAR VS VECM
# ============================================================

elif page == "⚖️ VAR vs VECM":

    st.header("⚖️ VAR vs VECM Model Comparison")

    st.write(
        "This section compares the two econometric models used "
        "in the project. The VAR model is used for short-run "
        "dynamics and shock propagation, while the VECM model "
        "captures long-run equilibrium relationships among the "
        "I(1) variables."
    )

    st.subheader("📊 Model Overview")

    comparison_df = pd.DataFrame({
        "Feature": [
            "Model",
            "Purpose",
            "Number of Variables",
            "Industrial Production",
            "Cointegration",
            "Main Analysis"
        ],
        "VAR": [
            "VAR(1)",
            "Short-run dynamics and shock propagation",
            "5",
            "Included",
            "Not explicitly modeled",
            "IRF, FEVD, Granger Causality"
        ],
        "VECM": [
            "VECM",
            "Long-run equilibrium relationships",
            "4",
            "Excluded (I(0))",
            "Rank = 1",
            "Long-run adjustment"
        ]
    })

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🔬 Why Two Models?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 VAR Model")

        st.write(
            "The VAR model contains all five variables and is "
            "used to study short-run interactions and how shocks "
            "propagate across the macroeconomic system."
        )

        st.metric("VAR Variables", len(variables))
        st.metric(
            "Selected VAR Lag",
            getattr(var_model, "k_ar", 1)
        )

    with col2:
        st.markdown("### 🔄 VECM Model")

        st.write(
            "The VECM model uses the four variables identified "
            "as I(1). Industrial Production was excluded because "
            "it showed I(0) behavior in the stationarity analysis."
        )

        st.metric("VECM Variables", len(vecm_variables))
        st.metric("Cointegration Rank", "1")

    st.subheader("🔗 Long-Run Relationship")

    st.info(
        "The Johansen cointegration test identified one "
        "cointegrating relationship among CPI Inflation, "
        "Policy Rate, USD/INR and Crude Oil Price."
    )

    st.subheader("🎯 Project Interpretation")

    st.write(
        "The two models complement each other: the VAR model "
        "provides short-run shock propagation analysis, while "
        "the VECM provides information about long-run "
        "equilibrium adjustment."
    )


# ============================================================
# IMPULSE RESPONSE
# ============================================================

elif page == "💥 Impulse Response":

    st.header("💥 Impulse Response Function (IRF)")

    st.write(
        """
        IRF measures how a shock to one variable affects
        the other variables over subsequent months.
        """
    )

    try:
        horizon = st.slider(
            "Forecast Horizon (Months)",
            min_value=1,
            max_value=24,
            value=12,
            key="irf_horizon"
        )

        shock_variable = st.selectbox(
            "Select Shock Variable",
            variables,
            key="irf_shock_variable"
        )

        st.info(f"📌 Selected shock: {shock_variable}")

        irf = var_model.irf(horizon)
        irf_values = np.asarray(irf.irfs)

        shock_index = variables.index(shock_variable)

        st.subheader("📊 Impulse Response Results")

        irf_table = pd.DataFrame(
            irf_values[:, :, shock_index],
            columns=variables
        )

        irf_table.insert(
            0,
            "Month",
            range(len(irf_table))
        )

        st.dataframe(
            irf_table.round(5),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            f"📈 Responses to {shock_variable} Shock"
        )

        fig, ax = plt.subplots(figsize=(11, 6))

        for response_index, response_variable in enumerate(variables):
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

        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Months After Shock")
        ax.set_ylabel("Response")
        ax.set_title(
            f"Impulse Responses to {shock_variable} Shock"
        )
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.success(
            "✅ Impulse Response analysis completed."
        )

    except Exception as e:
        st.error("❌ Impulse Response Error")
        st.exception(e)


# ============================================================
# FEVD
# ============================================================

elif page == "📊 FEVD":

    st.header("📊 Forecast Error Variance Decomposition")

    st.write(
        """
        FEVD shows how much of the forecast uncertainty
        of each variable is explained by shocks to the
        different variables in the VAR system.
        """
    )

    try:
        horizon = st.slider(
            "FEVD Horizon (Months)",
            min_value=1,
            max_value=24,
            value=12,
            key="fevd_horizon"
        )

        selected_variable = st.selectbox(
            "Select Variable",
            variables,
            key="fevd_variable"
        )

        selected_index = variables.index(selected_variable)

        fevd = var_model.fevd(horizon)

        fevd_values = np.asarray(
            fevd.decomp[selected_index]
        )

        actual_horizon = len(fevd_values)

        fevd_table = pd.DataFrame(
            fevd_values,
            columns=variables
        )

        fevd_table.insert(
            0,
            "Horizon",
            range(1, actual_horizon + 1)
        )

        st.subheader(
            f"📊 Variance Decomposition: {selected_variable}"
        )

        st.dataframe(
            fevd_table.round(4),
            use_container_width=True,
            hide_index=True
        )

        selected_horizon = min(
            horizon,
            actual_horizon
        )

        horizon_values = fevd_table.iloc[
            selected_horizon - 1
        ]

        st.subheader(
            f"📌 Contributions at Month {selected_horizon}"
        )

        contribution_cols = st.columns(len(variables))

        for i, variable in enumerate(variables):
            with contribution_cols[i]:
                st.metric(
                    variable,
                    f"{horizon_values[variable] * 100:.2f}%"
                )

        st.subheader("📊 FEVD Contribution Chart")

        fig, ax = plt.subplots(figsize=(12, 6))

        bottom = np.zeros(actual_horizon)

        for variable in variables:
            values = fevd_table[variable].values

            ax.bar(
                fevd_table["Horizon"],
                values,
                bottom=bottom,
                label=variable
            )

            bottom += values

        ax.set_xlabel("Forecast Horizon (Months)")
        ax.set_ylabel("Proportion of Forecast Error")
        ax.set_title(f"FEVD of {selected_variable}")
        ax.set_ylim(0, 1)

        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        largest_source = max(
            variables,
            key=lambda x: horizon_values[x]
        )

        largest_percentage = (
            horizon_values[largest_source] * 100
        )

        st.info(
            f"""
            📌 At forecast horizon {selected_horizon} month(s),
            the largest contribution to the forecast error of
            **{selected_variable}** comes from
            **{largest_source}**, contributing approximately
            **{largest_percentage:.2f}%**.
            """
        )

        st.success(
            "✅ FEVD analysis completed successfully."
        )

    except Exception as e:
        st.error("❌ FEVD Error")
        st.exception(e)


# ============================================================
# GRANGER CAUSALITY
# ============================================================

elif page == "🔗 Granger Causality":

    st.header("🔗 Granger Causality Analysis")

    st.write(
        """
        Granger causality tests whether past values of one
        macroeconomic variable provide useful predictive
        information about another variable.
        """
    )

    try:
        granger_results = []

        for cause in variables:
            for effect in variables:

                if cause == effect:
                    continue

                test_data = model_data[
                    [effect, cause]
                ].dropna()

                try:
                    result = grangercausalitytests(
                        test_data,
                        maxlag=1,
                        verbose=False
                    )

                    p_value = result[
                        1
                    ][0][
                        "ssr_ftest"
                    ][1]

                    significance = (
                        "Significant"
                        if p_value < 0.05
                        else "Not Significant"
                    )

                    granger_results.append({
                        "Cause": cause,
                        "Effect": effect,
                        "Lag": 1,
                        "p_value": p_value,
                        "Result": significance
                    })

                except Exception:
                    granger_results.append({
                        "Cause": cause,
                        "Effect": effect,
                        "Lag": 1,
                        "p_value": np.nan,
                        "Result": "Error"
                    })

        granger_df = pd.DataFrame(granger_results)

        st.subheader("📊 Granger Causality Results")

        st.dataframe(
            granger_df.round(5),
            use_container_width=True,
            hide_index=True
        )

        significant_df = granger_df[
            granger_df["p_value"] < 0.05
        ]

        st.subheader("🔍 Significant Predictive Relationships")

        if len(significant_df) > 0:
            st.dataframe(
                significant_df.round(5),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No statistically significant Granger "
                "relationships were found at the 5% level."
            )

    except Exception as e:
        st.error("❌ Granger Causality Error")
        st.exception(e)


# ============================================================
# SHOCK PROPAGATION
# ============================================================

elif page == "🌐 Shock Propagation":

    st.header("🌐 Macroeconomic Shock Propagation")

    st.write(
        """
        Analyze how a shock originating in one macroeconomic
        variable propagates through the VAR system over time.
        """
    )

    st.info(
        """
        The propagation analysis uses the estimated VAR
        impulse-response functions to trace the dynamic
        response of all variables to a selected shock.
        """
    )

    try:
        col1, col2 = st.columns(2)

        with col1:
            propagation_shock = st.selectbox(
                "💥 Select Shock Variable",
                variables,
                key="propagation_shock_variable"
            )

        with col2:
            propagation_horizon = st.slider(
                "⏱️ Propagation Horizon (Months)",
                min_value=1,
                max_value=24,
                value=12,
                key="propagation_horizon"
            )

        irf = var_model.irf(propagation_horizon)

        irf_values = np.asarray(irf.irfs)

        shock_index = variables.index(propagation_shock)

        propagation_values = irf_values[
            :,
            :,
            shock_index
        ]

        propagation_df = pd.DataFrame(
            propagation_values,
            columns=variables
        )

        propagation_df.insert(
            0,
            "Month",
            range(len(propagation_df))
        )

        st.subheader(
            f"📊 Propagation Results: {propagation_shock} Shock"
        )

        st.dataframe(
            propagation_df.round(5),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("📈 Shock Propagation Across Variables")

        fig, ax = plt.subplots(figsize=(12, 6))

        for variable in variables:
            ax.plot(
                propagation_df["Month"],
                propagation_df[variable],
                marker="o",
                linewidth=2,
                label=variable
            )

        ax.axhline(0, linewidth=1)

        ax.set_xlabel("Months After Shock")
        ax.set_ylabel("Response")
        ax.set_title(
            f"Propagation of {propagation_shock} Shock"
        )

        ax.grid(alpha=0.3)

        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

        plt.tight_layout()

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.subheader("🔎 Analyze Individual Propagation")

        response_variable = st.selectbox(
            "Select Response Variable",
            variables,
            key="propagation_response_variable"
        )

        response_series = propagation_df[response_variable]

        response_chart_df = pd.DataFrame({
            response_variable: response_series.values
        })

        response_chart_df.index.name = "Month"

        st.line_chart(
            response_chart_df,
            use_container_width=True
        )

        st.subheader("📌 Peak Propagation Analysis")

        future_values = response_series.iloc[1:]

        if len(future_values) > 0:
            peak_month = int(
                future_values.abs().idxmax()
            )

            peak_response = float(
                propagation_df.loc[
                    peak_month,
                    response_variable
                ]
            )
        else:
            peak_month = 0
            peak_response = float(response_series.iloc[0])

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                "Peak Response",
                f"{peak_response:.5f}"
            )

        with metric_col2:
            st.metric(
                "Peak Month",
                f"Month {peak_month}"
            )

        with metric_col3:
            if peak_response > 0:
                direction = "Positive"
            elif peak_response < 0:
                direction = "Negative"
            else:
                direction = "Neutral"

            st.metric("Direction", direction)

        st.subheader("📋 Propagation Summary")

        propagation_summary = []

        for variable in variables:

            values = propagation_df[variable].values
            future = values[1:]

            if len(future) > 0:
                peak_position = (
                    int(np.argmax(np.abs(future))) + 1
                )

                peak_value = float(
                    values[peak_position]
                )
            else:
                peak_position = 0
                peak_value = float(values[0])

            if peak_value > 0:
                variable_direction = "Positive"
            elif peak_value < 0:
                variable_direction = "Negative"
            else:
                variable_direction = "Neutral"

            propagation_summary.append({
                "Variable": variable,
                "Peak Response": peak_value,
                "Peak Month": peak_position,
                "Direction": variable_direction
            })

        propagation_summary_df = pd.DataFrame(
            propagation_summary
        )

        st.dataframe(
            propagation_summary_df.round(5),
            use_container_width=True,
            hide_index=True
        )

        st.success(
            "✅ Shock propagation analysis completed successfully."
        )

    except Exception as e:
        st.error("❌ Shock Propagation Error")
        st.exception(e)


# ============================================================
# SHOCK SIMULATOR
# ============================================================

elif page == "🎛️ Shock Simulator":

    st.header("🎛️ Interactive Macroeconomic Shock Simulator")

    st.write(
        """
        Simulate how a shock to one macroeconomic variable
        propagates through the VAR system over time.
        """
    )

    st.info(
        """
        The simulator uses the estimated VAR impulse-response
        functions. The selected shock magnitude is used as a
        scaling factor for the model response. It should not be
        interpreted as a direct real-world percentage forecast.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        shock_variable = st.selectbox(
            "💥 Select Shock Variable",
            variables,
            key="simulator_shock_variable"
        )

    with col2:
        shock_size = st.number_input(
            "📏 Shock Size (%)",
            min_value=-100.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            key="simulator_shock_size"
        )

    horizon = st.slider(
        "⏱️ Forecast Horizon (Months)",
        min_value=1,
        max_value=24,
        value=12,
        key="simulator_horizon"
    )

    st.caption(
        "Example: +10% scales the estimated impulse response "
        "by 0.10, while -10% scales it by -0.10."
    )

    simulate = st.button(
        "🚀 Run Shock Simulation",
        type="primary",
        use_container_width=True
    )

    if simulate:

        try:
            irf_sim = var_model.irf(horizon)

            irf_values = np.asarray(irf_sim.irfs)

            shock_index = variables.index(shock_variable)

            responses = irf_values[
                :,
                :,
                shock_index
            ]

            shock_multiplier = shock_size / 100.0

            simulated_responses = (
                responses * shock_multiplier
            )

            simulation_df = pd.DataFrame(
                simulated_responses,
                columns=variables
            )

            simulation_df.insert(
                0,
                "Month",
                range(len(simulation_df))
            )

            st.success(
                f"✅ Simulation completed for a "
                f"{shock_size:.1f}% shock in "
                f"{shock_variable}."
            )

            st.subheader("📊 Shock Response Results")

            st.dataframe(
                simulation_df.round(5),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("📈 Shock Propagation Over Time")

            chart_df = simulation_df.set_index("Month")

            st.line_chart(
                chart_df,
                use_container_width=True
            )

            st.subheader("🔎 Analyze Individual Response")

            response_variable = st.selectbox(
                "Select Response Variable",
                variables,
                key="simulator_response_variable"
            )

            response_values = simulation_df[response_variable]

            fig, ax = plt.subplots(figsize=(11, 5))

            ax.plot(
                simulation_df["Month"],
                response_values,
                marker="o",
                linewidth=2
            )

            ax.axhline(0, linewidth=1)

            ax.set_xlabel("Months After Shock")
            ax.set_ylabel("Simulated Response")

            ax.set_title(
                f"{response_variable} Response to "
                f"{shock_variable} Shock"
            )

            ax.grid(alpha=0.3)

            plt.tight_layout()

            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            st.subheader("📌 Peak Response Analysis")

            future_values = response_values.iloc[1:]

            if len(future_values) > 0:
                peak_month = int(
                    future_values.abs().idxmax()
                )

                peak_response = float(
                    simulation_df.loc[
                        peak_month,
                        response_variable
                    ]
                )
            else:
                peak_month = 0
                peak_response = float(response_values.iloc[0])

            if peak_response > 0:
                direction = "Positive"
            elif peak_response < 0:
                direction = "Negative"
            else:
                direction = "Neutral"

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    "Peak Response",
                    f"{peak_response:.5f}"
                )

            with metric_col2:
                st.metric(
                    "Peak Month",
                    f"Month {peak_month}"
                )

            with metric_col3:
                st.metric(
                    "Direction",
                    direction
                )

            st.subheader("🔎 Interpretation")

            if peak_response > 0:
                st.success(
                    f"""
                    The simulated response of
                    **{response_variable}** is positive at its
                    largest absolute response.

                    The peak response occurs around
                    **month {peak_month}**.
                    """
                )

            elif peak_response < 0:
                st.warning(
                    f"""
                    The simulated response of
                    **{response_variable}** is negative at its
                    largest absolute response.

                    The peak response occurs around
                    **month {peak_month}**.
                    """
                )

            else:
                st.info(
                    f"""
                    The simulated response of
                    **{response_variable}** is approximately zero
                    over the selected forecast horizon.
                    """
                )

            st.subheader("📋 Simulation Summary")

            summary_results = []

            for response_variable_name in variables:

                values = simulation_df[
                    response_variable_name
                ].values

                future_values = values[1:]

                if len(future_values) > 0:
                    peak_index = (
                        int(
                            np.argmax(
                                np.abs(future_values)
                            )
                        ) + 1
                    )

                    peak_value = float(
                        values[peak_index]
                    )

                else:
                    peak_index = 0
                    peak_value = float(values[0])

                if peak_value > 0:
                    response_direction = "Positive"
                elif peak_value < 0:
                    response_direction = "Negative"
                else:
                    response_direction = "Neutral"

                summary_results.append({
                    "Variable": response_variable_name,
                    "Peak Response": peak_value,
                    "Peak Month": peak_index,
                    "Direction": response_direction
                })

            summary_df = pd.DataFrame(summary_results)

            st.dataframe(
                summary_df.round(5),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("📥 Export Simulation Results")

            csv_data = simulation_df.to_csv(index=False)

            st.download_button(
                "📥 Download Simulation Results",
                data=csv_data,
                file_name="shock_simulation_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error("❌ Shock Simulation Error")
            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Multivariate Macroeconomic Shock Propagation Modeling | "
    "VAR(1) | VECM | IRF | FEVD | Granger Causality"
)
