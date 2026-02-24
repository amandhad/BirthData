import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(layout="wide")
st.title("Provisional Natality Data Dashboard")
st.subheader("Birth Analysis by State and Gender")

# -----------------------------
# Load and Prepare Data
# -----------------------------
try:
    df = pd.read_csv("Provisional_Natality_2025_CDC.csv")
except FileNotFoundError:
    st.error("Dataset file not found in repository.")
    st.stop()

# Normalize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Required logical fields
required_fields = {
    "state_of_residence",
    "month",
    "month_code",
    "year_code",
    "sex_of_infant",
    "births"
}

missing_fields = required_fields - set(df.columns)
if missing_fields:
    st.error(f"Missing required logical fields: {sorted(list(missing_fields))}")
    st.write(df.columns)
    st.stop()

# Convert births to numeric and drop nulls
df["births"] = pd.to_numeric(df["births"], errors="coerce")
df = df.dropna(subset=["births"])

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

def multiselect_with_all(label, options):
    options = sorted(options)
    return st.sidebar.multiselect(
        label,
        options=["All"] + options,
        default=["All"]
    )

selected_months = multiselect_with_all(
    "Select Month(s)",
    df["month"].dropna().unique()
)

selected_genders = multiselect_with_all(
    "Select Gender(s)",
    df["sex_of_infant"].dropna().unique()
)

selected_states = multiselect_with_all(
    "Select State(s)",
    df["state_of_residence"].dropna().unique()
)

# -----------------------------
# Filtering Logic
# -----------------------------
filtered_df = df.copy()

if "All" not in selected_months:
    filtered_df = filtered_df[filtered_df["month"].isin(selected_months)]

if "All" not in selected_genders:
    filtered_df = filtered_df[filtered_df["sex_of_infant"].isin(selected_genders)]

if "All" not in selected_states:
    filtered_df = filtered_df[filtered_df["state_of_residence"].isin(selected_states)]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# -----------------------------
# Aggregation
# -----------------------------
agg_df = (
    filtered_df
    .groupby(["state_of_residence", "sex_of_infant"], as_index=False)["births"]
    .sum()
    .sort_values("state_of_residence")
)

# -----------------------------
# Plot
# -----------------------------
fig = px.bar(
    agg_df,
    x="state_of_residence",
    y="births",
    color="sex_of_infant",
    title="Total Births by State and Gender",
    labels={
        "state_of_residence": "State",
        "births": "Total Births",
        "sex_of_infant": "Gender"
    }
)

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend_title_text="Gender",
    xaxis_title="State",
    yaxis_title="Total Births",
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Display Filtered Table
# -----------------------------
st.subheader("Filtered Data")
st.dataframe(
    agg_df.reset_index(drop=True),
    use_container_width=True
)
