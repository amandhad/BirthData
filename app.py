import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(layout="wide")
st.title("NovaRetail Customer Intelligence Dashboard")
st.subheader("Interactive Dashboard for Decision Making")

# ---------------------------------
# Load Data
# ---------------------------------
try:
    df = pd.read_excel("NR_dataset.xlsx")
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
    "label",
    "customerid",
    "transactionid",
    "transactiondate",
    "productcategory",
    "purchaseamount",
    "customeragegroup",
    "customergender",
    "customerregion",
    "customersatisfaction",
    "retailchannel"
}

missing_fields = required_fields - set(df.columns)
if missing_fields:
    st.error(f"Missing required fields: {sorted(list(missing_fields))}")
    st.write(df.columns)
    st.stop()

# Data cleaning
df["purchaseamount"] = pd.to_numeric(df["purchaseamount"], errors="coerce")
df["customersatisfaction"] = pd.to_numeric(df["customersatisfaction"], errors="coerce")
df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")

df = df.dropna(subset=["purchaseamount"])

# ---------------------------------
# Sidebar Filters
# ---------------------------------
st.sidebar.header("Filters")

def multiselect_all(label, values):
    values = sorted(values)
    return st.sidebar.multiselect(
        label,
        ["All"] + values,
        default=["All"]
    )

segments = multiselect_all("Customer Segment", df["label"].unique())
regions = multiselect_all("Region", df["customerregion"].unique())
categories = multiselect_all("Product Category", df["productcategory"].unique())
channels = multiselect_all("Retail Channel", df["retailchannel"].unique())

# ---------------------------------
# Filtering Logic
# ---------------------------------
filtered_df = df.copy()

if "All" not in segments:
    filtered_df = filtered_df[filtered_df["label"].isin(segments)]

if "All" not in regions:
    filtered_df = filtered_df[filtered_df["customerregion"].isin(regions)]

if "All" not in categories:
    filtered_df = filtered_df[filtered_df["productcategory"].isin(categories)]

if "All" not in channels:
    filtered_df = filtered_df[filtered_df["retailchannel"].isin(channels)]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# ---------------------------------
# KPIs
# ---------------------------------
total_revenue = filtered_df["purchaseamount"].sum()
unique_customers = filtered_df["customerid"].nunique()
avg_satisfaction = filtered_df["customersatisfaction"].mean()

kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric("Total Revenue ($)", f"{total_revenue:,.0f}")
kpi2.metric("Unique Customers", f"{unique_customers}")
kpi3.metric("Avg. Satisfaction", f"{avg_satisfaction:.2f}")

# ---------------------------------
# Charts
# ---------------------------------
col1, col2 = st.columns(2)

# Revenue by Segment
segment_revenue = (
    filtered_df
    .groupby("label", as_index=False)["purchaseamount"]
    .sum()
    .sort_values("label")
)

fig_segment = px.bar(
    segment_revenue,
    x="label",
    y="purchaseamount",
    title="Revenue by Customer Segment",
    labels={"label": "Customer Segment", "purchaseamount": "Revenue ($)"}
)

fig_segment.update_layout(plot_bgcolor="white", paper_bgcolor="white")

col1.plotly_chart(fig_segment, use_container_width=True)

# Revenue by Region
region_revenue = (
    filtered_df
    .groupby("customerregion", as_index=False)["purchaseamount"]
    .sum()
    .sort_values("customerregion")
)

fig_region = px.bar(
    region_revenue,
    x="customerregion",
    y="purchaseamount",
    title="Revenue by Region",
    labels={"customerregion": "Region", "purchaseamount": "Revenue ($)"}
)

fig_region.update_layout(plot_bgcolor="white", paper_bgcolor="white")

col2.plotly_chart(fig_region, use_container_width=True)

# ---------------------------------
# Satisfaction vs Revenue (Risk Detection)
# ---------------------------------
st.subheader("Customer Satisfaction vs Revenue")

sat_rev = (
    filtered_df
    .groupby("label", as_index=False)
    .agg(
        avg_satisfaction=("customersatisfaction", "mean"),
        total_revenue=("purchaseamount", "sum")
    )
)

fig_scatter = px.scatter(
    sat_rev,
    x="avg_satisfaction",
    y="total_revenue",
    color="label",
    size="total_revenue",
    title="Customer Segment Risk & Opportunity Analysis",
    labels={
        "avg_satisfaction": "Average Satisfaction",
        "total_revenue": "Total Revenue ($)",
        "label": "Customer Segment"
    }
)

fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white")

st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------
# Filtered Data Table
# ---------------------------------
st.subheader("Filtered Transaction Data")
st.dataframe(
    filtered_df.reset_index(drop=True),
    use_container_width=True
)
