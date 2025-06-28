import streamlit as st

st.set_page_config(page_title="EDA - Telco Churn", layout="wide")
st.title("Dataset Exploration: Telco Customer Churn")

st.image("image.png", use_container_width=True)

# --- Dataset description ---
st.markdown("""
### About Dataset  
**Context**  
"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]  

**Content**  
Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

**The data set includes information about:**
- Customers who left within the last month – the column is called `Churn`
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

**Columns:**
- `customerID`: Customer ID.
- `gender`: Whether the customer is a male or a female.
- `SeniorCitizen`: Whether the customer is a senior citizen (1, 0).
- `Partner`: Whether the customer has a partner (Yes, No).
- `Dependents`: Whether the customer has dependents (Yes, No).
- `tenure`: Number of months the customer has stayed with the company.
- `PhoneService`: Whether the customer has a phone service (Yes, No).
- `MultipleLines`: Whether the customer has multiple lines (Yes, No, No phone service).
- `InternetService`: Customer’s internet service provider (DSL, Fiber optic, No).
- `OnlineSecurity`: Whether the customer has online security (Yes, No, No internet service).
- `OnlineBackup`: Whether the customer has online backup (Yes, No, No internet service).
- `DeviceProtection`: Whether the customer has device protection (Yes, No, No internet service).
- `TechSupport`: Whether the customer has tech support (Yes, No, No internet service).
- `StreamingTV`: Whether the customer has streaming TV (Yes, No, No internet service).
- `StreamingMovies`: Whether the customer has streaming movies (Yes, No, No internet service).
- `Contract`: The contract term of the customer (Month-to-month, One year, Two year).
- `PaperlessBilling`: Whether the customer has paperless billing (Yes, No).
- `PaymentMethod`: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
- `MonthlyCharges`: The amount charged to the customer monthly.
- `TotalCharges`: The total amount charged to the customer.
- `Churn`: Whether the customer churned (Yes, No).
""")

import pandas as pd

# --- Loading dataset ---
@st.cache_data
def cargar_dataset():
    return pd.read_csv("data/telco_churn.csv")

df = cargar_dataset()

# --- Show preview ---
st.markdown("### Dataset Preview")
st.markdown("Explore how the Telco dataset is structured and organized for this churn analysis study.")

st.dataframe(df.head(), use_container_width=True)

# --- Descriptive statistics by variable ---
st.markdown("### Descriptive statistics by variable")
st.markdown("Select any variable to view its descriptive statistics, including mean, standard deviation, minimum, maximum, and more.")


columna_seleccionada = st.selectbox("Select a column", df.columns)

st.write(f"**Statistics for `{columna_seleccionada}`:**")

# Centrar la tabla usando columnas vacías a los lados
col1, col2, col3 = st.columns([7, 6, 7])  # proporción de espacio

with col2:
    st.dataframe(df[columna_seleccionada].describe().to_frame())

import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("### Distribution of the selected column by `Churn`")
st.markdown("Choose a feature to visualize its distribution segmented by the churn label.")

# Do not include the target variable in the selection
columnas_disponibles = [col for col in df.columns if col != "Churn"]
columna_dist = st.selectbox("Select a variable to plot", columnas_disponibles)

fig, ax = plt.subplots(figsize=(6, 4))

if df[columna_dist].dtype in ["float64", "int64"]:
    # Histogram for numerical variables
    sns.histplot(data=df, x=columna_dist, hue="Churn", kde=True, element="step", ax=ax)
else:
    # Count plot for categorical variables
    barplot = sns.countplot(data=df, x=columna_dist, hue="Churn", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Add centered text to each bar
    for container in barplot.containers:
        for bar in container:
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            y = height / 2
            ax.text(x, y, f"{int(height)}", ha='center', va='center', fontsize=9, color='black', fontweight='bold')

ax.set_title(f"Distribution of {columna_dist} by Churn", fontsize=10)
ax.set_xlabel(columna_dist, fontsize=9)
ax.set_ylabel("Count", fontsize=9)
ax.tick_params(axis='both', labelsize=8)
ax.legend(title="Churn", fontsize=8, title_fontsize=9)

fig.tight_layout()
st.pyplot(fig)

# --- Distribution of the target variable: Churn ---
st.markdown("### Distribution of the target variable: `Churn`")
st.markdown("View how customers are distributed based on whether they churned or not.")

churn_counts = df["Churn"].value_counts(normalize=True).rename("Percentage").to_frame()
churn_counts.index.name = "Churn"

# Center the table using columns
col1, col2, col3 = st.columns([7,6,7])
with col2:
    st.table(churn_counts.style.format("{:.2%}"))

# --- Plot with centered text ---
fig_churn, ax_churn = plt.subplots(figsize=(4.5, 3))

barplot = sns.countplot(data=df, x="Churn", palette="pastel", ax=ax_churn)
ax_churn.set_title("Distribution of Churn", fontsize=10)
ax_churn.set_xlabel("Churn", fontsize=9)
ax_churn.set_ylabel("Count", fontsize=9)
ax_churn.tick_params(axis='both', labelsize=8)

# Add centered text to each bar
for container in barplot.containers:
    for bar in container:
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        y = height / 2
        ax_churn.text(x, y, f"{int(height)}", ha='center', va='center', fontsize=9, color='black', fontweight='bold')

fig_churn.tight_layout()
st.pyplot(fig_churn)