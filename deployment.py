import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Scrum Team Information
st.sidebar.title("SCRUM Methodology")
st.sidebar.subheader("Team Roles")
st.sidebar.write("👤 **Product Owner:** Manages backlog & prioritizes work")
st.sidebar.write("🛠 **Scrum Master:** Facilitates the SCRUM process")
st.sidebar.write("👨‍💻 **Developers:** Implement and test the project")

# Title & Dataset Upload
st.title("Data Analysis Dashboard - SCRUM Model")
st.write("### Upload a CSV file or use the default dataset")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

def load_data():
    iris = datasets.load_iris()
    return pd.DataFrame(data=iris.data, columns=iris.feature_names)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

# Data Preprocessing
st.write("## Data Preprocessing")
st.write("Handling missing values and converting categorical features...")

# Handle missing values
df.dropna(inplace=True)

# Convert categorical columns to numeric using encoding
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

st.write("### Processed Dataset Preview")
st.dataframe(df.head())

# Data Visualization
st.write("## Data Visualizations")

# Histogram
st.write("### Histogram")
feature = st.selectbox("Select feature to visualize", df.columns)
fig, ax = plt.subplots()
sns.histplot(df[feature], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# Scatter Plot
st.write("### Scatter Plot")
x_axis = st.selectbox("Select X-axis", df.columns)
y_axis = st.selectbox("Select Y-axis", df.columns)
fig, ax = plt.subplots()
sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
st.pyplot(fig)

# Box Plot
st.write("### Box Plot")
fig, ax = plt.subplots()
sns.boxplot(data=df, ax=ax)
st.pyplot(fig)

# Pair Plot
st.write("### Pair Plot")
st.pyplot(sns.pairplot(df))

# Data Insights
st.write("## Data Insights")
st.write("### Summary Statistics")
st.dataframe(df.describe())

st.write("### Correlation Matrix")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.write("### Key Observations")
st.write("- The dataset contains {} rows and {} columns.".format(df.shape[0], df.shape[1]))
st.write("- The highest correlated features are:")
corr_matrix = df.corr().abs()
st.write(corr_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(5))
st.write("- The distribution of features varies, with some showing normal distribution while others are skewed.")
st.write("- Box plots help identify outliers in each feature.")
st.write("- Pair plots provide an overview of feature relationships.")

# SCRUM Sprint Board
st.write("## Sprint Backlog")
st.write("### User Stories")
st.write("- As a data analyst, I want to visualize trends so that I can make data-driven decisions.")
st.write("- As a manager, I want to see key insights quickly so that I can strategize effectively.")
st.write("- As a developer, I want to clean data efficiently so that models work accurately.")

st.write("### Sprint Tasks")
sprint_tasks = ["Load dataset", "Preprocess data", "Visualize data", "Generate insights", "Deploy application"]
st.table({"Task": sprint_tasks, "Status": ["Done", "Done", "Done", "Done", "Pending"]})
