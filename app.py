import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Powered Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Powered Data Analyst Agent")
st.write("Upload your dataset or use the default store dataset to automatically generate Exploratory Data Analysis (EDA), univariate, bivariate, multivariate charts, and chat with your data!")

# ==========================================
# SIDEBAR: API KEYS & FILE UPLOAD
# ==========================================
st.sidebar.header("Configuration")
GOOGLE_API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")
GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")

# Use default or user-provided keys
default_google = os.environ.get("GOOGLE_API_KEY", "")
default_groq = os.environ.get("GROQ_API_KEY", "")

active_google_key = GOOGLE_API_KEY if GOOGLE_API_KEY else default_google
active_groq_key = GROQ_API_KEY if GROQ_API_KEY else default_groq

# Initialize LLMs safely
try:
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=active_google_key
    )
except Exception as e:
    gemini_llm = None

try:
    groq_llm = ChatGroq(
        model="qwen-2.5-coder-32b",
        api_key=active_groq_key
    )
except Exception as e:
    groq_llm = None

# Dummy Tool Definition
def temp_tool():
    """This is just a dummy tool"""
    return "Hello world"

# Initialize Agent
if gemini_llm:
    try:
        agent = create_agent(
            model=gemini_llm,
            tools=[temp_tool]
        )
    except Exception:
        agent = None
else:
    agent = None

st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV, XLS, or XLSX file", type=["csv", "xlsx", "xls"])

# Dataset loading logic
@st.cache_data
def load_data(file_path_or_buffer):
    if isinstance(file_path_or_buffer, str):
        if file_path_or_buffer.endswith('.csv'):
            return pd.read_csv(file_path_or_buffer)
        else:
            return pd.read_excel(file_path_or_buffer)
    else:
        # Streamlit uploaded file buffer
        try:
            return pd.read_csv(file_path_or_buffer)
        except Exception:
            file_path_or_buffer.seek(0)
            return pd.read_excel(file_path_or_buffer)

# Load default or uploaded dataset
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("Successfully loaded uploaded file!")
else:
    default_url = 'https://raw.githubusercontent.com/axisgras-hash/DATASETS/refs/heads/main/Superstore.csv'
    st.sidebar.info("Using default Superstore Dataset. Upload your own to switch.")
    try:
        df = load_data(default_url)
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
        df = pd.DataFrame()

if not df.empty:
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # ==========================================
    # AUTOMATED & ADVANCED EDA SECTION
    # ==========================================
    st.subheader("📈 Automated Exploratory Data Analysis (EDA)")
    
    if st.button("Run Comprehensive AI EDA"):
        with st.spinner("Generating and executing analysis code using AI Agent..."):
            try:
                # Basic Stats & Missing Values Display
                st.markdown("### 1. Basic Dataset Summary & Missing Values")
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", df.shape[0])
                col2.metric("Columns", df.shape[1])
                col3.metric("Missing Values", df.isnull().sum().sum())

                st.write("**Data Description:**")
                st.dataframe(df.describe())

                st.write("**Missing Values Per Column:**")
                st.bar_chart(df.isnull().sum())

                # ==========================================
                # UNIVARIATE, BIVARIATE & MULTIVARIATE CHARTS
                # ==========================================
                st.markdown("### 2. Univariate Analysis (Numerical Distributions)")
                num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if len(num_cols) > 0:
                    selected_num_col = st.selectbox("Select Numerical Column for Distribution", num_cols)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(df[selected_num_col], kde=True, ax=ax, color='blue')
                    st.pyplot(fig)

                st.markdown("### 3. Bivariate Analysis")
                if len(num_cols) >= 2:
                    col_x = st.selectbox("Select X-axis Column", num_cols, index=0)
                    col_y = st.selectbox("Select Y-axis Column", num_cols, index=1 if len(num_cols)>1 else 0)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax, color='green')
                    st.pyplot(fig)

                st.markdown("### 4. Multivariate Analysis & Correlation")
                if len(num_cols) > 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    corr = df[num_cols].corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)

                # Categorical vs Numerical Multivariate example if available
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if len(cat_cols) > 0 and len(num_cols) > 0:
                    st.markdown("### 5. Categorical Grouping Analysis (Multivariate Bar Plot)")
                    cat_group = st.selectbox("Select Categorical Column", cat_cols)
                    num_val = st.selectbox("Select Numerical Value Metric", num_cols)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    top_cat = df.groupby(cat_group)[num_val].sum().reset_index().sort_values(by=num_val, ascending=False).head(10)
                    sns.barplot(data=top_cat, x=cat_group, y=num_val, ax=ax, palette='viridis')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                st.success("EDA Completed Successfully!")

            except Exception as e:
                st.error(f"Error during EDA execution: {e}")

    # ==========================================
    # CHAT WITH DATA SECTION
    # ==========================================
    st.subheader("💬 Chat with your Dataset")
    user_query = st.text_input("Ask a question about your data (e.g., 'What is the total sales by region?'):")
    
    if user_query:
        if agent:
            with st.spinner("Analyzing your query with Agent..."):
                chat_prompt = f"""
                You are a data analyst assistant. Given the following pandas dataframe info:
                Columns: {list(df.columns)}
                Data Sample: {df.head(2).to_dict()}
                
                Answer the user question clearly: {user_query}
                """
                try:
                    response = agent.invoke({'messages': [{'role': 'user', 'content': chat_prompt}]})
                    answer_text = response["messages"][-1].content
                    if isinstance(answer_text, list):
                        answer_text = answer_text[-1].get('text', str(answer_text))
                    st.write("**Agent Response:**")
                    st.write(answer_text)
                except Exception as e:
                    # Fallback query response mechanism
                    st.write("**Agent Response (Fallback Analysis):**")
                    if "sales" in user_query.lower() and "sales" in df.columns:
                        st.write(df.groupby(df.select_dtypes(include=['object']).columns[0])['Sales'].sum() if 'Sales' in df.columns else df.sum())
                    else:
                        st.write("Here is a quick description matching your query context:")
                        st.write(df.describe())
        else:
            st.warning("Please configure valid API keys in the sidebar to use the chat agent feature.")

else:
    st.warning("Please upload a dataset or ensure the default dataset path is reachable.")

# ==========================================
# EXTRA CODE & BACKEND NOTES (kept in comments)
# ==========================================
# 
# # Backend file original workflow reference:
# # GOOGLE_API_KEY = ""
# # GROQ_API_KEY = ""
# # gemini_llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite", google_api_key = GOOGLE_API_KEY)
# # groq_llm = ChatGroq(model="qwen/qwen3.6-27b", api_key = GROQ_API_KEY)
# # 
# # def load_dataset(path:str, agent = agent):
# #   ...
# #   # Dynamic file reader generator code logic handled securely inside Streamlit cache and buffer.
# # 
# # def perform_eda_func(data, agent):
# #   ...
# # 
# # advance_prompt = "..."
# # new_prompt = "..."
# # 
# ==========================================