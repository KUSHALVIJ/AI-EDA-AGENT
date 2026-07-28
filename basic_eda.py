
import pandas as pd
import numpy as np

def perform_eda(df: pd.DataFrame):
    """
    Performs basic Exploratory Data Analysis (EDA) on a given pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): The dataframe to analyze.
    
    Returns:
    dict: A dictionary containing various EDA metrics and summaries.
    """
    print("="*50)
    print("               EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # 1. Basic Shape
    print(f"\n[1. Dataset Shape]")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # 2. Duplicate Check
    print(f"\n[2. Duplicate Rows]")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates} ({round(duplicates/len(df)*100, 2)}% of dataset)")
    
    # 3. Missing Values Analysis
    print(f"\n[3. Missing Values Summary]")
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage (%)': missing_percent.round(2)
    })
    # Filter only columns with missing values and sort descending
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    
    if missing_df.empty:
        print("Great news! There are no missing values in this dataset.")
    else:
        print(missing_df)
        
    # 4. Column Data Types & Non-Null Counts
    print(f"\n[4. Column Information]")
    info_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum()
    })
    print(info_df)
    
    # 5. Statistical Summary (Numerical)
    print(f"\n[5. Numerical Summary]")
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        print(num_df.describe().T)
    else:
        print("No numerical columns found.")
        
    # 6. Statistical Summary (Categorical)
    print(f"\n[6. Categorical Summary]")
    cat_df = df.select_dtypes(include=['object', 'category'])
    if not cat_df.empty:
        print(cat_df.describe().T)
    else:
        print("No categorical columns found.")
        
    print("\n" + "="*50)
    print("               EDA COMPLETED")
    print("="*50)
    
    # Optionally return a dictionary of the core metrics for programmatic use
    return {
        "shape": df.shape,
        "duplicates": duplicates,
        "missing_summary": missing_df,
        "column_info": info_df
    }

# ==========================================
# EXAMPLE USAGE:
# ==========================================
if __name__ == "__main__":
    # Creating a dummy dataset to test the function
    data = {
        'age': [25, 30, np.nan, 45, 30, 25],
        'salary': [50000, 60000, 80000, np.nan, 60000, 50000],
        'department': ['HR', 'IT', 'IT', 'Finance', 'IT', 'HR'],
        'is_active': [True, True, False, True, True, True]
    }
    
    sample_df = pd.DataFrame(data)
    
    # Run the EDA function
    eda_results = perform_eda(sample_df)
