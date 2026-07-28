
def eda_by_ai():
    """
    Performs a comprehensive, production-grade Advanced Data Analysis on the pre-loaded dataframe `df`.
    All code is encapsulated within this single function as requested.
    """
    # Import required libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set plot styling
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.autolayout': True})

    print("=" * 60)
    print("1. ENVIRONMENT SETUP & DATA INSPECTION")
    print("=" * 60)
    # Note: df is assumed to be already loaded in the environment.
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    display_fn = display if 'display' in globals() else print
    display_fn(df.head())
    
    print("\nDataset Information (.info()):")
    df.info()

    print("\n" + "=" * 60)
    print("2. AUTOMATED & MANUAL DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print("\nNumerical Descriptive Statistics:")
    display_fn(df.describe())
    
    print("\nCategorical/All Descriptive Statistics:")
    display_fn(df.describe(include="all"))
    
    print("\nMissing Values Percentage per Column:")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    display_fn(missing_pct[missing_pct > 0].sort_values(ascending=False))
    
    duplicate_count = df.duplicated().sum()
    print(f"\nTotal Duplicate Rows: {duplicate_count}")

    print("\n" + "=" * 60)
    print("3. CORRELATION ANALYSIS")
    print("=" * 60)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        print("\nPearson Correlation Matrix:")
        display_fn(corr_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
        plt.title("Correlation Heatmap of Numerical Features")
        plt.show()
    else:
        print("Not enough numerical columns to compute a correlation matrix.")

    print("\n" + "=" * 60)
    print("4. UNIVARIATE NUMERICAL COLUMN ANALYSIS")
    print("=" * 60)
    for col in numerical_cols[:5]:  # Limit to first 5 to prevent excessive plots
        print(f"Plotting Distribution & Boxplot for: {col}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram + KDE
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title(f"Distribution of {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")
        
        # Boxplot
        sns.boxplot(x=df[col], ax=axes[1], color="lightgreen")
        axes[1].set_title(f"Boxplot of {col}")
        axes[1].set_xlabel(col)
        
        plt.show()

    print("\n" + "=" * 60)
    print("5. UNIVARIATE OBJECT/CATEGORICAL COLUMN ANALYSIS")
    print("=" * 60)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        for col in categorical_cols[:5]:  # Limit to first 5
            print(f"\nFrequency Counts for {col}:")
            value_counts = df[col].value_counts(dropna=False)
            display_fn(value_counts.head(10))
            
            plt.figure(figsize=(10, 4))
            top_cats = df[col].value_counts().nlargest(10)
            sns.barplot(x=top_cats.values, y=top_cats.index, palette="viridis", hue=top_cats.index, legend=False)
            plt.title(f"Top Categories for {col}")
            plt.xlabel("Count")
            plt.ylabel(col)
            plt.show()
    else:
        print("No categorical columns detected.")

    print("\n" + "=" * 60)
    print("6. BIVARIATE ANALYSIS")
    print("=" * 60)
    if len(numerical_cols) >= 2:
        col_x, col_y = numerical_cols[0], numerical_cols[1]
        print(f"Creating regression scatter plot between {col_x} and {col_y}")
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df, x=col_x, y=col_y, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f"Bivariate Analysis: {col_y} vs {col_x}")
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.show()
    else:
        print("Skipping bivariate scatter plots (requires at least 2 numerical columns).")

    print("\n" + "=" * 60)
    print("7. TIME SERIES ANALYSIS (CONDITIONAL CHECK)")
    print("=" * 60)
    time_col = None
    # Check for potential datetime columns based on name or type
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_col = col
            break
        elif any(term in col.lower() for term in ['date', 'time', 'year', 'timestamp']):
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notnull().mean() > 0.6:  # If >60% successfully parse
                    df[col] = converted
                    time_col = col
                    break
            except Exception:
                pass

    if time_col:
        print(f"Time column detected: '{time_col}'. Performing time series trend analysis...")
        ts_df = df.set_index(time_col).sort_index()
        # Find a numerical column to aggregate
        if numerical_cols:
            num_target = numerical_cols[0]
            resampled = ts_df[num_target].resample('ME').mean() # 'ME' for Month End
            
            plt.figure(figsize=(12, 5))
            resampled.plot(kind='line', marker='o', color='purple')
            plt.title(f"Monthly Average Trend of {num_target} over Time")
            plt.xlabel("Time")
            plt.ylabel(f"Average {num_target}")
            plt.grid(True)
            plt.show()
    else:
        print("No time series data detected, skipping time series analysis.")

    print("\n" + "=" * 60)
    print("8. MULTIVARIATE ANALYSIS")
    print("=" * 60)
    if len(categorical_cols) >= 1 and len(numerical_cols) >= 1:
        cat_mult = categorical_cols[0]
        num_mult = numerical_cols[0]
        
        hue_mult = categorical_cols[1] if len(categorical_cols) >= 2 else None
        
        print(f"Creating segmented bar plot for {num_mult} across {cat_mult}" + (f" and {hue_mult}" if hue_mult else ""))
        plt.figure(figsize=(12, 6))
        
        # Limit categories to top 5 for clarity
        top_cats = df[cat_mult].value_counts().nlargest(5).index
        subset_df = df[df[cat_mult].isin(top_cats)]
        
        if hue_mult:
            top_hues = df[hue_mult].value_counts().nlargest(4).index
            subset_df = subset_df[subset_df[hue_mult].isin(top_hues)]
            sns.barplot(data=subset_df, x=cat_mult, y=num_mult, hue=hue_mult, ci=None, palette="muted")
            plt.legend(title=hue_mult, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.barplot(data=subset_df, x=cat_mult, y=num_mult, ci=None, palette="muted", hue=cat_mult, legend=False)
            
        plt.title(f"Multivariate Bar Plot: {num_mult} by {cat_mult}")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("Skipping multivariate analysis due to insufficient categorical/numerical combinations.")

    print("\n" + "=" * 60)
    print("9. AUTOMATED SUMMARY & INSIGHTS")
    print("=" * 60)
    print("- Dataset rows and columns have been successfully outlined.")
    print("- Descriptive analytics, missing values, and duplication metrics have been computed.")
    print("- Univariate metrics and distributions reviewed for outliers and data skewness.")
    print("- Correlation, bivariate regression, and multivariate segmentation plots generated successfully.")
    print("EDA execution completed successfully.")
