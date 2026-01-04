import pandas as pd

def preprocess(data):
    # 1. List of features the model was trained on (in exact order)
    # Derived from models/capital_risk/dataset/test.csv
    feature_names = [
        'Credit_Score', 'DTI_Ratio', 'Employment_Years', 'Annual_Income', 
        'Num_Credit_Accounts', 'Payment_History_Score', 'Loan_Amount', 
        'Loan_Term_Months', 'Interest_Rate', 'LTV_Ratio', 'Is_Secured', 
        'Is_Metropolitan', 'GDP_Growth_Rate', 'Unemployment_Rate', 
        'Inflation_Rate', 'Risk_Free_Rate', 'VIX_Level', 'Loan_Age_Months', 
        'Origination_Year', 'Origination_Month', 'Origination_Quarter', 
        'Industry_Sector_Energy', 'Industry_Sector_Financial_Services', 
        'Industry_Sector_Healthcare', 'Industry_Sector_Manufacturing', 
        'Industry_Sector_Real_Estate', 'Industry_Sector_Retail', 
        'Industry_Sector_Technology', 'Geographic_Region_Northeast', 
        'Geographic_Region_Northwest', 'Geographic_Region_Southeast', 
        'Geographic_Region_Southwest', 'Geographic_Region_West', 
        'Loan_Purpose_Business', 'Loan_Purpose_Education', 
        'Loan_Purpose_Home_Purchase', 'Loan_Purpose_Personal', 
        'Loan_Purpose_Refinance'
    ]
    
    # 2. Extract the target column if it exists (for metric calculation)
    # The training data used 'Expected_Loss' as the target
    target_col = 'Expected_Loss'
    y = data[target_col] if target_col in data.columns else None
    
    # 3. Ensure X only contains the features in the correct order
    # This handles "missing features" and "unseen features" (like the target column)
    X = data.reindex(columns=feature_names, fill_value=0)
    
    return X, y