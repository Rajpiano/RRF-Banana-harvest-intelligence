import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def ensure_week_index(df):
    """
    Ensures a continuous 'Week_Index' exists.
    """
    df = df.copy()
    if 'Week_Index' not in df.columns:
        min_year = df['Year'].min()
        df['Week_Index'] = (df['Year'] - min_year) * 52 + df['Week']
    return df

def create_lag_features(df, target_col, source_col, lags):
    """
    Generic lag creator.
    target_col: The column we want to predict (e.g., Harvest)
    source_col: The column we use to predict (e.g., Bagging)
    """
    df_processed = df.copy()
    feature_cols = []
    for lag in lags:
        col_name = f'{source_col}_Lag_{lag}'
        df_processed[col_name] = df_processed[source_col].shift(lag)
        feature_cols.append(col_name)
    
    df_processed = df_processed.dropna()
    return df_processed, feature_cols

def train_model(df, target_col, source_col, lags):
    """
    Generic trainer.
    """
    # 0. Preprocess
    df = ensure_week_index(df)
    
    # 1. Create Features
    df_lags, feature_cols = create_lag_features(df, target_col, source_col, lags)
    
    # 2. Split
    X = df_lags[feature_cols]
    y = df_lags[target_col]
        
    # Time-based split (80/20)
    split_index = int(len(df_lags) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, predictions), 
        'R2': r2_score(y_test, predictions),
        'Std_Error': np.std(y_test - predictions) # Useful for confidence intervals
    }
    
    return model, metrics, feature_cols

def train_full_system(df):
    """
    Trains both stages of the biological cycle.
    """
    # Stage 1: Maturation (Bagging -> Harvest, 12-17w)
    harvest_model, harvest_metrics, harvest_feats = train_model(
        df, target_col='Harvest_Count', source_col='Bagging_Count', lags=[12,13,14,15,16,17]
    )
    
    # Stage 2: Succession (Harvest -> Daughter Bagging, 25-30w)
    bagging_model, bagging_metrics, bagging_feats = train_model(
        df, target_col='Bagging_Count', source_col='Harvest_Count', lags=[25,26,27,28,29,30]
    )
    
    return {
        'harvest_model': harvest_model,
        'harvest_metrics': harvest_metrics,
        'bagging_model': bagging_model,
        'bagging_metrics': bagging_metrics
    }

def predict_long_term(df, models, weeks_ahead=52):
    """
    Predicts 1 year ahead using the chain:
    Known Harvest -> Predict Future Bagging -> Predict Future Harvest
    """
    df = ensure_week_index(df)
    last_idx = df['Week_Index'].max()
    
    # We need a continuous simulation dataframe extending into the future
    future_indices = range(last_idx + 1, last_idx + weeks_ahead + 1)
    
    # Initialize future dataframe with Week/Year/Color logic
    future_data = []
    
    # Helper for Color Cycle (User logic: 30 week cycle)
    # Re-implement generator logic to be consistent
    BAG_COLORS = ['Yellow', 'Blue', 'Green', 'Red', 'Black']
    STRING_COLORS = ['White', 'Blue', 'Red', 'Green', 'Yellow', 'Black']
    
    for idx in future_indices:
        # Calculate Year/Week
        # Base it off the last known row's date logic
        weeks_from_start = idx - 1 # Assuming index 1 start
        # This is a bit rough, but sufficient for display
        # Better: use the last row and increment
        
        cycle_idx = (idx - 1) % 30
        
        # Consistent Color Logic with Data Generator
        # String changes every week
        str_color = STRING_COLORS[cycle_idx % 6]
        # Bag changes every 6 weeks
        bag_color = BAG_COLORS[(cycle_idx // 6) % 5]
        
        future_data.append({
            'Week_Index': idx,
            'Bag_Color': bag_color,
            'String_Color': str_color,
            'Bagging_Count': np.nan, # To be predicted
            'Harvest_Count': np.nan  # To be predicted
        })
        
    future_df = pd.DataFrame(future_data)
    
    # Concatenate Past + Future
    sim_df = pd.concat([df, future_df], ignore_index=True)
    
    # ---------------------------------------------------------
    # Step 1: Predict Future Bagging (Succession)
    # ---------------------------------------------------------
    # We iterate forward. For each future week, we look back at known (or predicted) Harvests.
    # Note: Succession lags are 25-30. So we can predict Bagging at T using Harvest at T-30.
    # Since we have Known Harvests up to T=0, we can predict Bagging up to T+25 safely without feedback loop.
    
    bag_model = models['bagging_model']
    har_model = models['harvest_model']
    
    for t in future_indices:
        # Predict Bagging at t
        # Input features: Harvest at t-30, t-29...
        feats = {}
        valid = True
        for lag in [25,26,27,28,29,30]:
            # Look up harvest
            try:
                val = sim_df.loc[sim_df['Week_Index'] == (t - lag), 'Harvest_Count'].values[0]
                if np.isnan(val): valid = False # Can't predict if input is missing
                feats[f'Harvest_Count_Lag_{lag}'] = val
            except IndexError:
                valid = False
        
        if valid:
            X = pd.DataFrame([feats])
            # Ensure order
            X = X[[f'Harvest_Count_Lag_{l}' for l in [25,26,27,28,29,30]]]
            pred_bag = bag_model.predict(X)[0]
            # Update Simulation
            sim_df.loc[sim_df['Week_Index'] == t, 'Bagging_Count'] = max(0, int(pred_bag))
            
    # ---------------------------------------------------------
    # Step 2: Predict Future Harvest (Maturation)
    # ---------------------------------------------------------
    # Now we have Future Bagging (predicted above). We use it to predict Future Harvest.
    
    for t in future_indices:
        feats = {}
        valid = True
        for lag in [12,13,14,15,16,17]:
            try:
                # This might pick up the Bagging we just predicted above!
                val = sim_df.loc[sim_df['Week_Index'] == (t - lag), 'Bagging_Count'].values[0]
                if np.isnan(val): valid = False
                feats[f'Bagging_Count_Lag_{lag}'] = val
            except IndexError:
                valid = False
                
        if valid:
            X = pd.DataFrame([feats])
            X = X[[f'Bagging_Count_Lag_{l}' for l in [12,13,14,15,16,17]]]
            pred_har = har_model.predict(X)[0]
            sim_df.loc[sim_df['Week_Index'] == t, 'Harvest_Count'] = max(0, int(pred_har))
            
    # Return just the future part
    result = sim_df[sim_df['Week_Index'] > last_idx].copy()
    
    # Calculate Year/Week for display (simple increment)
    last_year = df.iloc[-1]['Year']
    last_week = df.iloc[-1]['Week']
    
    years = []
    weeks = []
    curr_y, curr_w = last_year, last_week
    for _ in range(len(result)):
        curr_w += 1
        if curr_w > 52:
            curr_y += 1
            curr_w = 1
        years.append(curr_y)
        weeks.append(curr_w)
        
    result['Year'] = years
    result['Week'] = weeks
    
    # Probability Intervals (95% CI)
    # Harvest Error
    sigma = models['harvest_metrics']['Std_Error']
    result['Lower_Bound'] = result['Harvest_Count'] - (1.96 * sigma)
    result['Upper_Bound'] = result['Harvest_Count'] + (1.96 * sigma)
    
    # Fill NAs with 0 to prevent crash during Int conversion
    result['Lower_Bound'] = result['Lower_Bound'].fillna(0).clip(lower=0).astype(int)
    result['Upper_Bound'] = result['Upper_Bound'].fillna(0).astype(int)
    result['Bagging_Count'] = result['Bagging_Count'].fillna(0).astype(int)
    result['Harvest_Count'] = result['Harvest_Count'].fillna(0).astype(int)
    
    return result
