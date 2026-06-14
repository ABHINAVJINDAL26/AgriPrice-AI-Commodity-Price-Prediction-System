import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw_data.csv")

def generate_synthetic_data(num_days=180):
    np.random.seed(42)
    
    commodities = ["Onion", "Potato", "Tomato", "Gram", "Tur"]
    states = ["Maharashtra", "Karnataka", "Uttar Pradesh", "Delhi"]
    
    # Base prices (Rs/Quintal) and seasonal profiles
    base_info = {
        "Onion": {"base": 1500.0, "noise": 120.0, "trend": 0.8, "season_mult": {11: 0.8, 12: 0.8, 1: 0.8, 5: 1.3, 6: 1.4, 7: 1.5, 8: 1.4, 9: 1.3, 10: 1.2}},
        "Potato": {"base": 1000.0, "noise": 70.0, "trend": 0.4, "season_mult": {1: 0.7, 2: 0.7, 3: 0.8, 10: 1.2, 11: 1.2, 12: 1.1}},
        "Tomato": {"base": 1800.0, "noise": 200.0, "trend": 1.2, "season_mult": {6: 1.6, 7: 1.8, 8: 1.5, 11: 0.7, 12: 0.7, 1: 0.8}},
        "Gram": {"base": 5500.0, "noise": 350.0, "trend": 1.5, "season_mult": {3: 0.9, 4: 0.9, 5: 0.9, 9: 1.1, 10: 1.15, 11: 1.1}},
        "Tur": {"base": 7200.0, "noise": 450.0, "trend": 2.0, "season_mult": {1: 0.9, 2: 0.95, 10: 1.1, 11: 1.15, 12: 1.1}}
    }
    
    # State-wise price differences (multipliers)
    state_mults = {
        "Maharashtra": 0.9,      # Major producer for onion/pulses
        "Karnataka": 0.95,       # Major producer
        "Uttar Pradesh": 1.0,    # Average
        "Delhi": 1.25            # Metacity (high demand, high price)
    }

    start_date = datetime.now() - timedelta(days=num_days)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    records = []
    
    for date in dates:
        month = date.month
        for commodity in commodities:
            info = base_info[commodity]
            for state in states:
                # Calculate base price with trend
                day_idx = (date - start_date).days
                price = info["base"] + (info["trend"] * day_idx)
                
                # Apply seasonal multiplier
                mult = info["season_mult"].get(month, 1.0)
                price *= mult
                
                # Apply state-wise multiplier
                price *= state_mults[state]
                
                # Add random noise
                price += np.random.normal(0, info["noise"])
                
                # Ensure price is positive and realistic
                price = max(5.0, round(price, 2))
                
                # Add festival price bump in October
                if month == 10:
                    price *= 1.10
                
                # Generate realistic volume
                volume = int(np.random.normal(5000, 1000) * (1 / mult) * (1 / state_mults[state]))
                volume = max(100, volume)
                
                records.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Commodity": commodity,
                    "State": state,
                    "Price": price,
                    "Volume": volume
                })
                
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Generated synthetic raw data at: {RAW_DATA_PATH} ({len(df)} records)")

if __name__ == "__main__":
    generate_synthetic_data()
