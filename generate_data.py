import pandas as pd
import numpy as np

# Configuration
YEARS = 4
WEEKS = 52 * YEARS
START_YEAR = 2022

# Color Config (User Logic: 5 Bags * 6 Strings = 30 combinations)
BAG_COLORS = ['Yellow', 'Blue', 'Green', 'Red', 'Black']
STRING_COLORS = ['White', 'Blue', 'Red', 'Green', 'Yellow', 'Black']

# Lag Config
# 1. Bagging -> Harvest (Maturation)
MATURATION_LAGS = [12, 13, 14, 15, 16, 17]
MATURATION_WEIGHTS = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05]

# 2. Mother Harvest -> Daughter Bagging (Succession)
# User said: "Harvest of mother to Bagging of Daughter will take approx 22-29 weeks"
# Let's use a range slightly wider to add realism
SUCCESSION_MIN_LAG = 25
SUCCESSION_MAX_LAG = 32

np.random.seed(99) # Fixed seed

# Initialize Arrays
bagging_counts = np.zeros(WEEKS + 100) # Buffer for future overflow
harvest_counts = np.zeros(WEEKS + 100)

# Seed the first 6 months with random "Mother" Bagging to start the chain
# (Otherwise the farm is empty)
for i in range(26):
    bagging_counts[i] = np.random.randint(500, 1000)

# Simulation Loop
for t in range(WEEKS):
    
    # 1. Calculate Harvest occurring today (from Bagging 12-17 weeks ago)
    # ------------------------------------------------------------------
    todays_harvest = 0
    for lag, weight in zip(MATURATION_LAGS, MATURATION_WEIGHTS):
        if t - lag >= 0:
            todays_harvest += bagging_counts[t-lag] * weight
            
    # Add some noise (weather, loss, etc)
    if todays_harvest > 0:
        noise = np.random.normal(0, todays_harvest * 0.1)
        todays_harvest = max(0, int(todays_harvest + noise))
    
    harvest_counts[t] = todays_harvest
    
    # 2. Trigger Daughter Bagging (Succession Logic)
    # ------------------------------------------------------------------
    # "What we harvest today determines what we bag 25-30 weeks from now"
    # (Assuming 1:1 replacement ratio broadly, with some variation)
    
    if todays_harvest > 0:
        # Distribute this 'potential' into future bagging weeks
        # Not all daughters are ready at exact same time
        num_daughters = todays_harvest * np.random.uniform(0.9, 1.1) # Replacement rate
        
        # Spread these daughters over the succession window
        for _ in range(int(num_daughters)):
             lag = np.random.randint(SUCCESSION_MIN_LAG, SUCCESSION_MAX_LAG + 1)
             future_week = t + lag
             if future_week < len(bagging_counts):
                 bagging_counts[future_week] += 1
                 
    # 3. Add random external "New Planting" (Optional)
    # To keep the farm from dying out if replacement rate < 1, or to simulate expansion
    if t % 10 == 0:
        bagging_counts[t] += np.random.randint(0, 50)

# Build DataFrame
data = []
current_year = START_YEAR
current_week = 1

for t in range(WEEKS):
    # Color Logic
    # Cycle length = 30 weeks (5 Bags * 6 Strings)
    cycle_idx = (t) % 30
    
    # User Logic: "Yellow Bag and all different colours of Thread... then start other colour bag"
    # This means Bag changes only after String completes its full cycle (6 weeks).
    
    # String changes every week (cycle of 6)
    string_idx = cycle_idx % 6
    
    # Bag changes every 6 weeks (cycle of 5)
    bag_idx = (cycle_idx // 6) % 5
    
    row = {
        'Week_Index': t + 1,
        'Year': current_year,
        'Week': current_week,
        'Bag_Color': BAG_COLORS[bag_idx],
        'String_Color': STRING_COLORS[string_idx],
        'Bagging_Count': int(bagging_counts[t]),
        'Harvest_Count': int(harvest_counts[t])
    }
    data.append(row)
    
    current_week += 1
    if current_week > 52:
        current_year += 1
        current_week = 1

df = pd.DataFrame(data)

# Save to the "Live" file location
output_path = 'banana_farm_data.csv'
df.to_csv(output_path, index=False)

print(f"Generated advanced simulation data: {output_path}")
print(df.tail(10))
print("\nColor Check (First 10 rows):")
print(df[['Week_Index', 'Bag_Color', 'String_Color']].head(10))
