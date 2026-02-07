import pandas as pd

# 1. Load the History
df = pd.read_csv("data/user_transaction_history.csv")

# 2. FILTER: Isolate "High Regret" Transactions
# Logic: Show me places where I have regretted spending more than once.
regret_tx = df[df['regret'] == True]

# 3. CLUSTER: Group by Location (Lat/Lng)
# We count how many regrets happened at each unique coordinate
danger_zones = regret_tx.groupby(['merchant', 'lat', 'lng']).size().reset_index(name='regret_count')

# 4. THRESHOLD: Only flag places with > 2 regrets
# (For demo, we might just take them all)
confirmed_danger_zones = danger_zones[danger_zones['regret_count'] >= 1]

print("\n🚨 IDENTIFIED DANGER ZONES 🚨")
print("These coordinates should be sent to the iPhone to create Geofences:")
print("-" * 60)
print(confirmed_danger_zones[['merchant', 'lat', 'lng', 'regret_count']])

# 5. Export for the App
confirmed_danger_zones.to_json("data/danger_zones.json", orient="records")