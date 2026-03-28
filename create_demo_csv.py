import pandas as pd
import random
import os

# 1. Point exactly to your raw datasets
# Using Monday for pure Benign traffic, Friday for DDoS Attacks
BENIGN_SOURCE = "data/raw/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"
ATTACK_SOURCE = "data/raw/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# 2. Where to save your thesis presentation files
BENIGN_OUTPUT = "demo_benign_traffic.csv"
ATTACK_OUTPUT = "demo_attack_ddos.csv"

def generate_demo_file(file_path, target_type, output_name, num_rows=25):
    print(f"Loading {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # Strip invisible spaces from column names (Classic CIC-IDS quirk)
        df.columns = df.columns.str.strip()
        
        # Find the label column
        label_col = None
        for col in df.columns:
            if col.lower() in ['label', 'class', 'attack_type']:
                label_col = col
                break
                
        if not label_col:
            print(f"❌ Error: Could not find Label column in {file_path}")
            return

        # Filter the data
        if target_type == 'BENIGN':
            # Grab only normal traffic
            filtered_df = df[df[label_col].astype(str).str.upper() == 'BENIGN']
        else:
            # Grab only the hacker traffic
            filtered_df = df[df[label_col].astype(str).str.upper() != 'BENIGN']
            
        # Extract a random sample so the file is lightweight and fast for the UI
        if len(filtered_df) < num_rows:
            demo_df = filtered_df
        else:
            demo_df = filtered_df.sample(n=num_rows, random_state=42)
            
        # Save to a clean CSV
        demo_df.to_csv(output_name, index=False)
        print(f"✅ SUCCESS: Saved {len(demo_df)} {target_type} flows to -> {output_name}\n")
        
    except FileNotFoundError:
        print(f"❌ Could not find {file_path}. Please check your folder structure.")

print("--- FALCON Sandbox Data Generator ---")
# Generate the Benign File
generate_demo_file(BENIGN_SOURCE, 'BENIGN', BENIGN_OUTPUT)

# Generate the Attack File
generate_demo_file(ATTACK_SOURCE, 'ATTACK', ATTACK_OUTPUT)