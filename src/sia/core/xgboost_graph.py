import sys
sys.path.insert(0, '../')
import os

import pandas as pd

dir_addr = "/home/erfan/Projects/mobicom-2024/data/Raw_data/cluster_1/slicing_1/scheduling_0"

# 1 Load and process the data for 1 RESERVATION
# Step 1: Get sorted list of directories
reservations = sorted([f for f in os.listdir(f"{dir_addr}") if os.path.isdir(os.path.join(f"{dir_addr}", f))])

# Step 2: Initialize DataFrame to store data for the plot
plot_data = pd.DataFrame()

# Step 3: Iterate through the directories
for res in reservations[:1]:
    print(f"{dir_addr}/{res}")
    
    # Step 4: Get sorted list of CSV files
    csv_files = sorted([f for f in os.listdir(f"{dir_addr}/{res}/bs") if f.endswith('.csv') and not f.startswith('enb')])
    
    # Step 5: Iterate through the CSV files
    for csv_file in csv_files:
        df = pd.read_csv(
            f"{dir_addr}/{res}/bs/{csv_file}",
            usecols=['Timestamp', 'slice_id', 'slice_prb', 'tx_brate downlink [Mbps]', 'tx_pkts downlink', 'dl_buffer [bytes]', 'num_ues']
        )
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df['csv_name'] = csv_file
        
        # Step 6: Append data to plot_data DataFrame
        plot_data = pd.concat([plot_data, df], ignore_index=True)
        
        print(plot_data.head())