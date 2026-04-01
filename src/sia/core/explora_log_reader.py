# This function will receive a log file address and return a pandas dataframe from the log data
# Steps:
# receives a log file address
# cehcks if the file exists
# receives the processes data storing address
# checks if the processes file already exists or not
# if do_again is not set to True, it will return the processes dataframe
# if do_again is True, then processes the file
# Go line by line, if there is a specific phrase in the line then it contains the info we are looking for
# Extract the info and store it in a pandas

import os
import pandas as pd
import re

class LogFileProcessor:
    
    def __init__(self, log_file_path: str, process_data_path: str, file_name: str, do_again: bool = False):
        """
        Initialize the LogFileProcessor class with the log file path, process data path, file name, and do_again flag.

        :param log_file_path: str, path to the log file to be processed.
        :param process_data_path: str, path to the directory where the processed data will be stored.
        :param file_name: str, the base file name used to save the processed data.
        :param do_again: bool, whether to reprocess the log file if data already exists.
        """
        self.log_file_path = log_file_path
        self.process_data_path = process_data_path
        self.file_name = file_name
        self.do_again = do_again
        self.processed_data = None  # Tuple containing (experiment_data, decision_data)

    def check_file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists at the given path.

        :param file_path: str, path to the file to check.
        :return: bool, True if the file exists, False otherwise.
        """
        return os.path.isfile(file_path)

    def load_existing_data(self):
        """
        Load existing processed data if the processed CSV files exist.
        """
        experiment_data_path = os.path.join(self.process_data_path, f"{self.file_name}_experiment_data.csv")
        decision_data_path = os.path.join(self.process_data_path, f"{self.file_name}_decision_data.csv")

        # Check if both files exist
        if self.check_file_exists(experiment_data_path) and self.check_file_exists(decision_data_path):
            print(f"Loading existing data from {experiment_data_path}")
            experiment_data = pd.read_csv(experiment_data_path)
            decision_data = pd.read_csv(decision_data_path)
            self.processed_data = (experiment_data, decision_data)
        else:
            print(f"No existing processed data found at {self.process_data_path}")
            self.processed_data = None

    def save_processed_data(self, experiment_data: pd.DataFrame, decision_data: pd.DataFrame):
        """
        Save the processed data to the specified directory using the provided file name.

        :param experiment_data: pd.DataFrame, the cleaned experiment data.
        :param decision_data: pd.DataFrame, the decision data.
        """
        # Make sure the directory exists
        if not os.path.exists(self.process_data_path):
            os.makedirs(self.process_data_path)

        # Join the directory path with the user-provided file name
        experiment_data_path = os.path.join(self.process_data_path, f"{self.file_name}_experiment_data.csv")
        decision_data_path = os.path.join(self.process_data_path, f"{self.file_name}_decision_data.csv")

        print(f"Saving experiment data to {experiment_data_path}")
        experiment_data.to_csv(experiment_data_path, index=False)

        print(f"Saving decision data to {decision_data_path}")
        decision_data.to_csv(decision_data_path, index=False)

    def process_log_file(self):
        """
        Process the log file line by line, extracting relevant information and storing it in a DataFrame.
        """
        log_received_data = []
        log_prb_decision_data = []
        log_send_to_du_data = []
        agent_experiment_data = []

        timestep = 0
        valid_entry = False

        print(f"Processing log file: {self.log_file_path}")
        try:
            with open(self.log_file_path, 'r') as log_file:
                for line in log_file:
                    if "Received data:" in line:
                        valid_entry = True
                        timestep += 1
                        received_data = self.parse_received_data(line)
                        log_received_data.append({'timestep': timestep, 'data': received_data})

                    elif "Using previous socket data" in line:
                        valid_entry = False

                    elif "Action means slice_prb" in line and valid_entry:
                        prb_decision, sending_data = self.parse_action_means_data(line)
                        log_prb_decision_data.append({'timestep': timestep, 'data': prb_decision})
                        log_send_to_du_data.append({'timestep': timestep, 'data': sending_data})
            
            # Extract agent-specific experiment data from received data
            log_received_data_extracted_slices = self.extract_slice_data_from_log(log_received_data, timestep)
            agent_experiment_data.extend(log_received_data_extracted_slices)

        except FileNotFoundError:
            print(f"File not found: {self.log_file_path}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Combine the extracted data into DataFrames
        received_df = pd.DataFrame(agent_experiment_data)
        prb_df = pd.DataFrame(log_prb_decision_data)
        sending_df = pd.DataFrame(log_send_to_du_data)

        return received_df, prb_df, sending_df

    def extract_slice_data_from_log(self, log_received_data, timestep):
        """
        Extracts slice-specific experiment data from the log received data.

        :param log_received_data: list of dictionaries with 'timestep' and 'data'.
        :param timestep: int, the current timestep.
        :return: list of dictionaries with extracted slice-specific data.
        """
        agent_experiment_data = []
        
        for entry in log_received_data:
            for data in entry['data']:
                extracted_info = data.split(",")
                if self.check_received_data_elements_by_length(extracted_info):
                    agent_experiment_data.append({
                        "timestep": int(entry['timestep']),
                        "slice_id": int(extracted_info[0]),
                        "tx_brate": float(extracted_info[2]),
                        "tx_pckts": float(extracted_info[5]),
                        "dl_buffer": float(extracted_info[1]),
                        "slice_prb": int(extracted_info[4]),
                    })
        
        return agent_experiment_data
    
    def check_received_data_elements_by_length(self, array, length=6):
        """
        Checks if the received data array has the expected length.

        :param array: list, the data array to check.
        :param length: int, the expected length of the array.
        :return: bool, True if the array length matches the expected length.
        """
        return len(array) == length

    def parse_received_data(self, line: str):
        """
        Extracts and processes received data from a line.
        """
        return line.split("Received data:")[1][2:-2].split('\\n')

    def parse_action_means_data(self, line: str):
        """
        Extract the PRB decision and sending data from a line that starts with "Action means slice_prb".

        Example line:
        "Action means slice_prb [36, 3, 11] (slice_rbg [12, 1, 4]), sched [0, 1, 0]"

        :param line: str, the log line containing the action means data.
        :return: tuple, containing the extracted prb_decision and sending_data.
        """
        # Extract the PRB decision part: slice_prb [36, 3, 11]
        prb_match = re.search(r"slice_prb \[([0-9, ]+)\]", line)
        if prb_match:
            prb_decision = [int(x) for x in prb_match.group(1).split(', ')]
        else:
            prb_decision = []

        # Extract the sending data part: sched [0, 1, 0]
        sched_match = re.search(r"sched \[([0-9, ]+)\]", line)
        if sched_match:
            sending_data = [int(x) for x in sched_match.group(1).split(', ')]
        else:
            sending_data = []

        return prb_decision, sending_data

    def clean_process_experiment_data(self, received_data: pd.DataFrame, prb_decisions: pd.DataFrame, scheduling_policy: pd.DataFrame):
        """
        Cleans and processes the extracted raw data into two synchronized DataFrames.

        :param received_data: DataFrame containing the raw received data.
        :param prb_decisions: DataFrame containing the prb decision data.
        :param scheduling_policy: DataFrame containing the scheduling policy data.
        :return: Two DataFrames, one for the cleaned experiment data and one for decision data.
        """
        cleaned_experiment_data = []
        cleaned_decision_data = []
        
        timestep_counter = 0
        for i in range(1, received_data['timestep'].max() + 1):
            data_at_timestep = received_data[received_data['timestep'] == i]
            
            if self.is_complete_data(data_at_timestep):
                timestep_counter += 1
                
                # Group, average, and calculate std for data by slice_id
                timestep_data = self.group_and_aggregate_data(data_at_timestep)
                cleaned_experiment_data.extend(self.build_experiment_data(timestep_counter, timestep_data))
                
                # Get the PRB and scheduling policy decisions for the current timestep
                policy_state = scheduling_policy[scheduling_policy['timestep'] == i]
                prb_state = prb_decisions[prb_decisions['timestep'] == i]
                
                decision = self.make_decision(prb_state, policy_state, timestep_counter)
                cleaned_decision_data.append(decision)
        return pd.DataFrame(cleaned_experiment_data), pd.DataFrame(cleaned_decision_data)

    def is_complete_data(self, data):
        """
        Check if the data contains at least one unique slice_id.
        """
        return len(data['slice_id'].unique()) >= 1

    def group_and_aggregate_data(self, data):
        """
        Group the data by slice_id and calculate various metrics, including:
        - Mean
        - Standard Deviation (std)
        - Range
        - IQR (Interquartile Range)
        - MAD (Median Absolute Deviation)
        - Relative Range (Range / Mean)
        """
        # Group by 'slice_id' and calculate mean, std, range, IQR, MAD, and relative range for each column
        grouped = data.groupby('slice_id').agg({
            'tx_brate': [
                'mean', 
                # 'std', 
                # lambda x: x.max() - x.min(),  # Range
                lambda x: x.quantile(0.75) - x.quantile(0.25),  # IQR
                # lambda x: (x - x.median()).abs().median(),  # MAD
                # lambda x: (x.max() - x.min()) / x.mean() if x.mean() != 0 else 0  # Relative Range
            ],
            'tx_pckts': [
                'mean', 
                # 'std', 
                # lambda x: x.max() - x.min(),  # Range
                lambda x: x.quantile(0.75) - x.quantile(0.25),  # IQR
                # lambda x: (x - x.median()).abs().median(),  # MAD
                # lambda x: (x.max() - x.min()) / x.mean() if x.mean() != 0 else 0  # Relative Range
            ],
            'dl_buffer': [
                'mean', 
                # 'std', 
                # lambda x: x.max() - x.min(),  # Range
                lambda x: x.quantile(0.75) - x.quantile(0.25),  # IQR
                # lambda x: (x - x.median()).abs().median(),  # MAD
                # lambda x: (x.max() - x.min()) / x.mean() if x.mean() != 0 else 0  # Relative Range
            ],
        }).reset_index()

        # Rename columns for clarity
        grouped.columns = [
            'slice_id', 
            'tx_brate', 'tx_brate_iqr',
            'tx_pckts', 'tx_pckts_iqr',
            'dl_buffer','dl_buffer_iqr'
        ]
        # 
        
        return grouped

    def build_experiment_data(self, timestep, grouped_data):
        """
        Build a list of dictionaries representing the cleaned experiment data for a given timestep,
        including both the mean and standard deviation values.
        """
        return [
            {
                "timestep": timestep,
                "slice_id": int(item['slice_id']),
                "tx_brate": item['tx_brate'],
                "tx_brate_iqr": item['tx_brate_iqr'],
                "tx_pckts": item['tx_pckts'],
                "tx_pckts_iqr": item['tx_pckts_iqr'],
                "dl_buffer": item['dl_buffer'],
                "dl_buffer_iqr": item['dl_buffer_iqr']
            }
            for index, item in grouped_data.iterrows()
        ]

    def make_decision(self, prb_data, policy_state, timestep):
        """
        Make a decision based on the PRB data and the scheduling policy data.
        """
        prb_state = self.define_prb_state(prb_data)
        scheduling_policy_state = self.define_scheduling_policy_state(policy_state)

        return {
            "timestep": timestep,
            "decision": tuple(prb_state + scheduling_policy_state),
            "prb_decision": prb_state,
            "policy_decision": scheduling_policy_state
        }

    def define_prb_state(self, prb_data):
        """
        Define the PRB state based on the PRB decision data.
        """
        return [int(i) for i in prb_data['data'].iloc[0]]

    def define_scheduling_policy_state(self, df_timestep_scheduling_policy_data):
        """
        Define the scheduling policy state based on the scheduling policy data.
        """
        return [int(i) for i in df_timestep_scheduling_policy_data['data'].iloc[0]]

    def load_or_process_data(self):
        """
        Either load existing data or process the log file based on the `do_again` flag.
        """
        experiment_data_path = os.path.join(self.process_data_path, f"{self.file_name}_experiment_data.csv")
        decision_data_path = os.path.join(self.process_data_path, f"{self.file_name}_decision_data.csv")
        
        if not self.do_again:
            # Check if processed files exist
            if self.check_file_exists(experiment_data_path) and self.check_file_exists(decision_data_path):
                print("Processed data already exists. Loading from CSV files.")
                self.load_existing_data()
            else:
                print("Processed data not found, processing the log file.")
                received_data, prb_decisions, scheduling_policy = self.process_log_file()
                experiment_data, decision_data = self.clean_process_experiment_data(received_data, prb_decisions, scheduling_policy)
                
                # Save the newly processed data
                self.save_processed_data(experiment_data, decision_data)
                self.processed_data = (experiment_data, decision_data)
        else:
            print("Reprocessing the log file.")
            received_data, prb_decisions, scheduling_policy = self.process_log_file()
            experiment_data, decision_data = self.clean_process_experiment_data(received_data, prb_decisions, scheduling_policy)
            
            # Save the reprocessed data
            self.save_processed_data(experiment_data, decision_data)
            self.processed_data = (experiment_data, decision_data)

    def get_processed_data(self):
        """
        Get the processed data (either loaded from file or freshly processed).

        :return: tuple of pd.DataFrame, the cleaned experiment data and decision data.
        """
        if self.processed_data is None:
            self.load_or_process_data()
        return self.processed_data