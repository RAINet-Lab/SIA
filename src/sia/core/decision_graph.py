import os
# os.environ['NX_CUGRAPH_AUTOCONFIG'] = 'True'
import pandas as pd
import numpy as np
from pyvis.network import Network

import networkx as nx
# nx.config.warnings_to_ignore.add("cache")

class DecisionGraph:
    def __init__(self) -> None:
        self.G = nx.DiGraph()
        self.net = None
        self.previous_state = None
        return

    def update_graph(self, symbolic_form_df: pd.DataFrame):
        """
        This function will receive a new state and update the G graph object for the state and the transitions
        """
        # Get values for slice 0 and slice 1
        slice_0_data = symbolic_form_df[symbolic_form_df['slice_id'] == 0]
        slice_1_data = symbolic_form_df[symbolic_form_df['slice_id'] == 1]
        
        # Create the current state as a combination of tx_brate (slice 0) and dl_buffer (slice 1)
        current_state = (
            slice_0_data['tx_brate downlink [Mbps]'].iloc[0],  # tx_brate from slice 0
            slice_1_data['dl_buffer [bytes]'].iloc[0]          # dl_buffer from slice 1
        )
        current_reward = symbolic_form_df['reward'].iloc[0]
        
        # Create a unique ID for the state
        state_id = '_'.join(map(str, current_state))  # Ensure state_id is a string
        
        # Create the action
        action = f"{symbolic_form_df['slice_prb'].iloc[0]}_{symbolic_form_df['scheduling_policy'].iloc[0]}"
        
        # Add the current state as a new node if it doesn't exist
        if state_id not in self.G.nodes:
            self.G.add_node(state_id, state=current_state, occurrence=0, total_reward=0, mean_reward=0)
        
        # Update the occurrence count and total reward for the current state node
        self.G.nodes[state_id]['occurrence'] += 1
        self.G.nodes[state_id]['total_reward'] += current_reward
        self.G.nodes[state_id]['mean_reward'] = self.G.nodes[state_id]['total_reward'] / self.G.nodes[state_id]['occurrence']
        
        # Update the transition from the previous state to the current state
        if self.previous_state is not None:
            if self.G.has_edge(self.previous_state, state_id):
                self.G[self.previous_state][state_id]['occurrence'] += 1
                self.G[self.previous_state][state_id]['total_reward'] += current_reward
                
                if 'actions' not in self.G[self.previous_state][state_id]:
                    self.G[self.previous_state][state_id]['actions'] = {}
                
                if action not in self.G[self.previous_state][state_id]['actions']:
                    self.G[self.previous_state][state_id]['actions'][action] = {'count': 0, 'total_reward': 0}
                
                self.G[self.previous_state][state_id]['actions'][action]['count'] += 1
                self.G[self.previous_state][state_id]['actions'][action]['total_reward'] += current_reward
            else:
                self.G.add_edge(self.previous_state, state_id, occurrence=1, total_reward=current_reward, 
                                actions={action: {'count': 1, 'total_reward': current_reward}})
            
            self.G[self.previous_state][state_id]['mean_reward'] = self.G[self.previous_state][state_id]['total_reward'] / self.G[self.previous_state][state_id]['occurrence']
            
            # Update mean rewards and probabilities for actions
            for act, data in self.G[self.previous_state][state_id]['actions'].items():
                data['mean_reward'] = data['total_reward'] / data['count']
                data['probability'] = data['count'] / self.G[self.previous_state][state_id]['occurrence']
        
        self.previous_state = state_id
        
        # Update probabilities and sizes for nodes and edges
        self._update_probabilities_and_sizes()
        
        return

    def _update_probabilities_and_sizes(self):
        """
        This function updates the probabilities and sizes of nodes and edges in the graph.
        """
        # Calculate the total occurrence count for nodes
        total_node_occurrence = sum(nx.get_node_attributes(self.G, 'occurrence').values())
        
        # Calculate the probability for each state node
        node_probabilities = {}
        node_sizes = {}
        
        for node, data in self.G.nodes(data=True):
            prob = data['occurrence'] / total_node_occurrence
            node_probabilities[node] = prob
            
            # Power law scaling for node sizes
            min_size = 30
            max_size = 150
            power = 0.5  # Adjust this value to control the scaling (smaller values = more pronounced differences)
            
            # Scale probability to size using power law
            scaled_size = min_size + (max_size - min_size) * (prob ** power)
            node_sizes[node] = scaled_size
        
        # Set the node probabilities and sizes in the graph
        nx.set_node_attributes(self.G, node_probabilities, 'probability')
        nx.set_node_attributes(self.G, node_sizes, 'size')
        
        # Calculate the probability for each edge
        edge_probabilities = {}
        edge_widths = {}
        for u, v, data in self.G.edges(data=True):
            total_transitions_from_u = sum(self.G[u][nbr]['occurrence'] for nbr in self.G.successors(u))
            edge_probabilities[(u, v)] = data['occurrence'] / total_transitions_from_u if total_transitions_from_u > 0 else 0
            edge_widths[(u, v)] = 1 + 5 * edge_probabilities[(u, v)]  # Scale edge width based on probability
        
        # Set the edge probabilities and widths in the graph
        nx.set_edge_attributes(self.G, edge_probabilities, 'probability')
        nx.set_edge_attributes(self.G, edge_widths, 'width')

    def build_graph(self):
        """
        This function will return the pyvis object that can be used to plot the created graph.
        """
        # Ensure probabilities and sizes are up to date
        self._update_probabilities_and_sizes()
        
        self.net = Network(height="1500px", width="100%", bgcolor="#222222", font_color="white", directed=True, notebook=True, filter_menu=True, select_menu=True, cdn_resources="in_line")
        
        # Create the Pyvis network
        self.net.from_nx(self.G)
        
        # Iterate over each node in the Pyvis network to set the title with occurrence, mean reward, and probability
        for node in self.net.nodes:
            state = self.G.nodes[node['id']]['state']
            occurrence = self.G.nodes[node['id']]['occurrence']
            probability = round(100 * self.G.nodes[node['id']]['probability'], 1)
            mean_reward = self.G.nodes[node['id']]['mean_reward']
            node['title'] = f"State: {state} \n Occurrence: {occurrence} \n Mean Reward: {mean_reward:.2f} \n Probability: {probability}%"
        
        # Iterate over each edge in the Pyvis network to set the title with occurrence, mean reward, and probability
        for edge in self.net.edges:
            u, v = edge['from'], edge['to']
            occurrence = self.G[u][v]['occurrence']
            probability = round(100 * self.G[u][v]['probability'], 1)
            mean_reward = self.G[u][v]['mean_reward']
            
            action_info = "\nActions:\n"
            for action, data in self.G[u][v]['actions'].items():
                action_info += f"  {action}: Count: {data['count']}, Mean Reward: {data['mean_reward']:.2f}, Probability: {data['probability']:.2f}\n"
            
            edge['title'] = f"Transition from {u} to {v} \n Occurrence: {occurrence} \n Mean Reward: {mean_reward:.2f} \n Probability: {probability}%{action_info}"
        
        # Calculate the graph size
        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        
        # Create a text element for the graph size information
        size_text = f"Number of Nodes: {num_nodes}<br>Number of Edges: {num_edges}"
        
        # Add the size information as an HTML element to the Pyvis network
        self.net.add_node("size_info", label=size_text, shape="text", x='-95%', y=0, physics=False)
        
        self.net.barnes_hut(overlap=1)
        self.net.show_buttons(filter_=['physics'])
        return

    def get_graph(self, mode="all"):
        """
        This function will return the networkX object of the graph in order to perform analysis "Action Steering"
        """
        if mode == "all":
            return self.G, self.net
        
        if mode == "networkX":
            return self.G
        
        if mode == "pyvis":
            return self.net
