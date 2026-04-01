from pyvis.network import Network
import networkx as nx
from typing import Dict, Any, Optional

class DecisionGraph:
    def __init__(self) -> None:
        self.G = nx.DiGraph()
        self.net = None
        self.previous_state = None
        return

    def _extract_state(self, symbolic_form_dict: Dict[str, Any]):
        """
        This function extracts the current state and state_id from the symbolic_form_df.
        """
        current_state = (
            symbolic_form_dict.get('buffer'),    # Buffer category (e.g., 'Medium')
            symbolic_form_dict.get('dl_tput'),   # Download throughput category (e.g., 'Low')
            symbolic_form_dict.get('dl_delay'),   # Delay category (e.g., 'Medium')
            symbolic_form_dict.get('bwidth'),   # Upload throughput category (e.g., 'Low')
        )

        # Create a unique ID for the state by joining categories with underscores
        state_id = '_'.join(map(str, current_state))  # e.g., 'Medium_Low_Medium'

        return current_state, state_id

    def update_graph(self, symbolic_form: Dict[str, Any]):
        """
        Receives a new state and updates the graph with the state and transitions.
        
        :param symbolic_form: A dictionary containing the symbolic representation of the current state.
        """
        current_state, state_id = self._extract_state(symbolic_form)
        current_reward = symbolic_form.get('reward', 0)
        
        # Create the action
        action = symbolic_form.get('sel_brate')
        
        
        # Add the current state as a new node if it doesn't exist
        if state_id not in self.G.nodes:
            self.G.add_node(state_id, state=current_state, occurrence=0, total_reward=0.0, mean_reward=0.0)
        
        # Update the occurrence count and total reward for the current state node
        self.G.nodes[state_id]['occurrence'] += 1
        self.G.nodes[state_id]['total_reward'] += current_reward
        self.G.nodes[state_id]['mean_reward'] = self.G.nodes[state_id]['total_reward'] / self.G.nodes[state_id]['occurrence']
        
        # Update the transition from the previous state to the current state
        if self.previous_state is not None:
            if self.G.has_edge(self.previous_state, state_id):
                edge_data = self.G[self.previous_state][state_id]
                edge_data['occurrence'] += 1
                edge_data['total_reward'] += current_reward
                
                if 'actions' not in edge_data:
                    edge_data['actions'] = {}
                    
                if action not in edge_data['actions']:
                    edge_data['actions'][action] = {'count': 0, 'total_reward': 0.0}
                
                edge_data['actions'][action]['count'] += 1
                edge_data['actions'][action]['total_reward'] += current_reward
            else:
                self.G.add_edge(
                    self.previous_state,
                    state_id,
                    occurrence=1,
                    total_reward=current_reward,
                    actions={action: {'count': 1, 'total_reward': current_reward}}
                )
            
            # Update mean rewards for the edge
            edge_data = self.G[self.previous_state][state_id]
            edge_data['mean_reward'] = edge_data['total_reward'] / edge_data['occurrence']
            
            # Update mean rewards and probabilities for actions
            if 'actions' in edge_data:
                for act, data in edge_data['actions'].items():
                    if data['count'] > 0:
                        data['mean_reward'] = data['total_reward'] / data['count']
                        data['probability'] = data['count'] / edge_data['occurrence']
                    else:
                        data['mean_reward'] = 0.0
                        data['probability'] = 0.0
                        
        self.previous_state = state_id
        
        # Update probabilities and sizes for nodes and edges
        self._update_probabilities_and_sizes()
        
        return

    def _get_current_bitrate_from_action(self, symb_action: str) -> Optional[float]:
        """
        Extracts the current bitrate from the symbolic action string.

        :param symb_action: A symbolic action string (e.g., 'const(sel_brate, 750.0)').
        :return: The current bitrate as a float, or None if parsing fails.
        """
        try:
            predicate, rest = symb_action.split('(', 1)
        except ValueError:
            return None  # Input format is incorrect

        # Remove the closing ')' and split the arguments
        rest = rest.rstrip(')')
        args = [arg.strip() for arg in rest.split(',')]

        # Extract numeric values from the arguments
        numbers = []
        for arg in args:
            try:
                numbers.append(float(arg))
            except ValueError:
                continue  # Skip non-numeric arguments like 'sel_brate'

        # Return the appropriate bitrate based on the predicate
        if predicate == 'const' and len(numbers) >= 1:
            return numbers[0]
        elif predicate in ('inc', 'dec') and len(numbers) >= 2:
            return numbers[0]  # Return the first number
        else:
            return None  # Invalid input format

    def _get_next_bitrate_from_action(self, symb_action: str) -> Optional[float]:
        """
        Extracts the next bitrate from the symbolic action string.

        :param symb_action: A symbolic action string (e.g., 'const(sel_brate, 750.0)').
        :return: The next bitrate as a float, or None if parsing fails.
        """
        try:
            predicate, rest = symb_action.split('(', 1)
        except ValueError:
            return None  # Input format is incorrect

        # Remove the closing ')' and split the arguments
        rest = rest.rstrip(')')
        args = [arg.strip() for arg in rest.split(',')]

        # Extract numeric values from the arguments
        numbers = []
        for arg in args:
            try:
                numbers.append(float(arg))
            except ValueError:
                continue  # Skip non-numeric arguments like 'sel_brate'

        # Return the appropriate bitrate based on the predicate
        if predicate == 'const' and len(numbers) >= 1:
            return numbers[0]
        elif predicate in ('inc', 'dec') and len(numbers) >= 2:
            return numbers[-1]  # Return the last number
        else:
            return None  # Invalid input format

    def _check_action_compatibility(self, current_bitrate: Optional[float], action: str) -> bool:
        """
        Checks if the action is compatible with the current bitrate.

        :param current_bitrate: The current bitrate.
        :param action: The action string.
        :return: True if compatible, False otherwise.
        """
        action_bitrate = self._get_current_bitrate_from_action(action)
        return action_bitrate == current_bitrate

    def get_recommendation(self, symbolic_form: Dict[str, Any]) -> Optional[float]:
        """
        Returns a recommended bitrate based on the current state and existing graph data.

        :param symbolic_form: A dictionary containing the symbolic representation of the current state.
        :return: The recommended bitrate as a float, or None if no recommendation is made.
        """
        print(f"Agent wants to make this decision: {symbolic_form.get('sel_brate', 'Unknown')} \n----")
        current_bitrate = self._get_current_bitrate_from_action(symbolic_form.get('sel_brate', ''))

        current_state, state_id = self._extract_state(symbolic_form)

        if state_id in self.G.nodes:
            outgoing_edges = self.G.out_edges(state_id, data=True)
            current_action = symbolic_form.get('sel_brate', 'Unknown')

            action_exists = False
            current_action_reward = None
            best_alternative_action = None
            best_alternative_reward = float('-inf')

            for _, _, edge_data in outgoing_edges:
                edge_probability = edge_data.get('probability', 0.0)
                if 'actions' in edge_data:
                    for action, action_data in edge_data['actions'].items():
                        print(f"action is \n{action}")
                        if action == current_action:
                            action_exists = True
                            current_action_reward = action_data.get('mean_reward', 0.0)
                        elif (action_data.get('mean_reward', 0.0) > best_alternative_reward and self._check_action_compatibility(current_bitrate, action)):
                            print("action is compatible with the current bitrate")
                            print(f"action: {action}")
                            best_alternative_action = action
                            best_alternative_reward = action_data.get('mean_reward', 0.0)

            if action_exists and best_alternative_reward > current_action_reward:
                # Update the recommendation
                recommended_bitrate = self._get_next_bitrate_from_action(best_alternative_action)
                print(f"Recommendation: Change the action from {current_action} ({current_action_reward}) to {best_alternative_action} ({best_alternative_reward})")
                print(f"The bitrate from agent's action {self._get_next_bitrate_from_action(current_action)} and the suggested bitrate is {recommended_bitrate}")
                return recommended_bitrate

        return None
    
    def _update_probabilities_and_sizes(self):
        """
        Updates the probabilities and sizes of nodes and edges in the graph.
        """
        # Calculate the total occurrence count for nodes
        total_node_occurrence = sum(nx.get_node_attributes(self.G, 'occurrence').values())

        if total_node_occurrence == 0:
            return  # Avoid division by zero

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
            if total_transitions_from_u > 0:
                prob = data['occurrence'] / total_transitions_from_u
            else:
                prob = 0
            edge_probabilities[(u, v)] = prob
            edge_widths[(u, v)] = 1 + 5 * prob  # Scale edge width based on probability

        # Set the edge probabilities and widths in the graph
        nx.set_edge_attributes(self.G, edge_probabilities, 'probability')
        nx.set_edge_attributes(self.G, edge_widths, 'width')

    def build_graph(self):
        """
        Builds and visualizes the graph using Pyvis.
        """
        # Ensure probabilities and sizes are up to date
        self._update_probabilities_and_sizes()

        self.net = Network(
            height="1500px", 
            width="100%", 
            bgcolor="#222222", 
            font_color="white", 
            directed=True, 
            notebook=True, 
            filter_menu=True, 
            select_menu=True, 
            cdn_resources="in_line"
        )

        # Create the Pyvis network
        self.net.from_nx(self.G)

        # Iterate over each node in the Pyvis network to set the title with occurrence, mean reward, and probability
        for node in self.net.nodes:
            node_id = node['id']
            state = self.G.nodes[node_id].get('state', 'Unknown')
            occurrence = self.G.nodes[node_id].get('occurrence', 0)
            probability = round(100 * self.G.nodes[node_id].get('probability', 0), 1)
            mean_reward = self.G.nodes[node_id].get('mean_reward', 0.0)
            node['title'] = f"State: {state} \n Occurrence: {occurrence} \n Mean Reward: {mean_reward:.2f} \n Probability: {probability}%"

        # Iterate over each edge in the Pyvis network to set the title with occurrence, mean reward, and probability
        for edge in self.net.edges:
            u, v = edge['from'], edge['to']
            occurrence = self.G[u][v].get('occurrence', 0)
            probability = round(100 * self.G[u][v].get('probability', 0), 1)
            mean_reward = self.G[u][v].get('mean_reward', 0.0)

            action_info = "\nActions:\n"
            if 'actions' in self.G[u][v]:
                for action, data in self.G[u][v]['actions'].items():
                    count = data.get('count', 0)
                    mean_r = data.get('mean_reward', 0.0)
                    prob = data.get('probability', 0.0)
                    action_info += f"  {action}: Count: {count}, Mean Reward: {mean_r:.2f}, Probability: {prob:.2f}\n"

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
        Returns the graph in the specified format.

        :param mode: The format to return ('all', 'networkX', 'pyvis').
        :return: The graph in the requested format.
        """
        if mode == "all":
            return self.G, self.net

        if mode == "networkX":
            return self.G

        if mode == "pyvis":
            return self.net
