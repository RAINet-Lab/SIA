from sia.core.quantile_manager import QuantileManager  # Required for DataFrame operations


class Symbolizer:
    def __init__(
        self, 
        quantile_manager: QuantileManager, 
        kpi_list: list, 
        # kpi_names: dict, 
        kpi_change_threshold_percent: int = 10
    ):
        """
        Initializes the Symbolizer with quantile manager, KPI list, and KPI name mappings.

        :param quantile_manager: An instance of QuantileManager to manage quantiles for KPIs.
        :param kpi_list: A list of KPI names to be symbolized.
        :param kpi_change_threshold_percent: Threshold percentage to determine if a KPI has increased or decreased.
        """
        self.quantile_manager = quantile_manager
        self.kpis_list = kpi_list
        self.kpi_change_threshold_percent = kpi_change_threshold_percent
        
        # Define category names for standard KPIs
        self.standard_category_names = ["VeryLow", "Low", "Medium", "High", "VeryHigh"]
        # Define category names for special KPIs (buffer_kpi and dl_tput)
        self.special_category_names = ["Low", "Medium", "High"]
        # List of KPIs that use the special three-category system
        # self.special_kpis = ["buffer", "dl_tput", "dl_delay"]
        self.special_kpis = []
        
        self.prev_state_dict = None

    def create_symbolic_form(self, curr_step_dict, update_approximators=True):
        """
        Receives the current state of an agent at a timestep and returns the symbolic representation.
        Internally manages the previous state.

        :param curr_step_dict: Current timestep dictionary containing state information.
        :return: A list of dictionaries with symbolic representations for each slice.
        """
        effects_symbolic_representation = []
        prev_step_dict = self.prev_state_dict

        if prev_step_dict is not None:
            # 1. Create symbolic rep for each KPI 
            kpi_symb = self._calculate_kpi_symbolic_state(curr_step_dict, prev_step_dict)
            
            # 2. Create symbolic rep for the decision 
            decision_symb = self._calculate_decision_symbolic_state(
                curr_step_dict, prev_step_dict)
        
            symbolic_representation = {
                "Timestep": curr_step_dict['Timestep'],
                **kpi_symb,
                **decision_symb,
                "reward": curr_step_dict['reward'],
            }
            effects_symbolic_representation.append(symbolic_representation)
        else:

            # Handle the first timestep where there is no previous state
            symbolic_representation = {
                "Timestep": curr_step_dict['Timestep'],
                **self._initialize_kpi_symbols(curr_step_dict),
                **self._initialize_decision_symbols(curr_step_dict),
                "reward": curr_step_dict['reward'],
            }    
            effects_symbolic_representation.append(symbolic_representation)

        if update_approximators:
            # Update the quantile approximators with current KPI data
            self._add_timestep_kpi_data_to_approximator(curr_step_dict)
            
            # Update the internal previous state with the current slice data
            self.prev_state_dict = curr_step_dict.copy()

        return effects_symbolic_representation
    
    def _initialize_kpi_symbols(self, curr_step_dict: dict):
        """
        Initialize KPI symbols for the first timestep where previous data is unavailable.

        :param curr_step_dict: Current step dictionary.
        :return: A dictionary with KPI symbolic representations.
        """
        kpi_symbolic_representation = {}
        for kpi in self.kpis_list:
            curr_value = curr_step_dict[kpi]
            kpi_symbolic_representation[kpi] = self._get_initial_kpi_symb(curr_value, kpi)
        return kpi_symbolic_representation

    def _initialize_decision_symbols(self, curr_step_dict: dict):
        """
        Initialize decision symbols for the first timestep where previous data is unavailable.

        :param curr_step_dict: Current step dictionary.
        :return: A dictionary with decision symbolic representations.
        """
        symb_decision = {}
        sel_brate = {
            "curr": curr_step_dict['sel_brate'],
        }
        
        predicate = 'const'
        symb_decision['sel_brate'] = f"{predicate}(sel_brate, {sel_brate['curr']})"

        return symb_decision

    def _get_initial_kpi_symb(self, curr_value, kpi_name):
        """
        Generate KPI symbolic representation for the first timestep.

        :param curr_value: Current KPI value.
        :param kpi_name: Name of the KPI.
        :return: Symbolic representation string.
        """
        markers = self.quantile_manager.get_markers(kpi_name)
        category = self._get_category(curr_value, markers, kpi_name)
        predicate = 'const'
        return f"{predicate}({kpi_name}, {category})"
    
    def _calculate_decision_symbolic_state(self, curr_step_dict, prev_step_dict):
        """
        Calculate the symbolic state for decisions based on current and previous decision values.

        :param curr_step_dict: Current dictionary.
        :param prev_step_dict: Previous dictionary.
        :return: A dictionary with decision symbolic representations.
        """
        symb_decision = {}
        
        sel_brate = {
            "curr": curr_step_dict['sel_brate'],
            "prev": prev_step_dict['sel_brate'],
        }
        
        change_percentage = self._find_change_percentage(sel_brate['curr'], sel_brate['prev'])
        predicate = self._get_predicate(change_percentage)
        
        if predicate == 'const':
            symb_decision['sel_brate'] = f"{predicate}(sel_brate, {sel_brate['curr']})"
        else:
            symb_decision['sel_brate'] = f"{predicate}(sel_brate, {sel_brate['prev']}, {sel_brate['curr']})"

        return symb_decision
    
    def _calculate_kpi_symbolic_state(self, curr_step_dict: dict, prev_step_dict: dict):
        """
        Calculate the symbolic state for KPIs based on current and previous KPI values.

        :param curr_step_dict: Current slice dictionary.
        :param prev_step_dict: Previous slice dictionary.
        :return: A dictionary with KPI symbolic representations.
        """
        kpi_symbolic_representation = {}

        for kpi in self.kpis_list:
            curr_value = curr_step_dict[kpi]
            prev_value = prev_step_dict[kpi]
            kpi_symbolic_representation[kpi] = self._get_kpi_symb(curr_value, prev_value, kpi)
        
        return kpi_symbolic_representation
    
    def _get_kpi_symb(self, curr_value, prev_value, kpi_name):
        '''
        Calculate the symbolic representation of KPI changes.
        Args:
            curr_value (float): The current value of the KPI.
            prev_value (float): The previous value of the KPI.
            kpi_name (str): The name of the KPI.
        Returns:
            str: A string representing the symbolic representation of the KPI change.
        '''
        change_percentage = self._find_change_percentage(curr_value, prev_value)
        # predicate = self._get_predicate_category_based(curr_value, prev_value, kpi_name)
        predicate = self._get_predicate(change_percentage)
        
        # Get markers and category for current value
        markers = self.quantile_manager.get_markers(kpi_name)
        category = self._get_category(curr_value, markers, kpi_name)
        
        return f'{predicate}({kpi_name}, {category})'
    
    def _get_category(self, value, markers, kpi_name):
        """
        Categorize a value based on percentile markers for standard KPIs
        or based on fixed percentiles (30, 70) for special KPIs (buffer_kpi and dl_tput).

        :param value: The value to categorize.
        :param markers: List of percentile markers.
        :param kpi_name: Name of the KPI.
        :return: The category name as a string.
        """
        if kpi_name in self.special_kpis:
            # Use fixed percentiles for special KPIs
            if len(markers) < 8:  # Need at least up to P70 (index 7)
                return "Unknown"
            
            p30 = markers[3]  # Assuming markers are [P0, P10, P20, P30, P40, P50, P60, P70, P80, P90, P100]
            p70 = markers[7]
            
            if value < p30:
                return self.special_category_names[0]  # Low
            elif value <= p70:
                return self.special_category_names[1]  # Medium
            else:
                return self.special_category_names[2]  # High
        else:
            # Use standard 5 categories for other KPIs
            if len(markers) < 11:  # Need all markers for proper categorization
                return "Unknown"
            
            # Get the relevant percentile markers
            p20 = markers[2]  # P20
            p40 = markers[4]  # P40
            p60 = markers[6]  # P60
            p80 = markers[8]  # P80
            
            if value <= p20:
                return self.standard_category_names[0]  # VeryLow
            elif value <= p40:
                return self.standard_category_names[1]  # Low
            elif value <= p60:
                return self.standard_category_names[2]  # Medium
            elif value <= p80:
                return self.standard_category_names[3]  # High
            else:
                return self.standard_category_names[4]  # VeryHigh

    def _find_change_percentage(self, curr_value, prev_value):
        """
        Calculate the change percentage of the given parameter.

        :param curr_value: Current value.
        :param prev_value: Previous value.
        :return: Change percentage as an integer or 'inf' for infinite change.
        """
        if prev_value == 0:
            if curr_value == 0:
                return 0
            else:
                return 'inf'
        else:
            return int(100 * (curr_value - prev_value) / prev_value)
    
    def _get_predicate(self, change_percentage):
        """
        Return the correct predicate according to the change percentage.

        :param change_percentage: The percentage change.
        :return: A string representing the predicate ('inc', 'dec', 'const').
        """
        if change_percentage == 'inf':
            return "inc"
        elif change_percentage > self.kpi_change_threshold_percent:
            return "inc"
        elif change_percentage < -self.kpi_change_threshold_percent:
            return "dec"
        else:
            return "const"
    
    def _get_predicate_category_based(self, curr_value, prev_value, kpi_name):
        markers = self.quantile_manager.get_markers(kpi_name)
        curr_cat = self._get_category(curr_value, markers, kpi_name)
        prev_cat = self._get_category(prev_value, markers, kpi_name)

        if curr_cat == prev_cat:
            return "const"
        elif curr_cat == "High" and prev_cat != "High":
            return "inc"
        elif curr_cat == "Low" and prev_cat != "Low":
            return "dec"
        elif curr_cat == "Medium" and prev_cat == "Low":
            return "inc"
        elif curr_cat == "Low" and prev_cat == "Medium":
            return "dec"
        elif curr_cat == "Medium" and prev_cat == "High":
            return "dec"
        else:
            return "const"  # Default case
    
    def _add_timestep_kpi_data_to_approximator(self, timestep_dict):
        """Adds KPI data of one timestep to the quantile approximators."""
        for kpi_name in self.kpis_list:
            self.quantile_manager.partial_fit(kpi_name, [timestep_dict[kpi_name]])
