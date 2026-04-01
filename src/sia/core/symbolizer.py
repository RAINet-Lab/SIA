import pandas as pd
import numpy as np
from sia.core.quantile_manager import QuantileManager

class Symbolizer:
    def __init__(self, quantile_manager: QuantileManager, kpis: dict, decisions: dict, kpi_change_threshold_percent: int = 10):
        """
        Initializes the Symbolizer with quantile manager, KPIs, and decisions.

        :param quantile_manager: An instance of QuantileManager to manage quantiles for KPIs.
        :param kpis: A dictionary mapping full KPI names to their short forms.
        :param decisions: A dictionary mapping full decision variable names to their short forms.
        :param kpi_change_threshold_percent: Threshold percentage to determine if a KPI has increased or decreased.
        """
        self.quantile_manager = quantile_manager
        self.kpis = kpis  # dict mapping full KPI names to short names
        self.decisions = decisions  # dict mapping decision variable names to short names
        self.kpi_change_threshold_percent = kpi_change_threshold_percent

        # Define category names - descriptive terms
        self.category_names = ["VeryLow", "Low", "Medium", "High", "VeryHigh"]

        # Initialize internal storage for previous state per slice_id
        self.prev_state_dict = {}  # Stores previous state DataFrame per slice_id

        # Validate that the quantile_manager has all the KPIs and decisions
        all_keys = list(self.kpis.keys()) + list(self.decisions.keys())
        missing_kpis = [kpi for kpi in all_keys if kpi not in self.quantile_manager.quantile_approximators]
        if missing_kpis:
            raise ValueError(
                f"The following KPIs or decisions are missing in QuantileManager: {missing_kpis}"
            )

    def create_symbolic_form(self, curr_step_df: pd.DataFrame):
        """
        Receives the current state of an agent at a timestep and returns the symbolic representation.
        Internally manages the previous state.

        :param curr_step_df: Current timestep DataFrame containing state information.
        :return: A list of dictionaries with symbolic representations for each slice.
        """
        effects_symbolic_representation = []
        slices = curr_step_df['slice_id'].unique()

        for slice_id in slices:
            # Extract current slice data
            curr_slice_df = curr_step_df[curr_step_df['slice_id'] == slice_id]

            # Retrieve previous slice data if it exists
            prev_slice_df = self.prev_state_dict.get(slice_id)

            if prev_slice_df is not None:
                # 1. Create symbolic representation for each KPI of the slice
                slice_kpi_symb = self._calculate_kpi_symbolic_state(curr_slice_df, prev_slice_df)
                
                # 2. Create symbolic representation for the decision of each slice
                slice_decision_symb = self._calculate_decision_symbolic_state(curr_slice_df, prev_slice_df)

                # Combine symbolic representations
                symbolic_representation = {
                    "timestamp": curr_slice_df['timestamp'].iloc[0],
                    "slice_id": slice_id,
                    **slice_kpi_symb,
                    **slice_decision_symb
                }
                effects_symbolic_representation.append(symbolic_representation)
            else:
                # Handle the first timestep where there is no previous state
                symbolic_representation = {
                    "timestamp": curr_slice_df['timestamp'].iloc[0],
                    "slice_id": slice_id,
                    **self._initialize_kpi_symbols(curr_slice_df),
                    **self._initialize_decision_symbols(curr_slice_df)
                }
                effects_symbolic_representation.append(symbolic_representation)

            # Update the quantile approximators with current KPI data
            self._add_timestep_kpi_data_to_approximator(curr_slice_df)

            # Update the internal previous state with the current slice data
            self.prev_state_dict[slice_id] = curr_slice_df.copy()

        return effects_symbolic_representation

    def _initialize_kpi_symbols(self, curr_slice_df: pd.DataFrame):
        """
        Initialize KPI symbols for the first timestep where previous data is unavailable.

        :param curr_slice_df: Current slice DataFrame.
        :return: A dictionary with KPI symbolic representations.
        """
        kpi_symbolic_representation = {}
        for kpi in self.kpis:
            curr_value = curr_slice_df[kpi].iloc[0]
            short_kpi = self.kpis.get(kpi, kpi)  # Use short name if available
            kpi_symbolic_representation[kpi] = self._get_initial_kpi_symb(curr_value, short_kpi, kpi)
        return kpi_symbolic_representation

    def _initialize_decision_symbols(self, curr_slice_df: pd.DataFrame):
        """
        Initialize decision symbols for the first timestep where previous data is unavailable.

        :param curr_slice_df: Current slice DataFrame.
        :return: A dictionary with decision symbolic representations.
        """
        symb_decision = {}

        # Process each decision variable
        for decision in self.decisions:
            curr_value = curr_slice_df[decision].iloc[0]
            short_name = self.decisions[decision]

            # Handle 'scheduling_policy' separately if needed
            if decision == 'scheduling_policy':
                symb_decision[decision] = "const(sched)"
            else:
                markers = self.quantile_manager.get_markers(decision)
                curr_cat = self._get_category(curr_value, markers)
                symb_decision[decision] = f"const({short_name}, {curr_cat})"

        return symb_decision

    def _get_initial_kpi_symb(self, curr_value, short_kpi_name, kpi_name):
        """
        Generate KPI symbolic representation for the first timestep.

        :param curr_value: Current KPI value.
        :param short_kpi_name: Short name of the KPI.
        :param kpi_name: Full name of the KPI.
        :return: Symbolic representation string.
        """
        markers = self.quantile_manager.get_markers(kpi_name)
        category = self._get_category(curr_value, markers)
        return f"const({short_kpi_name}, {category})"

    def _calculate_decision_symbolic_state(self, curr_step_df: pd.DataFrame, prev_step_df: pd.DataFrame):
        """
        Calculate the symbolic state for decisions based on current and previous decision values.

        :param curr_step_df: Current slice DataFrame.
        :param prev_step_df: Previous slice DataFrame.
        :return: A dictionary with decision symbolic representations.
        """
        symb_decision = {}

        for decision in self.decisions:
            curr_value = curr_step_df[decision].iloc[0]
            prev_value = prev_step_df[decision].iloc[0]
            short_name = self.decisions[decision]

            if decision == 'scheduling_policy':
                # Handle scheduling policy separately
                scheduling_policy_string_helper = {0: "RR", 1: "WF", 2: "PF"}
                if curr_value == prev_value:
                    symb_decision[decision] = "const(sched)"
                else:
                    new_policy = scheduling_policy_string_helper.get(curr_value, "Unknown")
                    symb_decision[decision] = f"to{new_policy}(sched)"
            else:
                markers = self.quantile_manager.get_markers(decision)
                curr_cat = self._get_category(curr_value, markers)
                prev_cat = self._get_category(prev_value, markers)
                change_percentage = self._find_change_percentage(curr_value, prev_value)
                predicate = self._get_predicate(change_percentage)

                if curr_cat == prev_cat:
                    symb_decision[decision] = f"const({short_name}, {curr_cat})"
                else:
                    symb_decision[decision] = f"{predicate}({short_name}, {prev_cat}, {curr_cat})"

        return symb_decision

    def _calculate_kpi_symbolic_state(self, curr_state_df: pd.DataFrame, prev_state_df: pd.DataFrame):
        """
        Calculate the symbolic state for KPIs based on current and previous KPI values.

        :param curr_state_df: Current slice DataFrame.
        :param prev_state_df: Previous slice DataFrame.
        :return: A dictionary with KPI symbolic representations.
        """
        kpi_symbolic_representation = {}
        for kpi in self.kpis:
            curr_value = curr_state_df[kpi].iloc[0]
            prev_value = prev_state_df[kpi].iloc[0]
            short_kpi = self.kpis.get(kpi, kpi)  # Use short name if available
            kpi_symbolic_representation[kpi] = self._get_kpi_symb(curr_value, prev_value, short_kpi, kpi)
        
        return kpi_symbolic_representation

    def _get_kpi_symb(self, curr_value, prev_value, short_kpi_column, kpi_name):
        """
        Calculate the symbolic representation of KPI changes.

        :param curr_value: Current KPI value.
        :param prev_value: Previous KPI value.
        :param short_kpi_column: Short name of the KPI.
        :param kpi_name: Full name of the KPI.
        :return: Symbolic representation string.
        """
        change_percentage = self._find_change_percentage(curr_value, prev_value)
        predicate = self._get_predicate(change_percentage)

        # Get markers and category for current value
        markers = self.quantile_manager.get_markers(kpi_name)
        category = self._get_category(curr_value, markers)

        return f'{predicate}({short_kpi_column}, {category})'

    def _get_category(self, value, markers, category_names=None):
        """
        Categorize a value based on percentile markers (P20, P40, P60, P80).

        :param value: The value to categorize.
        :param markers: List of percentile markers.
        :param category_names: Optional list of category names.
        :return: The category name as a string.
        """
        if category_names is None:
            category_names = self.category_names

        if len(markers) < 11:  # Need all markers for proper categorization
            return "Unknown"

        # Get the relevant percentile markers
        p20 = markers[2]  # P20
        p40 = markers[4]  # P40
        p60 = markers[6]  # P60
        p80 = markers[8]  # P80

        if value <= p20:
            return category_names[0]
        elif value <= p40:
            return category_names[1]
        elif value <= p60:
            return category_names[2]
        elif value <= p80:
            return category_names[3]
        else:
            return category_names[4]

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

    def _add_timestep_kpi_data_to_approximator(self, timestep_df: pd.DataFrame):
        """
        Add KPI data of one timestep to the quantile approximators.

        :param timestep_df: DataFrame containing KPI data for the current timestep.
        """
        for kpi_name in self.kpis:
            self.quantile_manager.partial_fit(kpi_name, [timestep_df[kpi_name].iloc[0]])

        for decision_name in self.decisions:
            if decision_name != 'scheduling_policy':  # Exclude 'scheduling_policy' if needed
                self.quantile_manager.partial_fit(decision_name, [timestep_df[decision_name].iloc[0]])