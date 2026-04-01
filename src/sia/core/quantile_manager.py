import pandas as pd
from sia.core.p_square_approximator import PSquareQuantileApproximator

class QuantileManager:
    """
    Manages multiple PSquareQuantileApproximators for different KPIs.
    """
    def __init__(self, kpi_list):        
        self.quantile_approximators = {kpi: PSquareQuantileApproximator() for kpi in kpi_list}
    
    def fit(self):
        for approximator in self.quantile_approximators.values():
            approximator.fit([])
    
    def partial_fit(self, kpi_name, value):
        """Update the quantile approximation for a specific KPI."""
        if kpi_name in self.quantile_approximators:
            self.quantile_approximators[kpi_name].partial_fit(value)
        
    def get_markers(self, kpi_name):
        """Get the markers (percentile estimates) for a specific KPI."""
        if kpi_name in self.quantile_approximators:
            return self.quantile_approximators[kpi_name].get_markers()
        else:
            return []
        
    def reset(self):
        """Reset all approximators."""
        for kpi in self.quantile_approximators:
            self.quantile_approximators[kpi].reset()
    
    def export_markers(self):
        """
        Export all markers and their associated metadata to a DataFrame.
        Returns a DataFrame containing markers and internal state for all KPIs.
        """
        export_data = []
        for kpi in self.quantile_approximators:
            approx = self.quantile_approximators[kpi]
            if len(approx.q) == 11:  # Only export if we have all markers
                export_data.append({
                    'kpi': kpi,
                    'markers': approx.q,
                    'n': approx.n,
                    'ns': approx.ns,
                    'dns': approx.dns
                })
        
        return pd.DataFrame(export_data)
    
    def load_markers(self, markers_df):
        """
        Load markers and their associated metadata from a DataFrame.
        
        Parameters:
        markers_df: DataFrame containing the markers and internal state for all KPIs
        """
        for _, row in markers_df.iterrows():
            kpi = row['kpi']
            if kpi in self.quantile_approximators:
                approx = self.quantile_approximators[kpi]
                approx.q = list(row['markers'])
                approx.n = list(row['n'])
                approx.ns = list(row['ns'])
                approx.dns = list(row['dns'])

    def represent_markers(self):
        """
        Return a DataFrame with the marker values for each KPI.
        """
        markers_data = []
        for kpi in self.quantile_approximators:
            markers = self.get_markers(kpi)
            if len(markers) == 11:  # Ensure there are 11 markers
                markers_data.append({
                    "kpi": kpi,
                    "min": markers[0],
                    "p10": markers[1],
                    "p20": markers[2],
                    "p30": markers[3],
                    "p40": markers[4],
                    "p50": markers[5],
                    "p60": markers[6],
                    "p70": markers[7],
                    "p80": markers[8],
                    "p90": markers[9],
                    "max": markers[10],
                })
        
        return pd.DataFrame(markers_data)
