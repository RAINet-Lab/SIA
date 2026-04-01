import numpy as np

class PSquareQuantileApproximator:
    """
    This class applies the P² algorithm for dynamic quantile approximation.
    Tracks 11 markers for Min, P10, P20, ..., P90, Max.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Initialize or reset the quantile approximator."""
        self.n = list(range(11))  # Marker positions for Min, P10, P20, ..., Max
        # Desired marker positions: equally spaced from 0 to 10 (11 positions)
        self.ns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Desired position increments (10% increments between markers)
        self.dns = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
        self.q = []  # Marker heights

    def fit(self, X):
        """Fit the model to the data."""
        self.reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """Incrementally fit the model to the data."""
        for x in X:
            self._partial_fit_single(x)
        return self

    def _partial_fit_single(self, x):
        """Fit the model to a single data point."""
        if len(self.q) < 11:
            self.q.append(x)
            self.q.sort()
            return self
        
        # Determine marker position for the new data point
        if x <= self.q[0]:
            self.q[0] = x
            k = 0
        elif x >= self.q[-1]:
            self.q[-1] = x
            k = 10
        else:
            k = next(i for i, q in enumerate(self.q) if x < q)

        # Increment positions and desired positions
        for i in range(k, 11):
            self.n[i] += 1
        
        self.ns = [ns + dns for ns, dns in zip(self.ns, self.dns)]    

        # Adjust marker heights
        for i in range(1, 10):
            d = self.ns[i] - self.n[i]
            if (d >= 1 and self.n[i+1] - self.n[i] > 1) or (d <= -1 and self.n[i-1] - self.n[i] < -1):
                d_sign = np.sign(d)
                q_para = self._parabolic(i, d_sign)
                self.q[i] = q_para if self.q[i-1] < q_para < self.q[i+1] else self._linear(i, d_sign)
                self.n[i] += d_sign

    def _parabolic(self, i, d):
        """Calculate parabolic prediction for marker height adjustment."""
        i = int(i)
        d = int(d)
        return self.q[i] + (d * (self.n[i] - self.n[i-1] + d) * (self.q[i+1] - self.q[i]) / (self.n[i+1] - self.n[i])
                            + d * (self.n[i+1] - self.n[i] - d) * (self.q[i] - self.q[i-1]) / (self.n[i] - self.n[i-1])) / (self.n[i+1] - self.n[i-1])

    def _linear(self, i, d):
        """Calculate linear prediction for marker height adjustment."""
        i = int(i)
        d = int(d)
        return self.q[i] + d * (self.q[i+d] - self.q[i]) / (self.n[i+d] - self.n[i])

    def score(self):
        """Return the median (P50) or a default if no markers exist."""
        if not self.q:
            return None  # No data available

        if len(self.q) < 11:
            return np.percentile(self.q, 50)  # Use NumPy's percentile if fewer than 11 data points

        return self.q[5]  # P50 (Median) is at position 5 in the list of markers

    def get_markers(self):
        """Return the current marker heights."""
        return self.q
