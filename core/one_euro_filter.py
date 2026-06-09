import math
import time

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    def __init__(self, t0=None, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: Decreasing this minimizes jitter but increases lag
        beta: Increasing this reduces lag but increases jitter during fast movements
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = dx0
        self.t_prev = t0 if t0 is not None else time.monotonic()

    def __call__(self, t, x):
        """Compute the filtered signal."""
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        if t_e > 0:
            a_d = smoothing_factor(t_e, self.d_cutoff)
            dx = (x - self.x_prev) / t_e
            dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)
        else:
            dx_hat = self.dx_prev

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

class LandmarkOneEuroFilter:
    """A wrapper to apply the 1Euro filter to a 3D MediaPipe landmark."""
    def __init__(self, min_cutoff=1.0, beta=0.0):
        self.fx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.fy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.fz = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        
    def process(self, t, x, y, z):
        return (
            self.fx(t, x),
            self.fy(t, y),
            self.fz(t, z)
        )
