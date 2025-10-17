import numpy as np

def expectation(states: np.ndarray, observable: np.ndarray) -> np.ndarray:
    """
    Returns the expectation values of a given observable for a list of states.

    Parameters
    ----------
    states : np.ndarray
        Array of shape (N, d) where N is the number of states and d is the dimension of each state.
    observable : np.ndarray
        Operator matrix of shape (d, d).

    Returns
    -------
    np.ndarray
        Array of expectation values of the observable for each state.
    """
    expectations = np.array([np.vdot(psi, observable @ psi) for psi in states])
    return expectations
