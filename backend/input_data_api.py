import numpy as np


class InputData:
    def __init__(self):
        self.P = np.array([])  # Transition matrix
        self.A = np.array([])  # Left interval (best case)
        self.B = np.array([])  # Right interval (worst case)
        self.M = np.array([])  # PERT-mean (most likely)
        self.tA = np.array([])
        self.tB = np.array([])
        self.tM = np.array([])

    def set_matrices(self, P, A, B, M, tA, tB, tM):
        self.P = np.array(P)
        self.A = np.array(A)
        self.B = np.array(B)
        self.M = np.array(M)
        self.tA = np.array(tA)
        self.tB = np.array(tB)
        self.tM = np.array(tM)


# Global instance to be used across the application
input_data_instance = InputData()

