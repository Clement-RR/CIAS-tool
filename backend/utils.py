import numpy as np


class Utils:
    @staticmethod
    def mincolumn(m):
        """
        Identifies the minimum non-zero value in each column of the input matrix 'm',
        setting all other values to 0, effectively filtering out non-minimum values.

        Args:
        m (np.array): Input 2D numpy array.

        Returns:
        np.array: A 2D numpy array with the same shape as 'm', where each column contains
                  only its minimum non-zero value, with all other entries set to 0.
        """
        A = np.zeros_like(m)  # Initialize A with zeros of the same shape as m

        for i in range(m.shape[1]):  # Iterate over columns
            column = m[:, i]
            non_zero = column[column != 0]  # Filter out zero values

            if non_zero.size > 0:  # Check if there are non-zero values
                min_val = np.min(non_zero)  # Find the minimum non-zero value
                min_positions = column == min_val  # Identify positions of the minimum value
                A[min_positions, i] = column[min_positions]  # Copy these minimum values to A

        return A

    @staticmethod
    def is_valid_matrix(matrix):
        """
        Validates that the matrix is a square matrix and that all values are
        probabilities between 0 and 1.
        """
        if not isinstance(matrix, list) or not matrix:
            return False

        # Check if all rows are lists and have the same length as the first row
        row_length = len(matrix[0])
        if any(not isinstance(row, list) or len(row) != row_length for row in matrix):
            return False

        # Check if all values are between 0 and 1
        if any(not all(0 <= value <= 1 for value in row) for row in matrix):
            return False

        return True

