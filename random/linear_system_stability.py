import numpy as np

def find_min_alpha():
    """
    Find the smallest positive alpha that makes the system stable.
    Returns the value of alpha.
    """
    def is_stable(alpha):
        # Create the system matrix with current alpha
        A = np.array([
            [10 - alpha, -3],
            [-3, -100]
        ])
        # Get eigenvalues
        eigenvals = np.linalg.eigvals(A)
        # System is stable if all eigenvalues have negative real parts
        return np.all(np.real(eigenvals) < 0)

    # Binary search for the smallest alpha
    left = 0
    right = 100  # Start with a reasonable upper bound
    tolerance = 1e-6  # Precision for binary search

    while right - left > tolerance:
        mid = (left + right) / 2
        if is_stable(mid):
            right = mid
        else:
            left = mid

    # Return the result rounded to 6 decimal places
    return round(right, 6)

if __name__ == "__main__":
    result = find_min_alpha()
    print(f"Minimum stabilizing alpha: {result}") 