import numpy as np
def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    n = data.shape[0]
    
    # Initialize a random initial eigenvector
    eigenvector = np.random.rand(n)
    
    for _ in range(num_steps):
        # Power iteration step
        eigenvector = np.dot(data, eigenvector)
        
        # Normalize the eigenvector
        eigenvalue = np.linalg.norm(eigenvector)
        eigenvector /= eigenvalue
    
    return eigenvalue, eigenvector