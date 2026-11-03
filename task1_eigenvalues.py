import numpy as np


def calculate_eigenvalues_and_eigenvectors(matrix):
    if matrix.ndim != 2:
        raise ValueError("The matrix must be two-dimensional.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The matrix must be square.")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    return eigenvalues, eigenvectors


def verify_eigenvectors(matrix, eigenvalues, eigenvectors):
    for index in range(len(eigenvalues)):
        eigenvalue = eigenvalues[index]
        eigenvector = eigenvectors[:, index]

        left_side = np.dot(matrix, eigenvector)
        right_side = eigenvalue * eigenvector

        print("Eigenvalue:", eigenvalue)
        print("Eigenvector:", eigenvector)

        if np.allclose(left_side, right_side):
            print("Verification: A * v = lambda * v is correct.")
        else:
            print("Verification: A * v = lambda * v is incorrect.")

        print()


def main():
    matrix = np.array([
        [4, 1],
        [1, 3]
    ])

    print("Original matrix:")
    print(matrix)
    print()

    eigenvalues, eigenvectors = calculate_eigenvalues_and_eigenvectors(matrix)

    print("Eigenvalues:")
    print(eigenvalues)
    print()

    print("Eigenvectors:")
    print(eigenvectors)
    print()

    verify_eigenvectors(matrix, eigenvalues, eigenvectors)


if __name__ == "__main__":
    main()
