
import numpy as np


def encrypt_message(message, key_matrix):
    message_vector = np.array(
        [ord(character) for character in message]
    )

    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)

    diagonalized_key_matrix = np.dot(
        np.dot(
            eigenvectors,
            np.diag(eigenvalues)
        ),
        np.linalg.inv(eigenvectors)
    )

    encrypted_vector = np.dot(
        diagonalized_key_matrix,
        message_vector
    )

    return encrypted_vector


def decrypt_message_(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)

    if np.any(np.abs(eigenvalues) < 1e-10):
        raise ValueError("The key matrix must have non-zero eigenvalues.")

    inverse_diagonal_matrix = np.diag(
        1 / eigenvalues
    )

    inverse_diagonalized_matrix = np.dot(
        np.dot(
            eigenvectors,
            inverse_diagonal_matrix
        ),
        np.linalg.inv(eigenvectors)
    )

    decrypted_vector = np.dot(
        inverse_diagonalized_matrix,
        encrypted_vector
    )

    decrypted_vector = np.real(decrypted_vector)

    decrypted_vector = np.rint(
        decrypted_vector
    ).astype(int)

    decrypted_message = ""

    for character_code in decrypted_vector:
        decrypted_message = decrypted_message + chr(character_code)

    return decrypted_message


def generate_key_matrix(message_length):
    random_generator = np.random.default_rng(42)

    while True:
        key_matrix = random_generator.integers(
            0,
            256,
            size=(message_length, message_length)
        )

        eigenvalues, eigenvectors = np.linalg.eig(key_matrix)

        if np.any(np.abs(eigenvalues) < 1e-10):
            continue

        if np.linalg.cond(eigenvectors) > 1e10:
            continue

        return key_matrix


def main():
    original_message = "Hello, World!"

    key_matrix = generate_key_matrix(
        len(original_message)
    )

    print("Original Message:")
    print(original_message)
    print()

    print("Key Matrix:")
    print(key_matrix)
    print()

    encrypted_vector = encrypt_message(
        original_message,
        key_matrix
    )

    print("Encrypted Message:")
    print(encrypted_vector)
    print()

    decrypted_message = decrypt_message_(
        encrypted_vector,
        key_matrix
    )

    print("Decrypted Message:")
    print(decrypted_message)
    print()

    if original_message == decrypted_message:
        print("Verification: Encryption and decryption are successful.")
    else:
        print("Verification: Encryption and decryption failed.")


if __name__ == "__main__":
    main()
