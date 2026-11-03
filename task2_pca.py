
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path


def calculate_pca(image_matrix):
    mean_values = np.mean(image_matrix, axis=0)

    centered_matrix = image_matrix - mean_values

    left_singular_vectors, singular_values, right_singular_vectors = np.linalg.svd(
        centered_matrix,
        full_matrices=False
    )

    explained_variance = singular_values ** 2

    explained_variance_ratio = (
        explained_variance / np.sum(explained_variance)
    )

    cumulative_variance = np.cumsum(explained_variance_ratio)

    return (
        mean_values,
        left_singular_vectors,
        singular_values,
        right_singular_vectors,
        cumulative_variance
    )


def reconstruct_image(
    mean_values,
    left_singular_vectors,
    singular_values,
    right_singular_vectors,
    number_of_components
):
    if number_of_components < 1:
        raise ValueError("The number of components must be positive.")

    if number_of_components > len(singular_values):
        raise ValueError("The number of components is too large.")

    selected_left_vectors = (
        left_singular_vectors[:, :number_of_components]
    )

    selected_singular_values = (
        singular_values[:number_of_components]
    )

    selected_right_vectors = (
        right_singular_vectors[:number_of_components, :]
    )

    reconstructed_image = np.dot(
        selected_left_vectors,
        np.dot(
            np.diag(selected_singular_values),
            selected_right_vectors
        )
    )

    reconstructed_image = reconstructed_image + mean_values

    return reconstructed_image


def main():
    image_path = Path(__file__).parent / "image.jpg"

    with Image.open(image_path) as opened_image:
        color_image = opened_image.convert("RGB")
        grayscale_image = opened_image.convert("L")

    color_image_matrix = np.asarray(color_image)
    grayscale_image_matrix = np.asarray(
        grayscale_image,
        dtype=float
    )

    print("ORIGINAL COLOR IMAGE")

    print("Image dimensions:", color_image_matrix.shape)
    print("Image height:", color_image_matrix.shape[0])
    print("Image width:", color_image_matrix.shape[1])
    print("Number of color channels:", color_image_matrix.shape[2])
    print()

    print("BLACK AND WHITE IMAGE")

    print("Image dimensions:", grayscale_image_matrix.shape)
    print("Image height:", grayscale_image_matrix.shape[0])
    print("Image width:", grayscale_image_matrix.shape[1])
    print("Number of color channels: 1")
    print()

    (
        mean_values,
        left_singular_vectors,
        singular_values,
        right_singular_vectors,
        cumulative_variance
    ) = calculate_pca(grayscale_image_matrix)

    number_of_components_95 = (
        np.searchsorted(cumulative_variance, 0.95) + 1
    )

    print("PCA RESULTS")

    print(
        "Number of components needed for 95% variance:",
        number_of_components_95
    )

    print(
        "Cumulative variance:",
        cumulative_variance[number_of_components_95 - 1]
    )

    print()

    reconstructed_image_95 = reconstruct_image(
        mean_values,
        left_singular_vectors,
        singular_values,
        right_singular_vectors,
        number_of_components_95
    )

    reconstructed_image_95 = np.clip(
        reconstructed_image_95,
        0,
        255
    )

    print("Reconstructed image dimensions:")
    print(reconstructed_image_95.shape)
    print()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(color_image)
    plt.title("Original Color Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(grayscale_image_matrix, cmap="gray")
    plt.title("Black and White Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))

    number_of_components = np.arange(
        1,
        len(cumulative_variance) + 1
    )

    plt.plot(
        number_of_components,
        cumulative_variance,
        color="blue"
    )

    plt.axhline(
        y=0.95,
        color="red",
        linestyle="--",
        label="95% Variance"
    )

    plt.axvline(
        x=number_of_components_95,
        color="green",
        linestyle="--",
        label="Required Components"
    )

    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.title("Cumulative Variance from PCA")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))

    plt.imshow(
        reconstructed_image_95,
        cmap="gray"
    )

    plt.title(
        "Reconstructed Image with "
        + str(number_of_components_95)
        + " Components (95% Variance)"
    )

    plt.axis("off")

    plt.tight_layout()
    plt.show()

    component_numbers = [10, 25, 50, 100]

    plt.figure(figsize=(12, 9))

    for index in range(len(component_numbers)):
        current_number_of_components = component_numbers[index]

        reconstructed_image = reconstruct_image(
            mean_values,
            left_singular_vectors,
            singular_values,
            right_singular_vectors,
            current_number_of_components
        )

        reconstructed_image = np.clip(
            reconstructed_image,
            0,
            255
        )

        current_variance = cumulative_variance[
            current_number_of_components - 1
        ]

        plt.subplot(2, 2, index + 1)

        plt.imshow(
            reconstructed_image,
            cmap="gray"
        )

        plt.title(
            str(current_number_of_components)
            + " Components, Variance: "
            + str(round(current_variance * 100, 2))
            + "%"
        )

        plt.axis("off")

    plt.suptitle(
        "Image Reconstruction with Different Numbers of Components"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
