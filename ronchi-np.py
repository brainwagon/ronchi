import math
import time  # timing
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

"""
                     __   _                          
  _______  ___  ____/ /  (_)______  ___    ___  __ __
 / __/ _ \/ _ \/ __/ _ \/ /___/ _ \/ _ \_ / _ \/ // /
/_/  \___/_//_/\__/_//_/_/   /_//_/ .__(_) .__/\_, / 
                                 /_/    /_/   /___/  

A program for generating Ronchi test
patterns using elementary geometric
optics.

Written by Mark VandeWettering, based
on code first written in 2001, but updated in 2015,
and converted to Python in 2025.

Copyright 2025, Mark VandeWettering <mvandewettering@gmail.com>

"""

RESOLUTION = 800

# Global variables, equivalent to C's global definitions
diameter = 114 / 25.4
flen = 2 * 900 / 25.4
offset = -1.0
grating = 100.0
K = -1.0
output_filename = "ronchi.png"  # Default output filename


def np_vec_dot(a, b):
    return np.sum(
        a * b, axis=-1
    )  # Sum along the last axis for dot product of N vectors


def np_vec_len(a):
    return np.sqrt(np_vec_dot(a, a))


def np_vec_normalize(a):
    l = np_vec_len(a)
    # Handle division by zero for zero-length vectors
    l_safe = np.where(
        l == 0, 1.0, l
    )  # Use 1.0 for division if length is 0, then multiply by 0 later
    normalized_a = a / l_safe[..., np.newaxis]  # Ensure broadcasting aligns
    normalized_a = np.where(
        l[..., np.newaxis] == 0, 0.0, normalized_a
    )  # Set to 0 if original length was 0
    return normalized_a


def main():
    global diameter, flen, offset, grating, K, output_filename

    parser = argparse.ArgumentParser(description="Generate Ronchi test patterns.")
    parser.add_argument(
        "-d",
        "--diameter",
        type=float,
        help="Diameter (default: %(default)s)",
        default=12.5,
    )
    parser.add_argument(
        "-f",
        "--focal-length",
        type=float,
        help="Focal length (default: %(default)s)",
        default=62.5,
    )
    parser.add_argument(
        "-g",
        "--grating-frequency",
        type=float,
        help="Grating frequency (default: %(default)s)",
        default=85.0,
    )
    parser.add_argument(
        "-o",
        "--offset",
        type=float,
        help="Offset (default: %(default)s)",
        default=-0.25,
    )
    parser.add_argument(
        "-k",
        "--conic-constant",
        type=float,
        help="Conic constant K (default: -1.0)",
        default=-1,
    )
    parser.add_argument(
        "--out",
        type=str,
        default=output_filename,
        help=f"Output filename (default: {output_filename}). "
        "Pillow will infer format from extension (e.g., .pgm, .png).",
    )

    args = parser.parse_args()

    diameter = args.diameter
    flen = args.focal_length
    grating = args.grating_frequency
    offset = args.offset
    K = args.conic_constant
    output_filename = args.out

    print("Starting calculations...")
    start_total_time = time.perf_counter()

    r = diameter / 2.0

    coords_1d = (np.arange(RESOLUTION) + 0.5) * diameter / RESOLUTION - diameter / 2.0

    fx_grid, fy_grid = np.meshgrid(coords_1d, coords_1d, indexing="xy")

    pixels_array = np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)

    P_x = fx_grid
    P_y = fy_grid

    outside_aperture_mask = (P_x * P_x + P_y * P_y) > r * r

    P_z = np.zeros_like(fx_grid)

    parabola_mask = np.abs(K + 1.0) < 1e-5
    conic_mask = ~parabola_mask

    if np.any(parabola_mask):
        # Only compute where parabola_mask is True. This mask might be a single boolean if K is global.
        # It's better to compute for all and then apply the mask or conditional assignment.
        # Let's compute for all, and then selectively apply.
        P_z_parabola_val = (P_x**2 + P_y**2) / (4.0 * flen)
        # Apply only to the points where K is effectively -1
        # This requires careful masking if K itself is a global variable.
        # Since K is a global scalar, parabola_mask will be a single boolean.
        # So we just do a simple if/else for the whole array.
        if parabola_mask:  # K is -1 (scalar)
            P_z = (P_x**2 + P_y**2) / (4.0 * flen)
        else:  # K is not -1 (scalar)
            discriminant = (K + 1) * (-(P_x**2) - P_y**2) + 4 * flen * flen
            # Handle negative discriminant: set Pz to a value that will result in black
            # or handle it later by masking those pixels.
            # For now, let's set to NaN or a sentinel value, then deal with it.
            P_z_conic_val = (2 * flen - np.sqrt(np.maximum(0, discriminant))) / (K + 1)
            # Mark points with negative discriminant
            invalid_discriminant_mask = discriminant < 0
            P_z_conic_val[invalid_discriminant_mask] = (
                np.nan
            )  # Use NaN to mark invalid points
            P_z = P_z_conic_val

    # Combine Px, Py, Pz into a single 3D array of vectors for P
    # Shape: (RESOLUTION, RESOLUTION, 3)
    P_points = np.stack([P_x, P_y, P_z], axis=-1)  # Stacks along a new last axis

    # Origin O
    O_point = np.array([0.0, 0.0, 2.0 * flen + offset])

    I_vectors = P_points - O_point
    I_vectors = np_vec_normalize(I_vectors)

    N_vectors = np.empty_like(P_points)
    N_vectors[..., 0] = -2.0 * P_points[..., 0]  # N_x = -2 * P_x
    N_vectors[..., 1] = -2.0 * P_points[..., 1]  # N_y = -2 * P_y
    N_vectors[..., 2] = 4.0 * flen - 2.0 * (K + 1) * P_points[..., 2]  # N_z
    N_vectors = np_vec_normalize(N_vectors)

    c_values = -2.0 * np_vec_dot(I_vectors, N_vectors)

    c_values_expanded = c_values[..., np.newaxis]

    R_vectors = c_values_expanded * N_vectors + I_vectors
    R_vectors = np_vec_normalize(R_vectors)
    R_z_values = R_vectors[..., 2]

    # Numerator
    numerator = 2.0 * flen + offset - P_points[..., 2]

    # Handle division by zero for t: where R_z_values is zero, t would be inf or NaN.
    # We'll mask these points out later.
    t_values = np.zeros_like(R_z_values)  # Initialize t_values

    # Create a mask for valid denominators (non-zero R_z_values)
    valid_denom_mask = R_z_values != 0
    t_values[valid_denom_mask] = (
        numerator[valid_denom_mask] / R_z_values[valid_denom_mask]
    )

    # Mark points where R_z_values is zero as NaN or some sentinel, or directly handle them.
    # Let's set pixels to black if R_z_values is zero or t <= 0.
    invalid_t_mask = (R_z_values == 0) | (t_values <= 0.0)  # Combine conditions

    gx_values = (P_points[..., 0] + t_values * R_vectors[..., 0]) * 2.0 * grating - 0.5

    # Calculate pixel values
    # Pixels are white (255) if floor(gx) is odd, else 32 (gray)
    floor_gx = np.floor(gx_values).astype(int)

    # Apply the Ronchi pattern logic
    ronchi_pattern_mask = (
        floor_gx % 2
    ) != 0  # Odd -> True (white), Even -> False (gray)

    # Initialize all pixels to the gray value (32) first
    pixels_array.fill(32)
    # Then set white pixels
    pixels_array[ronchi_pattern_mask] = 255

    # --- Apply all masks for invalid points ---
    # Points outside aperture are black
    pixels_array[outside_aperture_mask] = 0

    # Points with invalid discriminant are black
    # pixels_array[invalid_discriminant_mask] = 0

    # Points with invalid t are black
    pixels_array[invalid_t_mask] = 0

    end_calc_time = time.perf_counter()
    print(f"Calculation time: {end_calc_time - start_total_time:.6f} seconds")

    # Save the image using Pillow
    start_save_time = time.perf_counter()
    try:
        img = Image.fromarray(pixels_array)  # Create image from NumPy array

        # label the image
        label_str = f"{diameter} inch, {flen} FL, f/{flen/diameter:.2f}, {grating} lpi"
        print(label_str)
        font = ImageFont.load_default(size=16)
        i1 = ImageDraw.Draw(img)
        i1.text((0, 0), label_str, fill=(255), font=font)
        i1.text(
            (0, 16),
            f'{abs(offset)} inch {"inside" if offset < 0 else "outside"} of focus',
            fill=(255),
            font=font,
        )

        img.save(output_filename)
        print(f"Image saved to {output_filename}")
        img.show()
    except Exception as e:
        print(f"Error saving image: {e}")
    end_save_time = time.perf_counter()
    print(f"Image saving time: {end_save_time - start_save_time:.6f} seconds")
    print(f"Total execution time: {end_save_time - start_total_time:.6f} seconds")


if __name__ == "__main__":
    main()
