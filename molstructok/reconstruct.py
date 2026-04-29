"""
3D coordinate reconstruction from SE(3)-invariant spherical descriptors.
Recovers atomic positions by autoregressively traversing the molecular graph
and rebuilding each local spherical frame.
"""

import numpy as np
from rdkit import Chem


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def transform_back_to_original(A, B, C, r, theta, phi):
    """Convert local spherical coordinates back to global Cartesian coordinates."""
    x_double_prime, y_double_prime, z_double_prime = spherical_to_cartesian(r, theta, phi)
    D_double_prime = np.array([x_double_prime, y_double_prime, z_double_prime])

    B_prime = B - A
    C_prime = C - A

    u = B_prime / np.linalg.norm(B_prime)
    AC_prime = C_prime
    AC_prime_parallel_u = np.dot(AC_prime, u) * u
    v = AC_prime - AC_prime_parallel_u
    v /= np.linalg.norm(v)
    w = np.cross(u, v)

    R = np.column_stack((u, v, w))
    D_prime = np.dot(R, D_double_prime)
    D_original = D_prime + A
    return D_original


def generate_coords(feats, mol, choose_c2):
    """Reconstruct 3D coordinates from per-atom spherical descriptors (r, theta, phi, sign)."""
    reference_node_idx = []
    coords = []

    for i, (r, theta, phi, sign) in enumerate(feats):
        if i == 0:
            reference_node_idx.append((0, 0, 0))
            coords.append(np.array([0, 0, 0]))
        elif i == 1:
            reference_node_idx.append((0, 0, 0))
            coords.append(np.array([r, 0, 0]))
        elif i == 2:
            reference_node_idx.append((1, 0, -1))
            rel_x = np.cos(phi) * r
            rel_y = np.sin(phi) * r
            coords.append(np.array([rel_x, rel_y, 0]))
        else:
            for j in range(i)[::-1]:
                if mol.GetBondBetweenAtoms(i, j) is not None:
                    focus_atom_idx = j
                    if choose_c2 == 'recurrent-index':
                        if focus_atom_idx == 1:
                            focus_c1_atom_idx = 0
                            focus_c2_atom_idx = 2
                        else:
                            focus_c1_atom_idx = reference_node_idx[j][0]
                            focus_c2_atom_idx = reference_node_idx[j][1]
                    elif choose_c2 == 'c1-closest':
                        distance_array = np.sqrt(
                            ((np.array(coords) - np.array(coords)[focus_atom_idx]) ** 2).sum(axis=-1)
                        )
                        sorted_indices = np.argsort(distance_array)
                        nearest_candidates = sorted_indices[sorted_indices < i][1:]
                        focus_c1_atom_idx = nearest_candidates[0]
                        focus_c2_atom_idx = nearest_candidates[1]

                    reference_node_idx.append((focus_atom_idx, focus_c1_atom_idx, focus_c2_atom_idx))
                    break
            assert (
                focus_c2_atom_idx >= 0 and focus_c1_atom_idx >= 0 and focus_atom_idx >= 0
            ), (
                f'focus_c2_atom_idx: {focus_c2_atom_idx}, '
                f'focus_c1_atom_idx: {focus_c1_atom_idx}, '
                f'focus_atom_idx: {focus_atom_idx}'
            )

            focus_atom_positions = coords[focus_atom_idx]
            focus_c1_atom_positions = coords[focus_c1_atom_idx]
            focus_c2_atom_positions = coords[focus_c2_atom_idx]

            new_coords = transform_back_to_original(
                focus_atom_positions, focus_c1_atom_positions,
                focus_c2_atom_positions, r, theta, phi * sign,
            )
            coords.append(new_coords)

    return np.array(coords)
