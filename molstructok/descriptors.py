"""
SE(3)-invariant geometric descriptor extraction for molecular 3D structures.
Computes per-atom local spherical coordinate descriptors used by MolStrucTok.
"""

import numpy as np
from rdkit import Chem


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def transform_and_convert(A, B, C, D):
    """Transform D into a local spherical frame defined by (A, B, C)."""
    B_prime = B - A
    C_prime = C - A
    D_prime = D - A

    u = B_prime / np.linalg.norm(B_prime)
    AC_prime = C_prime
    AC_prime_parallel_u = np.dot(AC_prime, u) * u
    v = AC_prime - AC_prime_parallel_u
    v /= np.linalg.norm(v)
    w = np.cross(u, v)

    R = np.column_stack((u, v, w))
    D_double_prime = np.dot(R.T, D_prime)
    x_double_prime, y_double_prime, z_double_prime = D_double_prime
    r, theta, phi = cartesian_to_spherical(x_double_prime, y_double_prime, z_double_prime)
    return r, theta, phi


def find_nearest_neighbors(positions):
    distance_matrix = np.sqrt(((positions[:, None, :] - positions) ** 2).sum(axis=2))
    sorted_indices = np.argsort(distance_matrix, axis=1)
    nearest_indices = sorted_indices[:, :5]
    nearest_positions = positions[nearest_indices]
    return nearest_positions


def extract_understanding_descriptors(positions):
    nearest_neighbor_positions = find_nearest_neighbors(positions)
    vecs = nearest_neighbor_positions[:, 1:5, :] - nearest_neighbor_positions[:, 0:1, :]
    assert vecs.shape == (len(positions), 4, 3), (
        f'Expected more than 4 neighbors (in dim=1), got {vecs.shape}'
    )

    length_descriptors = np.linalg.norm(vecs, axis=-1)
    angle_descriptors = []
    for j in range(4):
        normed_vec_i = vecs[:, j, :] / np.linalg.norm(vecs[:, j, :], axis=1, keepdims=True)
        for k in range(j + 1, 4):
            normed_vec_j = vecs[:, k, :] / np.linalg.norm(vecs[:, k, :], axis=1, keepdims=True)
            angle_cos = np.sum(normed_vec_i * normed_vec_j, axis=1)
            angle_feat = np.arccos(angle_cos)
            angle_descriptors.append(angle_feat)
    angle_descriptors = np.stack(angle_descriptors, axis=1)
    understanding_descriptors = np.concatenate((length_descriptors, angle_descriptors), axis=1)
    return understanding_descriptors


def extract_generation_descriptors(positions, mol, choose_c2):
    num_atoms = mol.GetNumAtoms()
    reference_node_idx = []
    for i in range(num_atoms):
        if i <= 2:
            reference_node_idx.append((i - 1, i - 2, i - 3))
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
                            ((positions - positions[focus_atom_idx]) ** 2).sum(axis=-1)
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

    reference_node_idx = np.array(reference_node_idx)
    focus_atom_positions = positions[reference_node_idx[:, 0]]
    focus_c1_atom_positions = positions[reference_node_idx[:, 1]]
    focus_c2_atom_positions = positions[reference_node_idx[:, 2]]

    descriptors = [[], [], [], []]  # r, theta, phi, sign
    for idx, (A, B, C, D) in enumerate(
        zip(focus_atom_positions, focus_c1_atom_positions, focus_c2_atom_positions, positions)
    ):
        if idx == 0:
            descriptors[0].append(1.2608129506349746)
            descriptors[1].append(1.593612854073428)
            descriptors[2].append(2.1366727766526967)
            descriptors[3].append(0.0)
        elif idx == 1:
            descriptors[0].append(np.linalg.norm(A - D))
            descriptors[1].append(1.593612854073428)
            descriptors[2].append(2.1366727766526967)
            descriptors[3].append(0.0)
        elif idx == 2:
            focus_c1 = np.linalg.norm(B - A)
            focus_cur = np.linalg.norm(B - D)
            c1_cur = np.linalg.norm(A - D)
            x_cur = (focus_c1 ** 2 + focus_cur ** 2 - c1_cur ** 2) / (2 * focus_c1)
            y_cur = np.sqrt(focus_cur ** 2 - x_cur ** 2)
            phi = np.arctan2(y_cur, x_cur)

            descriptors[0].append(focus_cur)
            descriptors[1].append(1.593612854073428)
            descriptors[2].append(phi)
            descriptors[3].append(0.0)
        else:
            r, theta, phi = transform_and_convert(A, B, C, D)
            descriptors[0].append(r)
            descriptors[1].append(theta)
            descriptors[2].append(np.abs(phi))
            descriptors[3].append(np.sign(phi))

    return np.array(descriptors).T


def process_one_mol(mol, choose_c2):
    """Extract both understanding and generation descriptors from an RDKit mol with 3D conformer."""
    num_atoms = mol.GetNumAtoms()
    positions = mol.GetConformer().GetPositions()
    assert num_atoms == len(positions), (
        f'Number of atoms {num_atoms} does not match number of positions {len(positions)}'
    )
    understanding_descriptors = extract_understanding_descriptors(positions)
    generation_descriptors = extract_generation_descriptors(positions, mol, choose_c2)
    return understanding_descriptors, generation_descriptors
