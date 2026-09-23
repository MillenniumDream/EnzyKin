//! Data structures and relative functions for descriptor
use crate::common::constants::{Element, ANGSTROM_TO_BOHR};
use crate::common::error::*;
use crate::nn::fragment_definition::{AminoAcid, Molecule, Cap, FragmentType, FragmentType::Atom};
use crate::nn::protein::ProteinSystem;
use std::collections::HashMap;
use ndarray::{Array2, Array3, Array4, Array5};
use lazy_static::lazy_static;










pub const RCUT: f64 = 5.0 * ANGSTROM_TO_BOHR;
pub const NATOM_WITHIN_2RCUT: usize = 2000;        // The estimated maximum number of atoms within 2*Rcut around a fragment, only using for allocation
pub const NATOM_WITHIN_RCUT: usize = 400;        // The estimated maximum number of atoms within Rcut around a fragment, only using for allocation

// Mean value and standard deviation for the terms of atomic number, distance, and angle in descriptor
pub const MU_Z: [f64; 4] = [0.0, 1.0/3.0, 2.0/3.0, 1.0];
pub const SIGMA_Z: f64 = 1.0/3.0;
pub const MU_D: [f64; 5] = [RCUT/10.0, RCUT*3.0/10.0, RCUT*5.0/10.0, RCUT*7.0/10.0, RCUT*9.0/10.0];
pub const SIGMA_D: f64 = RCUT/5.0;
pub const MU_THETA: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];
pub const SIGMA_THETA: f64 = 0.5;
// Whole number of the descriptor
pub const N_DES: usize = (MU_Z.len()*MU_D.len()) * (MU_Z.len()*MU_D.len() + 1) * MU_THETA.len() / 2;










/// Update the neighbor list (within 2*RCUT) of a specified fragment
///
/// # Parameters
/// ```
/// s: the input protein system
/// index: the input index of the fragment in the protein system
/// neighbor_list: the updated neighbor list of the specified fragment
/// ```
pub fn update_neighbor_list(s: &mut ProteinSystem, index: usize)
{
    let rnei_powi2: f64 = (RCUT * 2.0).powi(2);
    let mut neighbor_list: Vec<usize> = Vec::with_capacity(NATOM_WITHIN_2RCUT);
    let mut within_fragment: bool;

    match &s.fragment[index].atom_list
    {
        Some(atom_list) =>
        {
            match &s.cell
            {
                // If the input protein system is periodic
                Some(cell) =>
                {
                    // Extend the cell to 3*3*3 
                    let mut coord: Array2<f64> = Array2::zeros((s.fragment[index].natom*27, 3));
                    let mut n: usize = 0;
                    for i in [0, -1, 1]
                    {
                        for j in [0, -1, 1]
                        {
                            for k in [0, -1, 1]
                            {
                                for l in 0..s.fragment[index].natom
                                {
                                    coord[[n,0]] = s.fragment[index].coord[[l,0]] + cell[[0,0]] * (i as f64) + cell[[1,0]] * (j as f64) + cell[[2,0]] * (k as f64);
                                    coord[[n,1]] = s.fragment[index].coord[[l,1]] + cell[[0,1]] * (i as f64) + cell[[1,1]] * (j as f64) + cell[[2,1]] * (k as f64);
                                    coord[[n,2]] = s.fragment[index].coord[[l,2]] + cell[[0,2]] * (i as f64) + cell[[1,2]] * (j as f64) + cell[[2,2]] * (k as f64);
                                    n += 1;
                                }
                            }
                        }
                    }

                    // Achieve the neighbor list
                    for i in 0..s.natom
                    {
                        within_fragment = false;
                        for j in 0..s.fragment[index].natom
                        {
                            if i == atom_list[j]
                            {
                                within_fragment = true;
                                break
                            }
                        }
                        if within_fragment == false
                        {
                            for j in 0..coord.nrows()
                            {
                                if ( (s.coord[[i,0]] - coord[[j,0]]).powi(2) + (s.coord[[i,1]] - coord[[j,1]]).powi(2) + (s.coord[[i,2]] - coord[[j,2]]).powi(2) ) < rnei_powi2
                                {
                                    neighbor_list.push(i);
                                    break
                                }
                            }
                        }
                    }
                },

                // If the input protein system is aperiodic
                None =>
                {
                    // Achieve the neighbor list
                    for i in 0..s.natom
                    {
                        within_fragment = false;
                        for j in 0..s.fragment[index].natom
                        {
                            if i == atom_list[j]
                            {
                                within_fragment = true;
                                break
                            }
                        }
                        if within_fragment == false
                        {
                            for j in 0..s.fragment[index].natom
                            {
                                if ( (s.coord[[i,0]] - s.fragment[index].coord[[j,0]]).powi(2) + (s.coord[[i,1]] - s.fragment[index].coord[[j,1]]).powi(2) + (s.coord[[i,2]] - s.fragment[index].coord[[j,2]]).powi(2) ) < rnei_powi2
                                {
                                    neighbor_list.push(i);
                                    break
                                }
                            }
                        }
                    }
                },
            }
        },

        None => panic!("{}", error_none_value("s.fragment[index].atom_list")),
    }

    s.fragment[index].neighbor_list = Some(neighbor_list);
}










/// Truncate a cluster from the input protein system, containing all the atoms within Rcut around the specified fragment
///
/// # Parameters
/// ```
/// s: the input protein system
/// index: the input index of the fragment in the protein system
/// coord_within_rcut: the output coordinates of the atoms within Rcut around the specified fragment
/// list_within_rcut: the output indices (corresponding to s.coord) of the atoms within Rcut around the specified fragment
/// atomic_number_within_rcut: the output atomic numbers of the atoms within Rcut around the specified fragment
/// 
/// ```
pub fn truncate_cluster(s: &ProteinSystem, index: usize) -> (Array2<f64>, Vec<usize>, Vec<usize>)
{
    let mut coord_within_rcut: Vec<f64> = Vec::with_capacity(NATOM_WITHIN_RCUT*3);
    let mut list_within_rcut: Vec<usize> = Vec::with_capacity(NATOM_WITHIN_RCUT);
    let centered_atom_index: usize;



    // Obtain the index the centered atom (corresponding to s.fragment[index].coord)
    match &s.fragment[index].fragment_type
    {
        // For amino acid residue
        FragmentType::Residue(amino_acid) =>
        {
            match amino_acid
            {
                // For the special proline, the index of the C_alpha atom is 10
                AminoAcid::PRO =>
                {
                    centered_atom_index = 10;
                },

                // For the other ordinary amino acid residue, the index of the C_alpha atom is 2
                _ =>
                {
                    centered_atom_index = 2;
                },
            }
        },

        // For molecule, the index of the O atom is 0
        FragmentType::Molecule(molecule) =>
        {
            match molecule
            {
                // For water
                Molecule::WAT =>
                {
                    centered_atom_index = 0;
                },
            }
        },

        // For head
        FragmentType::Head(cap) =>
        {
            match cap
            {
                // For ACE
                Cap::ACE =>
                {
                    centered_atom_index = 1;
                },

                // For Head(H)
                _ =>
                {
                    centered_atom_index = 0;
                },
            }
        },

        // For tail
        FragmentType::Tail(cap) =>
        {
            match cap
            {
                // For NME
                Cap::NME =>
                {
                    centered_atom_index = 2;
                },

                // For NHE
                Cap::NHE =>
                {
                    centered_atom_index = 0;
                },

                // For Tail(O) or Tail(H)
                _ =>
                {
                    centered_atom_index = 0;
                },
            }
        },

        // For atom
        Atom(_element) =>
        {
            centered_atom_index = 0;
        },
    }



    // Copy the fragment: the centered atom copied first, and then followed by the other atoms in order
    match &s.fragment[index].atom_list
    {
        Some(atom_list) =>
        {
            coord_within_rcut.push(s.fragment[index].coord[[centered_atom_index,0]]);
            coord_within_rcut.push(s.fragment[index].coord[[centered_atom_index,1]]);
            coord_within_rcut.push(s.fragment[index].coord[[centered_atom_index,2]]);
            list_within_rcut.push(atom_list[centered_atom_index]);
            for i in 0..s.fragment[index].natom
            {
                if i != centered_atom_index
                {
                    coord_within_rcut.push(s.fragment[index].coord[[i,0]]);
                    coord_within_rcut.push(s.fragment[index].coord[[i,1]]);
                    coord_within_rcut.push(s.fragment[index].coord[[i,2]]);
                    list_within_rcut.push(atom_list[i]);
                }
            }
        },

        None => panic!("{}", error_none_value("s.fragment[index].atom_list")),
    }



    // Truncate the cluster
    let rcut_powi2: f64 = RCUT.powi(2);
    match (&s.cell, &s.fragment[index].neighbor_list)
    {
        // For aperiodic system without a neighbor list
        (None, None) =>
        {
            let mut within_fragment: bool;
            for i in 0..s.natom
            {
                within_fragment = false;
                for j in 0..s.fragment[index].natom
                {
                    if i == list_within_rcut[j]
                    {
                        within_fragment = true;
                        break
                    }
                }
                if within_fragment == false
                {
                    for j in 0..s.fragment[index].natom
                    {
                        if ( (s.coord[[i,0]] - s.fragment[index].coord[[j,0]]).powi(2) + (s.coord[[i,1]] - s.fragment[index].coord[[j,1]]).powi(2) + (s.coord[[i,2]] - s.fragment[index].coord[[j,2]]).powi(2) ) < rcut_powi2
                        {
                            coord_within_rcut.push(s.coord[[i,0]]);
                            coord_within_rcut.push(s.coord[[i,1]]);
                            coord_within_rcut.push(s.coord[[i,2]]);
                            list_within_rcut.push(i);
                            break
                        }
                    }
                }
            }
        },

        // For aperiodic system with a neighbor list
        (None, Some(neighbor_list)) =>
        {
            for i in 0..neighbor_list.len()
            {
                for j in 0..s.fragment[index].natom
                {
                    if ( (s.coord[[neighbor_list[i],0]] - s.fragment[index].coord[[j,0]]).powi(2) + (s.coord[[neighbor_list[i],1]] - s.fragment[index].coord[[j,1]]).powi(2) + (s.coord[[neighbor_list[i],2]] - s.fragment[index].coord[[j,2]]).powi(2) ) < rcut_powi2
                    {
                        coord_within_rcut.push(s.coord[[neighbor_list[i],0]]);
                        coord_within_rcut.push(s.coord[[neighbor_list[i],1]]);
                        coord_within_rcut.push(s.coord[[neighbor_list[i],2]]);
                        list_within_rcut.push(neighbor_list[i]);
                        break
                    }
                }
            }
        },

        // For periodic system without a neighbor list
        (Some(cell), None) =>
        {
            let mut within_fragment: bool;
            let mut within_rcut: bool;
            let mut x: f64;
            let mut y: f64;
            let mut z: f64;
            for i in 0..s.natom
            {
                within_fragment = false;
                for j in 0..s.fragment[index].natom
                {
                    if i == list_within_rcut[j]
                    {
                        within_fragment = true;
                        break
                    }
                }
                if within_fragment == false
                {
                    within_rcut = false;
                    for j in 0..s.fragment[index].natom
                    {
                        for n in [0, -1, 1]
                        {
                            for m in [0, -1, 1]
                            {
                                for l in [0, -1, 1]
                                {
                                    x = s.coord[[i,0]] + cell[[0,0]] * (n as f64) + cell[[1,0]] * (m as f64) + cell[[2,0]] * (l as f64);
                                    y = s.coord[[i,1]] + cell[[0,1]] * (n as f64) + cell[[1,1]] * (m as f64) + cell[[2,1]] * (l as f64);
                                    z = s.coord[[i,2]] + cell[[0,2]] * (n as f64) + cell[[1,2]] * (m as f64) + cell[[2,2]] * (l as f64);
                                    if ( (x - s.fragment[index].coord[[j,0]]).powi(2) + (y - s.fragment[index].coord[[j,1]]).powi(2) + (z - s.fragment[index].coord[[j,2]]).powi(2) ) < rcut_powi2
                                    {
                                        coord_within_rcut.push(x);
                                        coord_within_rcut.push(y);
                                        coord_within_rcut.push(z);
                                        list_within_rcut.push(i);
                                        within_rcut = true;
                                        break
                                    }
                                }
                                if within_rcut
                                {
                                    break
                                }
                            }
                            if within_rcut
                            {
                                break
                            }
                        }
                        if within_rcut
                        {
                            break
                        }
                    }
                }
            }
        },

        // For periodic system with a neighbor list
        (Some(cell), Some(neighbor_list)) =>
        {
            let mut within_rcut: bool;
            let mut x: f64;
            let mut y: f64;
            let mut z: f64;
            for i in 0..neighbor_list.len()
            {
                within_rcut = false;
                for j in 0..s.fragment[index].natom
                {
                    for n in [0, -1, 1]
                    {
                        for m in [0, -1, 1]
                        {
                            for l in [0, -1, 1]
                            {
                                x = s.coord[[neighbor_list[i],0]] + cell[[0,0]] * (n as f64) + cell[[1,0]] * (m as f64) + cell[[2,0]] * (l as f64);
                                y = s.coord[[neighbor_list[i],1]] + cell[[0,1]] * (n as f64) + cell[[1,1]] * (m as f64) + cell[[2,1]] * (l as f64);
                                z = s.coord[[neighbor_list[i],2]] + cell[[0,2]] * (n as f64) + cell[[1,2]] * (m as f64) + cell[[2,2]] * (l as f64);
                                if ( (x - s.fragment[index].coord[[j,0]]).powi(2) + (y - s.fragment[index].coord[[j,1]]).powi(2) + (z - s.fragment[index].coord[[j,2]]).powi(2) ) < rcut_powi2
                                {
                                    coord_within_rcut.push(x);
                                    coord_within_rcut.push(y);
                                    coord_within_rcut.push(z);
                                    list_within_rcut.push(neighbor_list[i]);
                                    within_rcut = true;
                                    break
                                }
                            }
                            if within_rcut
                            {
                                break
                            }
                        }
                        if within_rcut
                        {
                            break
                        }
                    }
                    if within_rcut
                    {
                        break
                    }
                }
            }
        },
    }
    let coord_within_rcut: Array2<f64> = Array2::from_shape_vec((list_within_rcut.len(), 3), coord_within_rcut).expect(&error_none_value("coord_within_rcut"));



    // Obtain the atomic numbers
    let mut atomic_number_within_rcut: Vec<usize> = Vec::with_capacity(list_within_rcut.len());
    for i in 0..list_within_rcut.len()
    {
        atomic_number_within_rcut.push(s.atom_type[list_within_rcut[i]].get_atomic_number());
    }



    (coord_within_rcut, list_within_rcut, atomic_number_within_rcut)
}










/// Derive the descriptor for the input cluster around the specified fragment based on the atomic coordinates.
///
/// # Parameters
/// ```
/// coord_within_rcut: the input coordinates of the atoms within Rcut around the specified fragment
/// atomic_number_within_rcut: the input atomic numbers of the atoms within Rcut around the specified fragment
/// natom: the number of atoms of the fragment
/// ```
pub fn get_descriptor(coord_within_rcut: Array2<f64>, atomic_number_within_rcut: Vec<usize>, natom: usize) -> (Vec<f64>, Array2<f64>)
{
    let ncut_1: usize = atomic_number_within_rcut.len() - 1;        // The number of atoms (exclude the central atom, e.g. C_alpha) within Rcut around the fragment
    let n_mu_z: usize = MU_Z.len();        // The number of sampling in atomic number
    let n_mu_d: usize = MU_D.len();        // The number of sampling in interatomic distance
    let n_mu_theta: usize = MU_THETA.len();                // The number of sampling in interatomic angle
    let mut x: f64;
    let mut y: f64;
    let mut z: f64;



    // About the atomic number terms
    let mut gaussian_z: Array2<f64> = Array2::zeros((ncut_1, n_mu_z));                // gaussian(z, mu_z, sigma_z)
    for i in 0..ncut_1                // For each neighbor atom (exclude the central atom, e.g. C_alpha)
    {
        for j in 0..n_mu_z
        {
            gaussian_z[[i,j]] = GAUSSIAN_Z_MUZ[[atomic_number_within_rcut[i+1]-1, j]];
        }
    }



    // About the distance terms
    // Obtain interatomic distances between the neighbor atoms and the fragment atoms, along with the function values based on the interatomic distances
    let mut d: Array2<f64> = Array2::zeros((ncut_1, natom));                // d_ik, i for the neighbor atoms, j for the fragment atoms
    let mut diff_xyz_d: Array3<f64> = Array3::zeros((ncut_1, natom, 3));                // (x_i - x_k) / d_ik
    let mut fcut_d: Array2<f64> = Array2::zeros((ncut_1, natom));                // fcut(d_ik, RCUT)
    let mut dfcut_d: Array2<f64> = Array2::zeros((ncut_1, natom));                // dfcut(d_ik, RCUT)
    let mut gaussian_d: Array3<f64> = Array3::zeros((ncut_1, natom, n_mu_d));                // gaussian(d_ik, mu_d, sigma_d)
    let mut d_fcut_gaussian: Array3<f64> = Array3::zeros((ncut_1, natom, n_mu_d));                // d(fcut*gaussian)(d_ik, mu_d, sigma_d)
    for i in 0..ncut_1                // For each neighbor atom (exclude the central atom, e.g. C_alpha)
    {
        for j in 0..natom                // For each fragment atom
        {
            if (i+1) != j
            {
                x = coord_within_rcut[[i+1,0]] - coord_within_rcut[[j,0]];
                y = coord_within_rcut[[i+1,1]] - coord_within_rcut[[j,1]];
                z = coord_within_rcut[[i+1,2]] - coord_within_rcut[[j,2]];
                d[[i,j]] = (x*x + y*y + z*z).sqrt();
                diff_xyz_d[[i,j,0]] = x / d[[i,j]];
                diff_xyz_d[[i,j,1]] = y / d[[i,j]];
                diff_xyz_d[[i,j,2]] = z / d[[i,j]];
            }
            fcut_d[[i,j]] = fcut(d[[i,j]], RCUT);
            dfcut_d[[i,j]] = dfcut(d[[i,j]], RCUT);
            for k in 0..n_mu_d
            {
                gaussian_d[[i,j,k]] = gaussian(d[[i,j]], MU_D[k], SIGMA_D);
                d_fcut_gaussian[[i,j,k]] = dfcut_d[[i,j]] * gaussian_d[[i,j,k]] + fcut_d[[i,j]] * dgaussian(d[[i,j]], MU_D[k], SIGMA_D, gaussian_d[[i,j,k]]);
            }
        }
    }
    // Achieve some summation terms over the atoms of the specified fragment (natom), or some combination terms
    let mut sum_fcut_gaussian: Array2<f64> = Array2::zeros((ncut_1, n_mu_d));                // sum(k) ( fcut(d_ik, RCUT)*gaussian(d_ik, mu_d, sigma_d) )
    let mut com_d_fcut_gaussian: Array4<f64> = Array4::zeros((ncut_1, natom, n_mu_d, 3));        // d(fcut*gaussian)(d_ik, mu_d, sigma_d) * (x_i - x_k) / d_ik
    let mut sum_d_fcut_gaussian: Array3<f64> = Array3::zeros((ncut_1, n_mu_d, 3));    // sum(k) ( d(fcut*gaussian)(d_ik, mu_d, sigma_d) * (x_i - x_k) / d_ik )
    for i in 0..ncut_1                // For each neighbor atom (exclude the central atom, e.g. C_alpha)
    {
        for j in 0..natom                // For each fragment atom
        {
            for k in 0..n_mu_d
            {
                sum_fcut_gaussian[[i,k]] += fcut_d[[i,j]] * gaussian_d[[i,j,k]];
                com_d_fcut_gaussian[[i,j,k,0]] = d_fcut_gaussian[[i,j,k]] * diff_xyz_d[[i,j,0]];
                com_d_fcut_gaussian[[i,j,k,1]] = d_fcut_gaussian[[i,j,k]] * diff_xyz_d[[i,j,1]];
                com_d_fcut_gaussian[[i,j,k,2]] = d_fcut_gaussian[[i,j,k]] * diff_xyz_d[[i,j,2]];
                sum_d_fcut_gaussian[[i,k,0]] += com_d_fcut_gaussian[[i,j,k,0]];
                sum_d_fcut_gaussian[[i,k,1]] += com_d_fcut_gaussian[[i,j,k,1]];
                sum_d_fcut_gaussian[[i,k,2]] += com_d_fcut_gaussian[[i,j,k,2]];
            }
        }
    }



    // About the angle terms
    // Obtain interatomic angles with central atom (e.g. C_alpha) being the center atom, along with the function values based on the interatomic angles
    let mut dot_r_d: Array2<f64> = Array2::zeros((ncut_1, ncut_1));                // (r_i0 * r_j0) / (d_i0 * d_j0)
    let mut gaussian_theta: Array3<f64> = Array3::zeros((ncut_1, ncut_1, n_mu_theta));        // gaussian((r_i0 * r_j0) / (d_i0 * d_j0), mu_theta, sigma_theta)
    let mut dgaussian_theta: Array3<f64> = Array3::zeros((ncut_1, ncut_1, n_mu_theta));        // dgaussian((r_i0 * r_j0) / (d_i0 * d_j0), mu_theta, sigma_theta)
    for i in 0..ncut_1                // For each neighbor atom (exclude the central atom, e.g. C_alpha)
    {
        for j in 0..ncut_1                // For each neighbor atom (exclude the central atom, e.g. C_alpha)
        {
            if i <= j
            {
                dot_r_d[[i,j]] = diff_xyz_d[[i,0,0]] * diff_xyz_d[[j,0,0]] + diff_xyz_d[[i,0,1]] * diff_xyz_d[[j,0,1]] + diff_xyz_d[[i,0,2]] * diff_xyz_d[[j,0,2]];
                for k in 0..n_mu_theta
                {
                    gaussian_theta[[i,j,k]] = gaussian(dot_r_d[[i,j]], MU_THETA[k], SIGMA_THETA);
                    dgaussian_theta[[i,j,k]] = dgaussian(dot_r_d[[i,j]], MU_THETA[k], SIGMA_THETA, gaussian_theta[[i,j,k]]);
                }
            }
            else
            {
                dot_r_d[[i,j]] = dot_r_d[[j,i]];
                for k in 0..n_mu_theta
                {
                    gaussian_theta[[i,j,k]] = gaussian_theta[[j,i,k]];
                    dgaussian_theta[[i,j,k]] = dgaussian_theta[[j,i,k]];
                }
            }
        }
    }
    // Achieve some combination terms
    let mut com_d_gaussian_theta: Array4<f64> = Array4::zeros((ncut_1, ncut_1, n_mu_theta, 3));
    for i in 0..ncut_1                // For each neighbor atom (exclude the central atom, e.g. C_alpha)
    {
        for j in 0..ncut_1                // For each neighbor atom (exclude the central atom, e.g. C_alpha)
        {
            x = (diff_xyz_d[[j,0,0]] - diff_xyz_d[[i,0,0]] * dot_r_d[[i,j]]) / d[[i,0]];
            y = (diff_xyz_d[[j,0,1]] - diff_xyz_d[[i,0,1]] * dot_r_d[[i,j]]) / d[[i,0]];
            z = (diff_xyz_d[[j,0,2]] - diff_xyz_d[[i,0,2]] * dot_r_d[[i,j]]) / d[[i,0]];
            for k in 0..n_mu_theta
            {
                com_d_gaussian_theta[[i,j,k,0]] = dgaussian_theta[[i,j,k]] * x;
                com_d_gaussian_theta[[i,j,k,1]] = dgaussian_theta[[i,j,k]] * y;
                com_d_gaussian_theta[[i,j,k,2]] = dgaussian_theta[[i,j,k]] * z;
            }
        }
    }



    // About the cross terms
    let mut cross_z_d: Array3<f64> = Array3::zeros((ncut_1, n_mu_z, n_mu_d));   // gaussian(z, mu_z, sigma_z) * sum(k) ( (fcut*gaussian)(d_ik, mu_d, sigma_d) )
    let mut cross_comd_z_d: Array4<f64> = Array4::zeros((ncut_1, n_mu_z, n_mu_d, natom*3));
    let mut cross_sumd_z_d: Array4<f64> = Array4::zeros((ncut_1, n_mu_z, n_mu_d, 3));
    for i in 0..ncut_1                // For each neighbor atom (exclude the central atom, e.g. C_alpha)
    {
        for j in 0..n_mu_z
        {
            for k in 0..n_mu_d
            {
                cross_z_d[[i,j,k]] = gaussian_z[[i,j]] * sum_fcut_gaussian[[i,k]];
                for n in 0..natom                // For each differential dimension
                {
                    cross_comd_z_d[[i,j,k,3*n]] = gaussian_z[[i,j]] * com_d_fcut_gaussian[[i,n,k,0]];
                    cross_comd_z_d[[i,j,k,3*n+1]] = gaussian_z[[i,j]] * com_d_fcut_gaussian[[i,n,k,1]];
                    cross_comd_z_d[[i,j,k,3*n+2]] = gaussian_z[[i,j]] * com_d_fcut_gaussian[[i,n,k,2]];
                }
                cross_sumd_z_d[[i,j,k,0]] = gaussian_z[[i,j]] * sum_d_fcut_gaussian[[i,k,0]];
                cross_sumd_z_d[[i,j,k,1]] = gaussian_z[[i,j]] * sum_d_fcut_gaussian[[i,k,1]];
                cross_sumd_z_d[[i,j,k,2]] = gaussian_z[[i,j]] * sum_d_fcut_gaussian[[i,k,2]];
            }
        }
    }
    let mut cross_z_d_theta: Array4<f64> = Array4::zeros((ncut_1, n_mu_z, n_mu_d, n_mu_theta));
    let mut cross_comd_z_d_theta: Array5<f64> = Array5::zeros((ncut_1, n_mu_z, n_mu_d, n_mu_theta, 3));
    for n_z in 0..n_mu_z
    {
        for n_d in 0..n_mu_d
        {
            for n_theta in 0..n_mu_theta
            {
                for i in 0..ncut_1                // For the first neighbor atom (exclude the central atom, e.g. C_alpha)
                {
                    for j in 0..ncut_1                // For the second neighbor atom (exclude the central atom, e.g. C_alpha)
                    {
                        cross_z_d_theta[[i, n_z, n_d, n_theta]] += gaussian_theta[[i, j, n_theta]] * cross_z_d[[j, n_z, n_d]];
                        cross_comd_z_d_theta[[i, n_z, n_d, n_theta, 0]] += com_d_gaussian_theta[[i, j, n_theta, 0]] * cross_z_d[[j, n_z, n_d]];
                        cross_comd_z_d_theta[[i, n_z, n_d, n_theta, 1]] += com_d_gaussian_theta[[i, j, n_theta, 1]] * cross_z_d[[j, n_z, n_d]];
                        cross_comd_z_d_theta[[i, n_z, n_d, n_theta, 2]] += com_d_gaussian_theta[[i, j, n_theta, 2]] * cross_z_d[[j, n_z, n_d]];
                    }
                }
            }
        }
    }



    // Derive the descriptor and the corresponding gradient based on the atomic coordinates and atomic numbers of the input cutoff cluster
    let mut descriptor: Vec<f64> = vec![0.0; N_DES];
    let mut gradient: Array2<f64> = Array2::zeros((ncut_1*3+3, N_DES));
    let mut index: usize = 0;
    let mut n_d2_start: usize;
    let mut c: f64;
    for n_z1 in 0..n_mu_z                // For each sampling in the atomic number for the first atom
    {
        for n_z2 in n_z1..n_mu_z                // For each sampling in the atomic number for the second atom
        {
            // A sampling is used for the interatomic distances and angles
            for n_d1 in 0..n_mu_d
            {
                // There is a exchange symmetry for the two atoms, so the exchange of (n_z1, n_d1) with (n_z2, n_z2) achieve the same descriptor
                if n_z2 == n_z1
                {
                    n_d2_start = n_d1;
                }
                else
                {
                    n_d2_start = 0;
                }
                for n_d2 in n_d2_start..n_mu_d
                {
                    for n_theta in 0..n_mu_theta
                    {
                        for i in 0..ncut_1                // For the first neighbor atom (exclude the central atom, e.g. C_alpha)
                        {
                            descriptor[index] += cross_z_d[[i, n_z1, n_d1]] * cross_z_d_theta[[i, n_z2, n_d2, n_theta]];

                            for n in 0..3*natom                // For each differential dimension
                            {
                                gradient[[n,index]] -= cross_comd_z_d[[i, n_z1, n_d1, n]] * cross_z_d_theta[[i, n_z2, n_d2, n_theta]] + cross_comd_z_d[[i, n_z2, n_d2, n]] * cross_z_d_theta[[i, n_z1, n_d1, n_theta]];
                            }

                            c = cross_z_d[[i, n_z1, n_d1]] * cross_comd_z_d_theta[[i, n_z2, n_d2, n_theta, 0]] + cross_z_d[[i, n_z2, n_d2]] * cross_comd_z_d_theta[[i, n_z1, n_d1, n_theta, 0]];
                            gradient[[0,index]] -= c;
                            gradient[[3*i+3,index]] += cross_sumd_z_d[[i, n_z1, n_d1, 0]] * cross_z_d_theta[[i, n_z2, n_d2, n_theta]] + cross_sumd_z_d[[i, n_z2, n_d2, 0]] * cross_z_d_theta[[i, n_z1, n_d1, n_theta]] + c;

                            c = cross_z_d[[i, n_z1, n_d1]] * cross_comd_z_d_theta[[i, n_z2, n_d2, n_theta, 1]] + cross_z_d[[i, n_z2, n_d2]] * cross_comd_z_d_theta[[i, n_z1, n_d1, n_theta, 1]];
                            gradient[[1,index]] -= c;
                            gradient[[3*i+4,index]] += cross_sumd_z_d[[i, n_z1, n_d1, 1]] * cross_z_d_theta[[i, n_z2, n_d2, n_theta]] + cross_sumd_z_d[[i, n_z2, n_d2, 1]] * cross_z_d_theta[[i, n_z1, n_d1, n_theta]] + c;

                            c = cross_z_d[[i, n_z1, n_d1]] * cross_comd_z_d_theta[[i, n_z2, n_d2, n_theta, 2]] + cross_z_d[[i, n_z2, n_d2]] * cross_comd_z_d_theta[[i, n_z1, n_d1, n_theta, 2]];
                            gradient[[2,index]] -= c;
                            gradient[[3*i+5,index]] += cross_sumd_z_d[[i, n_z1, n_d1, 2]] * cross_z_d_theta[[i, n_z2, n_d2, n_theta]] + cross_sumd_z_d[[i, n_z2, n_d2, 2]] * cross_z_d_theta[[i, n_z1, n_d1, n_theta]] + c;
                        }

                        index += 1;
                    }
                }
            }
        }
    }



    // Return the descriptor and gradient
    (descriptor, gradient)
}










/// Cutoff function for the descriptor
///
/// # Parameters
/// ```
/// r: the distance to the centered atom
/// rcut: the cutoff radius (e.g. 6 A = 11.338 bohr)
/// f: the returned function value
/// ```
fn fcut(r: f64, rcut: f64) -> f64
{
    if r < rcut
    {
        (1.0 - r / rcut).tanh().powi(3)
    }
    else
    {
        0.0
    }
}





/// Derivative of the cutoff function for the descriptor
///
/// # Parameters
/// ```
/// r: the distance to the centered atom
/// rcut: the cutoff radius (e.g. 6 A = 11.338 bohr)
/// df: the returned derivative of the cutoff function
/// ```
fn dfcut(r: f64, rcut: f64) -> f64
{
    if r < rcut
    {
        let x: f64 = 1.0 - r / rcut;
        - (3.0 / rcut) * (x.tanh() / x.cosh()).powi(2)
    }
    else
    {
        0.0
    }
}





/// Gaussian function for the descriptor
///
/// # Parameters
/// ```
/// x: the input independent variable
/// mu: the mean value of the Gaussian function
/// sigma: the standard deviation of the Gaussian function
/// g: the returned function value
/// ```
fn gaussian(x: f64, mu: f64, sigma: f64) -> f64
{
   ( -(x-mu).powi(2) / (2.0*sigma*sigma) ).exp()
}





/// Derivative of the gaussian function for the descriptor
///
/// # Parameters
/// ```
/// x: the input independent variable
/// mu: the mean value of the Gaussian function
/// sigma: the standard deviation of the Gaussian function
/// f: the input gaussian function value
/// dg: the returned derivative of the function value
/// ```
fn dgaussian(x: f64, mu: f64, sigma: f64, f: f64) -> f64
{
    f * (mu-x) / (sigma*sigma)
}
//fn dgaussian(x: f64, mu: f64, sigma: f64) -> f64
//{
//   ( -(x-mu).powi(2) / (2.0*sigma*sigma) ).exp() * (mu-x) / (sigma*sigma)
//}










lazy_static!
{
    // Each atom is characterized by a feature value in the range of [0, 1]
    static ref ATOMIC_NUMBER_TO_ATOMIC_FEATURE: HashMap<usize, f64> =
    {
        let mut atomic_number_to_atomic_feature: HashMap<usize, f64> = HashMap::new();

        atomic_number_to_atomic_feature.insert(1, 0.000);               // H
        atomic_number_to_atomic_feature.insert(6, 1.0/3.0);             // C
        atomic_number_to_atomic_feature.insert(7, 2.0/3.0);             // N
        atomic_number_to_atomic_feature.insert(8, 1.000);               // O
        atomic_number_to_atomic_feature.insert(16, 0.500);              // S
        
        atomic_number_to_atomic_feature.insert(5, 0.036);               // B
        atomic_number_to_atomic_feature.insert(9, 0.072);               // F
        atomic_number_to_atomic_feature.insert(11, 0.108);              // Na
        atomic_number_to_atomic_feature.insert(12, 0.144);              // Mg
        atomic_number_to_atomic_feature.insert(14, 0.180);              // Si
        atomic_number_to_atomic_feature.insert(15, 0.216);              // P
        atomic_number_to_atomic_feature.insert(17, 0.252);              // Cl
        atomic_number_to_atomic_feature.insert(19, 0.288);              // K
        atomic_number_to_atomic_feature.insert(20, 0.369);              // Ca
        atomic_number_to_atomic_feature.insert(23, 0.405);              // V
        atomic_number_to_atomic_feature.insert(24, 0.441);              // Cr
        atomic_number_to_atomic_feature.insert(25, 0.477);              // Mn
        atomic_number_to_atomic_feature.insert(26, 0.523);              // Fe
        atomic_number_to_atomic_feature.insert(27, 0.559);              // Co
        atomic_number_to_atomic_feature.insert(28, 0.595);              // Ni
        atomic_number_to_atomic_feature.insert(29, 0.631);              // Cu
        atomic_number_to_atomic_feature.insert(30, 0.712);              // Zn
        atomic_number_to_atomic_feature.insert(33, 0.748);              // As
        atomic_number_to_atomic_feature.insert(34, 0.784);              // Se
        atomic_number_to_atomic_feature.insert(35, 0.820);              // Br
        atomic_number_to_atomic_feature.insert(42, 0.856);              // Mo
        atomic_number_to_atomic_feature.insert(48, 0.892);              // Cd
        atomic_number_to_atomic_feature.insert(50, 0.928);              // Sn
        atomic_number_to_atomic_feature.insert(53, 0.964);              // I
        
        atomic_number_to_atomic_feature.insert(2, 0.009);               // He
        atomic_number_to_atomic_feature.insert(3, 0.018);               // Li
        atomic_number_to_atomic_feature.insert(4, 0.027);               // Be
        atomic_number_to_atomic_feature.insert(10, 0.045);              // Ne
        atomic_number_to_atomic_feature.insert(13, 0.054);              // Al
        atomic_number_to_atomic_feature.insert(18, 0.063);              // Ar
        atomic_number_to_atomic_feature.insert(21, 0.081);              // Sc
        atomic_number_to_atomic_feature.insert(22, 0.090);              // Ti
        atomic_number_to_atomic_feature.insert(31, 0.099);              // Ga
        atomic_number_to_atomic_feature.insert(32, 0.117);              // Ge
        atomic_number_to_atomic_feature.insert(36, 0.126);              // Kr
        atomic_number_to_atomic_feature.insert(37, 0.135);              // Rb
        atomic_number_to_atomic_feature.insert(38, 0.153);              // Sr
        atomic_number_to_atomic_feature.insert(39, 0.162);              // Y
        atomic_number_to_atomic_feature.insert(40, 0.171);              // Zr
        atomic_number_to_atomic_feature.insert(41, 0.189);              // Nb
        atomic_number_to_atomic_feature.insert(43, 0.198);              // Tc
        atomic_number_to_atomic_feature.insert(44, 0.207);              // Ru
        atomic_number_to_atomic_feature.insert(45, 0.225);              // Rh
        atomic_number_to_atomic_feature.insert(46, 0.234);              // Pd
        atomic_number_to_atomic_feature.insert(47, 0.243);              // Ag
        atomic_number_to_atomic_feature.insert(49, 0.261);              // In
        atomic_number_to_atomic_feature.insert(51, 0.270);              // Sb
        atomic_number_to_atomic_feature.insert(52, 0.279);              // Te
        atomic_number_to_atomic_feature.insert(54, 0.297);              // Xe
        atomic_number_to_atomic_feature.insert(55, 0.306);              // Cs
        atomic_number_to_atomic_feature.insert(56, 0.315);              // Ba
        atomic_number_to_atomic_feature.insert(57, 0.324);              // La
        atomic_number_to_atomic_feature.insert(58, 0.342);              // Ce
        atomic_number_to_atomic_feature.insert(59, 0.351);              // Pr
        atomic_number_to_atomic_feature.insert(60, 0.360);              // Nd
        atomic_number_to_atomic_feature.insert(61, 0.378);              // Pm
        atomic_number_to_atomic_feature.insert(62, 0.387);              // Sm
        atomic_number_to_atomic_feature.insert(63, 0.396);              // Eu
        atomic_number_to_atomic_feature.insert(64, 0.414);              // Gd
        atomic_number_to_atomic_feature.insert(65, 0.423);              // Tb
        atomic_number_to_atomic_feature.insert(66, 0.432);              // Dy
        atomic_number_to_atomic_feature.insert(67, 0.450);              // Ho
        atomic_number_to_atomic_feature.insert(68, 0.459);              // Er
        atomic_number_to_atomic_feature.insert(69, 0.468);              // Tm
        atomic_number_to_atomic_feature.insert(70, 0.486);              // Yb
        atomic_number_to_atomic_feature.insert(71, 0.514);              // Lu
        atomic_number_to_atomic_feature.insert(72, 0.532);              // Hf
        atomic_number_to_atomic_feature.insert(73, 0.541);              // Ta
        atomic_number_to_atomic_feature.insert(74, 0.550);              // W
        atomic_number_to_atomic_feature.insert(75, 0.568);              // Re
        atomic_number_to_atomic_feature.insert(76, 0.577);              // Os
        atomic_number_to_atomic_feature.insert(77, 0.586);              // Ir
        atomic_number_to_atomic_feature.insert(78, 0.604);              // Pt
        atomic_number_to_atomic_feature.insert(79, 0.613);              // Au
        atomic_number_to_atomic_feature.insert(80, 0.622);              // Hg
        atomic_number_to_atomic_feature.insert(81, 0.640);              // Tl
        atomic_number_to_atomic_feature.insert(82, 0.649);              // Pb
        atomic_number_to_atomic_feature.insert(83, 0.658);              // Bi
        atomic_number_to_atomic_feature.insert(84, 0.676);              // Po
        atomic_number_to_atomic_feature.insert(85, 0.685);              // At
        atomic_number_to_atomic_feature.insert(86, 0.694);              // Rn
        atomic_number_to_atomic_feature.insert(87, 0.703);              // Fr
        atomic_number_to_atomic_feature.insert(88, 0.721);              // Ra
        atomic_number_to_atomic_feature.insert(89, 0.730);              // Ac
        atomic_number_to_atomic_feature.insert(90, 0.739);              // Th
        atomic_number_to_atomic_feature.insert(91, 0.757);              // Pa
        atomic_number_to_atomic_feature.insert(92, 0.766);              // U
        atomic_number_to_atomic_feature.insert(93, 0.775);              // Np
        atomic_number_to_atomic_feature.insert(94, 0.793);              // Pu
        atomic_number_to_atomic_feature.insert(95, 0.802);              // Am
        atomic_number_to_atomic_feature.insert(96, 0.811);              // Cm
        atomic_number_to_atomic_feature.insert(97, 0.829);              // Bk
        atomic_number_to_atomic_feature.insert(98, 0.838);              // Cf
        atomic_number_to_atomic_feature.insert(99, 0.847);              // Es
        atomic_number_to_atomic_feature.insert(100, 0.865);             // Fm
        atomic_number_to_atomic_feature.insert(101, 0.874);             // Md
        atomic_number_to_atomic_feature.insert(102, 0.883);             // No
        atomic_number_to_atomic_feature.insert(103, 0.901);             // Lr
        atomic_number_to_atomic_feature.insert(104, 0.910);             // Rf
        atomic_number_to_atomic_feature.insert(105, 0.919);             // Db
        atomic_number_to_atomic_feature.insert(106, 0.937);             // Sg
        atomic_number_to_atomic_feature.insert(107, 0.946);             // Bh
        atomic_number_to_atomic_feature.insert(108, 0.955);             // Hs
        atomic_number_to_atomic_feature.insert(109, 0.973);             // Mt
        atomic_number_to_atomic_feature.insert(110, 0.982);             // Ds
        atomic_number_to_atomic_feature.insert(111, 0.991);             // Rg

        atomic_number_to_atomic_feature
    };





    static ref GAUSSIAN_Z_MUZ: Array2<f64> =
    {
        let n_z: usize = Element::get_element_number();
        let n_mu_z: usize = MU_Z.len();
        let mut gaussian_z_muz: Array2<f64> = Array2::zeros((n_z, n_mu_z));

        let mut atomic_feature: f64;
        for i in 0..n_z                // For each element
        {
            atomic_feature = *ATOMIC_NUMBER_TO_ATOMIC_FEATURE.get(&(i+1)).expect(&error_none_value("atomic_feature"));
            for j in 0..n_mu_z                // For each sampling in MU_Z
            {
                gaussian_z_muz[[i,j]] = gaussian(atomic_feature, MU_Z[j], SIGMA_Z);
            }
        }

        gaussian_z_muz
    };
}










