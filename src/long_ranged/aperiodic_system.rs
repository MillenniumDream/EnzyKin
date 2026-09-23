//! Long-ranged interaction calculations for aperiodic system
use crate::nn::protein::ProteinSystem;
use ndarray::{Array2};





/// Calculate the long-ranged interaction energy for aperiodic system (serial version)
///
/// # Parameters
/// ```
/// s: the input protein system
/// pot_es: the output total electrostatic potential energy
/// ```
///
/// # Examples
/// ```
/// ```
pub fn get_aperiodic_system_es_energy_serial(s: &ProteinSystem) -> f64
{
//    let mut vdw_pot: f64 = 0.0;
    let mut pot_es: f64 = 0.0;

    // Define the intermediate variables
    let mut r: f64;             // The distance between atoms i and j
//    let mut epsilon: f64;               // The cross vdW parameter (well depth) of atoms i and j
//    let mut rmin_r_6_power: f64;              // (Rmin / r)^6
    for i in 0..s.natom
    {
        for j in (i+1)..s.natom
        {
            r = ( (s.coord[[i,0]] - s.coord[[j,0]]).powi(2) + (s.coord[[i,1]] - s.coord[[j,1]]).powi(2) + (s.coord[[i,2]] - s.coord[[j,2]]).powi(2) ).sqrt();

            // Obtain the long-ranged interaction parameters of atoms i and j
            match (&s.long_ranged_para[i], &s.long_ranged_para[j])
            {
                // If both atoms i and j have fixed long-ranged interaction parameters, obtain them directly
                (Some(atom_para_au_i), Some(atom_para_au_j)) =>
                {
/*
                    // Get the cross long-ranged interaction parameters
                    epsilon = (atom_para_au_i.epsilon * atom_para_au_j.epsilon).sqrt();
                    rmin_r_6_power = ( (atom_para_au_i.vdw_r + atom_para_au_j.vdw_r) / r ).powi(6);

                    // Calculate the vdW potential
                    vdw_pot += epsilon * rmin_r_6_power * (rmin_r_6_power - 2.0);
*/

                    // Calculate the electrostatic potential and force
                    pot_es += atom_para_au_i.charge * atom_para_au_j.charge / r;
                },

                (Some(_atom_para_au_i), None) =>
                {
                    panic!("\n\n\n Atom j does not have long-ranged interaction parameters, check it. \n\n\n");
                },

                (None, Some(_atom_para_au_j)) =>
                {
                    panic!("\n\n\n Atom i does not have long-ranged interaction parameters, check it. \n\n\n");
                },

                (None, None) =>
                {
                    panic!("\n\n\n Atoms i and j do not have long-ranged interaction parameters, check it. \n\n\n");
                },
            }
        }
    }

    return pot_es
}





/// Calculate the long-ranged interaction energy and force for aperiodic system (serial version)
///
/// # Parameters
/// ```
/// s: the input protein system
/// pot_es: the output total electrostatic potential energy
/// force_es: the output total electrostatic force
/// ```
///
/// # Examples
/// ```
/// ```
pub fn get_aperiodic_system_es_energy_force_serial(s: &ProteinSystem) -> (f64, Array2<f64>)
{
//    let mut vdw_pot: f64 = 0.0;
//    let mut vdw_force: Array2<f64> = Array2::zeros(s.coord.raw_dim());
    let mut pot_es: f64 = 0.0;
    let mut force_es: Array2<f64> = Array2::zeros(s.coord.raw_dim());

    // Define the intermediate variables
    let mut r: f64;             // The distance between atoms i and j
    let mut rec_squ_r: f64;             // 1 / r^2
//    let mut epsilon: f64;               // The cross vdW parameter (well depth) of atoms i and j
//    let mut rmin_r_6_power: f64;              // (Rmin / r)^6
    let mut f: f64;             // An intermediate variable for force calculations
    let mut f_i: f64;               // An intermediate variable for force calculations
    for i in 0..s.natom
    {
        for j in (i+1)..s.natom
        {
            r = ( (s.coord[[i,0]] - s.coord[[j,0]]).powi(2) + (s.coord[[i,1]] - s.coord[[j,1]]).powi(2) + (s.coord[[i,2]] - s.coord[[j,2]]).powi(2) ).sqrt();
            rec_squ_r = 1.0 / (r*r);

            // Obtain the long-ranged interaction parameters of atoms i and j
            match (&s.long_ranged_para[i], &s.long_ranged_para[j])
            {
                // If both atoms i and j have fixed long-ranged interaction parameters, obtain them directly
                (Some(atom_para_au_i), Some(atom_para_au_j)) =>
                {
/*
                    // Get the cross long-ranged interaction parameters
                    epsilon = (atom_para_au_i.epsilon * atom_para_au_j.epsilon).sqrt();
                    rmin_r_6_power = ( (atom_para_au_i.vdw_r + atom_para_au_j.vdw_r) / r ).powi(6);

                    // Calculate the vdW potential and force
                    vdw_pot += epsilon * rmin_r_6_power * (rmin_r_6_power - 2.0);
                    f = 12.0 * epsilon * rec_squ_r * rmin_r_6_power * (rmin_r_6_power - 1.0);
                    for k in 0..3
                    {
                        f_i = (s.coord[[i,k]] - s.coord[[j,k]]) * f;
                        vdw_force[[i,k]] += f_i;
                        vdw_force[[j,k]] -= f_i;
                    }
*/

                    // Calculate the electrostatic potential and force
                    f = atom_para_au_i.charge * atom_para_au_j.charge / r;
                    pot_es += f;
                    f *= rec_squ_r;
                    for k in 0..3
                    {
                        f_i = (s.coord[[i,k]] - s.coord[[j,k]]) * f;
                        force_es[[i,k]] += f_i;
                        force_es[[j,k]] -= f_i;
                    }

                },

                (Some(_atom_para_au_i), None) =>
                {
                    panic!("\n\n\n Atom j does not have long-ranged interaction parameters, check it. \n\n\n");
                },

                (None, Some(_atom_para_au_j)) =>
                {
                    panic!("\n\n\n Atom i does not have long-ranged interaction parameters, check it. \n\n\n");
                },

                (None, None) =>
                {
                    panic!("\n\n\n Atoms i and j do not have long-ranged interaction parameters, check it. \n\n\n");
                },
            }
        }
    }

    return (pot_es, force_es)
}










