use crate::common::constants::*;
use crate::common::error::*;
use ndarray::{Array1, Array2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;










/// Initialize the velocity for a system according to the Maxwell-Boltzmann distribution.
/// Only remove the overall translational speed of the mass center
///
/// # Parameters
/// ```
/// temp: the target temperature of the system (K)
/// atom_mass: the input atomic masses of the system (a.u.)
/// vel: the output velocity for the system (a.u.)
/// ```
///
/// # Examples
/// ```
/// ```
pub fn vel_init(temp: f64, atom_mass: &Vec<f64>) -> Array2<f64>
{
    // Create the random velocities according to the Maxwell-Boltzmann distribution
    let mut std_dev: f64;
    let mut rand_vel: Array1<f64>;
    let mut vel: Array2<f64> = Array2::zeros((atom_mass.len(), 3));
    for i in 0..atom_mass.len()                // For each atom in the system
    {
        std_dev = (BOLTZMANN * temp * JOULE_TO_HARTREE / atom_mass[i]).sqrt();                // in a.u.
        rand_vel = Array1::random(3, Normal::new(0.0, std_dev).expect(&error_none_value("rand_vel")));
        vel[[i, 0]] = rand_vel[0];
        vel[[i, 1]] = rand_vel[1];
        vel[[i, 2]] = rand_vel[2];
    }

    // Remove the overall translational speed of the mass center
    let mut global_mass: f64 = 0.0;
    let mut global_vel: Array1<f64> = Array1::zeros(3);
    for i in 0..atom_mass.len()                // For each atom in the system
    {
        global_mass += atom_mass[i];
        global_vel += &(atom_mass[i] * &vel.slice(s![i, ..]));
    }
    global_vel /= global_mass;             // Global velocity
    for i in 0..atom_mass.len()                // For each atom in the system
    {
        vel[[i, 0]] -= global_vel[0];
        vel[[i, 1]] -= global_vel[1];
        vel[[i, 2]] -= global_vel[2];
    }

    // Calculate the transient temperature and scale the velocities to the target temperature
    let mut kin: f64 = 0.0;
    for i in 0..atom_mass.len()                // For each atom in the system
    {
        kin += atom_mass[i] * ( vel[[i, 0]].powi(2) + vel[[i, 1]].powi(2) + vel[[i, 2]].powi(2) );          // Hartree
    }
    let n_free: usize = 3 * atom_mass.len() - 3;                // Exclude the degrees of freedom for the global translation
    let temp_tran: f64 = kin * HARTREE_TO_JOULE / ( BOLTZMANN * n_free as f64 );               // K
    vel *= (temp / temp_tran).sqrt();

    vel
}










