use ndarray::{Array1, Array2, s};










/// Calculate the momentum and angular momentum for a cluster structure
///
/// # Parameters
/// ```
/// atom_mass: the input atomic masses of the cluster (a.u.)
/// coord: the input atomic coordinates of the cluster (bohr)
/// vel: the input atomic velocities of the cluster (a.u.)
/// momentum: the output momentum of the cluster (a.u.)
/// angular_momentum_orbital: the output orbital angular momentum, i.e., angular momentum of the center of mass with respect to origin (a.u.)
/// angular_momentum_cm: the output spin angular momentum, i.e., angular momentum of the atoms with respect to center of mass (a.u.)
/// ```
///
/// # Examples
/// ```
/// ```
pub fn get_linear_angular_momentum(atom_mass: &Vec<f64>, coord: &Array2<f64>, vel: &Array2<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>)
{
    // Obtain the coordinates and velocities for the center of mass of the cluster
    let mut global_mass: f64 = 0.0;                // The global mass of the cluster
    let mut global_coord: Array1<f64> = Array1::zeros(3);                // The coordinate of the center of mass of the cluster
    let mut global_vel: Array1<f64> = Array1::zeros(3);                // the velocity of the center of mass of the cluster
    for i in 0..atom_mass.len()
    {
        global_mass += atom_mass[i];
        global_coord += &(atom_mass[i] * &coord.slice(s![i, ..]));
        global_vel += &(atom_mass[i] * &vel.slice(s![i, ..]));
    }
    global_coord /= global_mass;
    global_vel /= global_mass;

    // Obtain the atomic coordinates and velocities with respect to the center of mass of the cluster
    let coord_cm: Array2<f64> = coord - &global_coord;
    let vel_cm: Array2<f64> = vel - &global_vel;

    // Calculate the momentum, orbital angular momentum and spin angular momentum for the cluster
    let momentum: Array1<f64> = global_mass * &global_vel;
    let mut angular_momentum_orbital: Array1<f64> = Array1::zeros(3);
    angular_momentum_orbital[0] = global_mass * (global_coord[1] * global_vel[2] - global_coord[2] * global_vel[1]);
    angular_momentum_orbital[1] = global_mass * (global_coord[2] * global_vel[0] - global_coord[0] * global_vel[2]);
    angular_momentum_orbital[2] = global_mass * (global_coord[0] * global_vel[1] - global_coord[1] * global_vel[0]);
    let mut angular_momentum_cm: Array1<f64> = Array1::zeros(3);
    for i in 0..atom_mass.len()
    {
        angular_momentum_cm[0] += atom_mass[i] * (coord_cm[[i, 1]] * vel_cm[[i, 2]] - coord_cm[[i, 2]] * vel_cm[[i, 1]]);
        angular_momentum_cm[1] += atom_mass[i] * (coord_cm[[i, 2]] * vel_cm[[i, 0]] - coord_cm[[i, 0]] * vel_cm[[i, 2]]);
        angular_momentum_cm[2] += atom_mass[i] * (coord_cm[[i, 0]] * vel_cm[[i, 1]] - coord_cm[[i, 1]] * vel_cm[[i, 0]]);
    }

    (momentum, angular_momentum_orbital, angular_momentum_cm)
}










