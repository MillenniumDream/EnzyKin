//! A specially designed linear potential for two-body system

use crate::common::traits::PES;
use crate::pes_exploration::system::System;
use ndarray::{Array2, array};





const EV: f64 = 0.0367493;

/// The structure for specially designed linear potential
///
/// # Fields
/// ```
/// ```
pub struct LinearPot
{
}





impl LinearPot
{
    /// Construct a new LinearPot
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// let linear_pot: LinearPot = LinearPot::new();
    /// ```
    pub fn new() -> Self
    {
        LinearPot
        {
        }
    }
}





impl PES for LinearPot
{
    /// Input a two-atom system, calculate and output its linear potential energy
    ///
    /// # Parameters
    /// ```
    /// s: the input system
    /// pot: the output potential energy (Unit: Hartree)
    /// ```
    fn get_energy(&self, s: &System) -> f64
    {
        assert!(s.natom == 2);

        let delta_x = s.coord[[0, 0]] - s.coord[[1, 0]];
        let delta_y = s.coord[[0, 1]] - s.coord[[1, 1]];
        let delta_z = s.coord[[0, 2]] - s.coord[[1, 2]];
        let dist = ( delta_x * delta_x + delta_y * delta_y + delta_z * delta_z ).sqrt();

        if dist < 2.0
        {
            (-dist + 2.0) * EV
        }
        else if dist < 3.0
        {
            (dist - 2.0) * EV
        }
        else if dist < 4.0
        {
            (-dist + 4.0) * EV
        }
        else
        {
            (dist - 4.0) * EV
        }
    }





    /// Input a two-atom system, calculate and output its linear potential energy and atomic forces
    ///
    /// # Parameters
    /// ```
    /// s: the input system
    /// pot: the output potential energy (Unit: Hartree)
    /// force: the output atomic forces (2 * 3, Unit: A.U.)
    /// ```
    fn get_energy_force(&self, s: &System) -> (f64, Array2<f64>)
    {
        assert!(s.natom == 2);

        let delta_x = s.coord[[0, 0]] - s.coord[[1, 0]];
        let delta_y = s.coord[[0, 1]] - s.coord[[1, 1]];
        let delta_z = s.coord[[0, 2]] - s.coord[[1, 2]];
        let dist = ( delta_x * delta_x + delta_y * delta_y + delta_z * delta_z ).sqrt();

        let force: Array2<f64> = array!
        [
            [ EV * delta_x / dist,  EV * delta_y / dist,  EV * delta_z / dist],
            [-EV * delta_x / dist, -EV * delta_y / dist, -EV * delta_z / dist],
        ];

        if dist < 2.0
        {
            ( (-dist + 2.0) * EV, force )
        }
        else if dist < 3.0
        {
            ( (dist - 2.0) * EV, -force )
        }
        else if dist < 4.0
        {
            ( (-dist + 4.0) * EV, force )
        }
        else
        {
            ( (dist - 4.0) * EV, -force )
        }
    }
}










