//! Lennard-Jones Potential
use crate::common::constants::*;
use crate::common::traits::PES;
use crate::pes_exploration::system::System;
use ndarray::{Array1, Array2, s};
use lazy_static::lazy_static;





// Adopt the parameters of Ar
pub const T_CHAR: f64 = 119.8;                // Characteristic temperature (Unit: K)
pub const EPSILON: f64 = BOLTZMANN * T_CHAR * JOULE_TO_HARTREE;                // Well depth of LJ potential (Unit: Hartree, Value: 0.0003793844781257473)
pub const SIGMA: f64 = 3.405 * ANGSTROM_TO_BOHR;                // Reduced distance unit of LJ potential (Unit: bohr)
lazy_static!
{
    // Reduced time unit of LJ potential (2156.3459819054438 fs)
    pub static ref TAU: f64 =
    {
        let ar_mass: f64 = Element::Ar.get_atomic_mass();                // (Unit: A.U.)
        (ar_mass * SIGMA * SIGMA / EPSILON).sqrt() * AU_TO_FEMTOSECOND
    };
}





/// The structure for LJ cluster calculation
///
/// # Fields
/// ```
/// attraction_prefactor: 4 * epsilon * sigma^6 (Unit: A.U.)
/// sigma_pow6: sigma^6 (Unit: A.U.)
/// ```
pub struct LJPot
{
    pub attraction_prefactor: f64,
    pub sigma_pow6: f64,
}





impl LJPot
{
    /// Construct a new LJPot with the parameters of Ar
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// let lj_pot: LJPot = LJPot::new();
    /// ```
    pub fn new() -> Self
    {
        LJPot
        {
            attraction_prefactor: 4.0 * EPSILON * SIGMA.powi(6),
            sigma_pow6: SIGMA.powi(6),
        }
    }
}





impl PES for LJPot
{
    /// Input a cluster, calculate and output its potential energy based on LJ model
    ///
    /// # Parameters
    /// ```
    /// s: the input cluster
    /// pot_lj: the output LJ potential energy (Unit: Hartree)
    /// ```
    fn get_energy(&self, s: &System) -> f64
    {
        let mut pot_lj: f64 = 0.0;

        let mut delta_r: Array1<f64>;
        let mut r_pow2: f64;
        let mut r_pow6: f64;
        let mut r_pow12: f64;
        for i in 0..(s.natom-1)
        {
            for j in (i+1)..s.natom
            {
                delta_r = &s.coord.slice(s![i, ..]) - &s.coord.slice(s![j, ..]);
                r_pow2 = delta_r[0] * delta_r[0] + delta_r[1] * delta_r[1] + delta_r[2] * delta_r[2];
                r_pow6 = r_pow2.powi(3);
                r_pow12 = r_pow6 * r_pow6;

                pot_lj += self.attraction_prefactor * (self.sigma_pow6 / r_pow12 - 1.0 / r_pow6);
            }
        }

        pot_lj
    }





    /// Input a cluster, calculate and output its potential energy and atomic forces based on LJ model
    ///
    /// # Parameters
    /// ```
    /// s: the input cluster
    /// pot_lj: the output LJ potential energy (Unit: Hartree)
    /// force_lj: the output LJ atomic forces (s.natom * 3, Unit: A.U.)
    /// ```
    fn get_energy_force(&self, s: &System) -> (f64, Array2<f64>)
    {
        let mut pot_lj: f64 = 0.0;
        let mut force_lj: Array2<f64> = Array2::zeros(s.coord.raw_dim());

        let mut delta_r: Array1<f64>;
        let mut r_pow2: f64;
        let mut r_pow6: f64;
        let mut r_pow12: f64;
        for i in 0..(s.natom-1)
        {
            for j in (i+1)..s.natom
            {
                delta_r = &s.coord.slice(s![i, ..]) - &s.coord.slice(s![j, ..]);
                r_pow2 = delta_r[0] * delta_r[0] + delta_r[1] * delta_r[1] + delta_r[2] * delta_r[2];
                r_pow6 = r_pow2.powi(3);
                r_pow12 = r_pow6 * r_pow6;

                pot_lj += self.attraction_prefactor * (self.sigma_pow6 / r_pow12 - 1.0 / r_pow6);
                delta_r *= (self.attraction_prefactor / r_pow2) * (12.0 * self.sigma_pow6 / r_pow12 - 6.0 / r_pow6);
                force_lj[[i, 0]] += delta_r[0];
                force_lj[[i, 1]] += delta_r[1];
                force_lj[[i, 2]] += delta_r[2];
                force_lj[[j, 0]] -= delta_r[0];
                force_lj[[j, 1]] -= delta_r[1];
                force_lj[[j, 2]] -= delta_r[2];
            }
        }

        (pot_lj, force_lj)
    }
}










