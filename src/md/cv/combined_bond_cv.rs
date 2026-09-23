//! About the combined bond collective variable (CV) for enhanced sampling

use crate::common::constants::*;
use crate::common::traits::ColVar;
use crate::common::error::*;
use crate::io::input::Para;
use crate::pes_exploration::system::System;
use crate::md::eabf_atf::BiasingForce1D;
use ndarray::Array2;





/// The basic structure containing the initial state and final state for combined bond CV.
///
/// # Fields
/// ```
/// bonds: containing coefficient and indices of the two bonded atoms
/// is: initial state
/// fs: final state
/// ```
#[derive(Debug)]
pub struct CombinedBondCV
{
    pub bonds: Vec<(f64, usize, usize)>,
    pub is: System,
    pub fs: System,
}





impl CombinedBondCV
{
    /// Construct a CombinedBondCV directly
    ///
    /// # Parameters
    /// ```
    /// bonds: containing coefficient and indices of the two bonded atoms
    /// is: initial state
    /// fs: final state
    /// ```
    ///
    /// # Examples
    /// ```
    /// let is: System = System::read_xyz("IS.xyz");
    /// let fs: System = System::read_xyz("FS.xyz");
    /// let combined_bond_cv: CombinedBondCV = CombinedBondCV::new(vec![(1.0, 0, 1)], is, fs);
    /// ```
    pub fn new(bonds: Vec<(f64, usize, usize)>, is: System, fs: System) -> Self
    {
        if is.natom != fs.natom || is.atom_type != fs.atom_type
        {
            panic!("{}", error_general_cv_mismatched());                // IS and FS should have the same number of atoms and the same atomic types
        }
        for i in 0..bonds.len()
        {
            if bonds[i].1 >= is.natom || bonds[i].2 >= is.natom
            {
                 panic!("{}", error_combined_bond_cv());                // The index of atom i and j should be within the range
            }
        }

        CombinedBondCV
        {
            bonds,
            is,
            fs,
        }
    }
}










impl ColVar for CombinedBondCV
{
    /// Return the initial structure for free energy calculation.
    ///
    /// # Parameters
    /// ```
    /// s: the output initial structure
    /// ```
    ///
    /// # Examples
    /// ```
    /// let s: System = combined_bond_cv.get_ini_str();
    /// ```
    fn get_ini_str(&self) -> System
    {
        self.is.clone()
    }





    /// Return an initialized BiasingForce1D for eABF-ATF free energy calculation.
    ///
    /// # Parameters
    /// ```
    /// para: the input parameters for construction
    /// ```
    ///
    /// # Examples
    /// ```
    /// let biasing_force_1d: BiasingForce1D = combined_bond_cv.initialize_biasing_force(&para);
    /// ```
    fn initialize_biasing_force(&self, para: &Para) -> BiasingForce1D
    {
        // Read in the needed parameters from para
        let cv_bin: f64 = para.md_para.eabf_atf_para.cv_bin * ANGSTROM_TO_BOHR;                // reference CV bin in eABF-ATF (Unit: bohr)
        let cv_ext: f64 = para.md_para.eabf_atf_para.cv_ext * ANGSTROM_TO_BOHR;                // extended CV in Combined Bond CV (Unit: bohr)

        // Get the minimum and maximum CV value
        let cv_is: f64 = self.get_cv(&self.is);                // Unit: bohr
        let cv_fs: f64 = self.get_cv(&self.fs);                // Unit: bohr
        let (cv_min, cv_max): (f64, f64) =
        if cv_is < cv_fs
        {
            (cv_is - cv_bin - cv_ext, cv_fs + cv_bin + cv_ext)
        }
        else if cv_fs < cv_is
        {
            (cv_fs - cv_bin - cv_ext, cv_is + cv_bin + cv_ext)
        }
        else
        {
            panic!("{}", error_general_cv_coincided());                // IS and FS should have distinct CV values
        };

        BiasingForce1D::from_range(&para, cv_min, cv_max)
    }





    /// Based on parameters of the bonds, calculate the combined bond collective variable (Combined Bond CV) for a given structure.
    ///
    /// # Parameters
    /// ```
    /// s: the given structure
    /// cv: the output value of Combined Bond CV
    /// ```
    ///
    /// # Examples
    /// ```
    /// let cv: f64 = combined_bond_cv.get_cv(&s);
    /// ```
    fn get_cv(&self, s: &System) -> f64
    {
        let mut delta_x: f64;
        let mut delta_y: f64;
        let mut delta_z: f64;
        let mut cv: f64 = 0.0;
        for i in 0..self.bonds.len()
        {
            delta_x = s.coord[[self.bonds[i].1, 0]] - s.coord[[self.bonds[i].2, 0]];
            delta_y = s.coord[[self.bonds[i].1, 1]] - s.coord[[self.bonds[i].2, 1]];
            delta_z = s.coord[[self.bonds[i].1, 2]] - s.coord[[self.bonds[i].2, 2]];
            cv += self.bonds[i].0 * ( delta_x * delta_x + delta_y * delta_y + delta_z * delta_z ).sqrt();
        }

        cv
    }





    /// Based on parameters of the bonds, calculate the combined bond collective variable (Combined Bond CV) and its gradients for a given structure.
    ///
    /// # Parameters
    /// ```
    /// s: the given structure
    /// cv: the output value of Combined Bond CV
    /// grads: the output gradients of Combined Bond CV with respect to structural coordinates
    /// ```
    ///
    /// # Examples
    /// ```
    /// let (cv, grads): (f64, Array2<f64>) = combined_bond_cv.get_cv_grads(&s);
    /// ```
    fn get_cv_grads(&self, s: &System) -> (f64, Array2<f64>)
    {
        let mut delta_x: f64;
        let mut delta_y: f64;
        let mut delta_z: f64;
        let mut dist: f64;
        let mut cv: f64 = 0.0;
        let mut grads: Array2<f64> = Array2::zeros(s.coord.raw_dim());
        for i in 0..self.bonds.len()
        {
            delta_x = s.coord[[self.bonds[i].1, 0]] - s.coord[[self.bonds[i].2, 0]];
            delta_y = s.coord[[self.bonds[i].1, 1]] - s.coord[[self.bonds[i].2, 1]];
            delta_z = s.coord[[self.bonds[i].1, 2]] - s.coord[[self.bonds[i].2, 2]];
            dist = ( delta_x * delta_x + delta_y * delta_y + delta_z * delta_z ).sqrt();

            cv += self.bonds[i].0 * dist;
            grads[[self.bonds[i].1, 0]] += self.bonds[i].0 * delta_x / dist;
            grads[[self.bonds[i].1, 1]] += self.bonds[i].0 * delta_y / dist;
            grads[[self.bonds[i].1, 2]] += self.bonds[i].0 * delta_z / dist;
            grads[[self.bonds[i].2, 0]] -= self.bonds[i].0 * delta_x / dist;
            grads[[self.bonds[i].2, 1]] -= self.bonds[i].0 * delta_y / dist;
            grads[[self.bonds[i].2, 2]] -= self.bonds[i].0 * delta_z / dist;
        }

        (cv, grads)
    }
}










