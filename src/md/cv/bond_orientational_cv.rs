//! About the bond orientational order parameter introduced by Steinhardt et al., as collective variable (CV) for enhanced sampling.

use crate::common::constants::*;
use crate::common::error::*;
use crate::common::traits::ColVar;
use crate::io::input::Para;
use crate::pes_exploration::system::System;
use crate::md::eabf_atf::BiasingForce1D;
use crate::md::cv::smoothing_function::QuinticSmooth;
use crate::math::spherical_harmonics::dsh_dxyz;
use ndarray::Array2;
use num::Complex;
use sphrs::{Coordinates, sh};





/// The basic structure containing the initial state and final state for bond orientational CV.
///
/// # Fields
/// ```
/// l: degree of the bond orientational order parameter
/// r_in: given distance for inner cutoff of quintic smoothing function (Unit: bohr)
/// r_cut: given distance for outer cutoff of quintic smoothing function (Unit: bohr)
/// is: initial state
/// fs: final state
/// ```
#[derive(Debug)]
pub struct BondOrientationalCV
{
    pub l: i64,
    pub r_in: f64,
    pub r_cut: f64,
    pub is: System,
    pub fs: System,
}





impl BondOrientationalCV
{
    /// Construct a BondOrientationalCV directly
    ///
    /// # Parameters
    /// ```
    /// l: degree of the bond orientational order parameter
    /// r_in: given distance for inner cutoff of quintic smoothing function (Unit: bohr)
    /// r_cut: given distance for outer cutoff of quintic smoothing function (Unit: bohr)
    /// is: initial state
    /// fs: final state
    /// ```
    ///
    /// # Examples
    /// ```
    /// let is: System = System::read_xyz("IS.xyz");
    /// let fs: System = System::read_xyz("FS.xyz");
    /// let bond_orientational_cv: BondOrientationalCV = BondOrientationalCV::new(6, r_in, r_cut, is, fs);
    /// ```
    pub fn new(l: i64, r_in: f64, r_cut: f64, is: System, fs: System) -> Self
    {
        assert!(r_in.abs() < r_cut.abs());
        if is.natom != fs.natom || is.atom_type != fs.atom_type
        {
            panic!("{}", error_general_cv_mismatched());                // IS and FS should have the same number of atoms and the same atomic types
        }

        BondOrientationalCV
        {
            l: l.abs(),
            r_in: r_in.abs(),
            r_cut: r_cut.abs(),
            is,
            fs,
        }
    }
}










impl ColVar for BondOrientationalCV
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
    /// let s: System = bond_orientational_cv.get_ini_str();
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
    /// let biasing_force_1d: BiasingForce1D = bond_orientational_cv.initialize_biasing_force(&para);
    /// ```
    fn initialize_biasing_force(&self, para: &Para) -> BiasingForce1D
    {
        // Read in the needed parameters from para
        let cv_bin: f64 = para.md_para.eabf_atf_para.cv_bin * ANGSTROM_TO_BOHR;                // reference CV bin in eABF-ATF (Unit: bohr)
        let cv_ext: f64 = para.md_para.eabf_atf_para.cv_ext * ANGSTROM_TO_BOHR;                // extended CV in bond orientational CV (Unit: bohr)

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





    /// Based on degree of the bond orientational order parameter, calculate the Bond Orientational CV for a given structure.
    ///
    /// # Parameters
    /// ```
    /// s: the given structure
    /// cv: the output value of Bond Orientational CV
    /// ```
    ///
    /// # Examples
    /// ```
    /// let cv: f64 = bond_orientational_cv.get_cv(&s);
    /// ```
    fn get_cv(&self, s: &System) -> f64
    {
        // Construct the quintic smoothing function
        let quintic_smooth: QuinticSmooth = QuinticSmooth::new(self.r_in, self.r_cut);


        // Define the intermediate variables
        let num_order: usize = 2 * (self.l as usize) + 1;                // Number of order (2l+1)
        let mut index: usize;                // Index of the order

        let mut delta_x: f64;                // Delta x of atom i and j
        let mut delta_y: f64;                // Delta y of atom i and j
        let mut delta_z: f64;                // Delta z of atom i and j
        let mut dist: f64;                // Distance between atom i and j
        let mut bond: Coordinates<f64>;                // Bond vector from atom j to atom i

        let mut weight: f64;                // Weight of each bond based on interatomic distance

        let mut y_lm: Complex<f64>;                // Complex spherical harmonics of bond vector from atom j to atom i


        // Calculate numerator and denominator of Q_lm, along with the derivative with respect to atomic coordinates
        let mut u: Vec< Complex<f64> > = vec![Complex::new(0.0, 0.0); num_order];                // Numerator of Q_lm (2l+1)
        let mut v: f64 = 0.0;                // Denominator of Q_lm
        for i in 0..(s.natom-1)
        {
            for j in (i+1)..s.natom
            {
                delta_x = s.coord[[i, 0]] - s.coord[[j, 0]];
                delta_y = s.coord[[i, 1]] - s.coord[[j, 1]];
                delta_z = s.coord[[i, 2]] - s.coord[[j, 2]];
                dist = (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z).sqrt();
                bond = Coordinates::cartesian(delta_x, delta_y, delta_z);

                weight = quintic_smooth.f(dist);

                v += weight;

                for m in (-self.l)..(self.l+1)
                {
                    index = (m + self.l) as usize;

                    y_lm = sh(self.l, m, &bond);

                    u[index] += weight * y_lm;
                }
            }
        }


        // Calculate Q_lm, along with the derivative with respect to atomic coordinates
        let mut q_lm: Vec< Complex<f64> > = Vec::with_capacity(num_order);                // Q_lm (2l+1)
        for k in 0..num_order
        {
            q_lm.push(u[k] / v);
        }


        // Calculate Q_l, along with the derivative with respect to atomic coordinates
        let mut q_l: f64 = 0.0;
        for k in 0..num_order
        {
            q_l += q_lm[k].norm_sqr();
        }
        q_l = (4.0 * PI * q_l / num_order as f64).sqrt();


        q_l
    }





    /// Based on degree of the bond orientational order parameter, calculate the Bond Orientational CV and its gradients for a given structure.
    ///
    /// # Parameters
    /// ```
    /// s: the given structure
    /// cv: the output value of Bond Orientational CV
    /// grads: the output gradients of Bond Orientational CV with respect to structural coordinates
    /// ```
    ///
    /// # Examples
    /// ```
    /// let (cv, grads): (f64, Array2<f64>) = bond_orientational_cv.get_cv_grads(&s);
    /// ```
    fn get_cv_grads(&self, s: &System) -> (f64, Array2<f64>)
    {
        // Construct the quintic smoothing function
        let quintic_smooth: QuinticSmooth = QuinticSmooth::new(self.r_in, self.r_cut);


        // Define the intermediate variables
        let num_order: usize = 2 * (self.l as usize) + 1;                // Number of order (2l+1)
        let mut index: usize;                // Index of the order

        let mut delta_x: f64;                // Delta x of atom i and j
        let mut delta_y: f64;                // Delta y of atom i and j
        let mut delta_z: f64;                // Delta z of atom i and j
        let mut dist: f64;                // Distance between atom i and j
        let mut bond: Coordinates<f64>;                // Bond vector from atom j to atom i

        let mut weight: f64;                // Weight of each bond based on interatomic distance
        let mut dweight: f64;                // Derivative of weight of each bond with respect to interatomic distance

        let mut y_lm: Complex<f64>;                // Complex spherical harmonics of bond vector from atom j to atom i
        let mut dy_lm_dx: Complex<f64>;                // Derivative of complex spherical harmonics of bond vector with respect to x
        let mut dy_lm_dy: Complex<f64>;                // Derivative of complex spherical harmonics of bond vector with respect to y
        let mut dy_lm_dz: Complex<f64>;                // Derivative of complex spherical harmonics of bond vector with respect to z


        // Calculate numerator and denominator of Q_lm, along with the derivative with respect to atomic coordinates
        let mut u: Vec< Complex<f64> > = vec![Complex::new(0.0, 0.0); num_order];                // Numerator of Q_lm (2l+1)
        let mut v: f64 = 0.0;                // Denominator of Q_lm
        let mut du: Vec< Array2<Complex<f64>> > = vec![Array2::from_elem((s.natom, 3), Complex::new(0.0, 0.0)); num_order];                // Derivative of numerator of Q_lm (2l+1)
        let mut dv: Array2<Complex<f64>> = Array2::from_elem((s.natom, 3), Complex::new(0.0, 0.0));                //  Derivative of denominator of Q_lm
        for i in 0..(s.natom-1)
        {
            for j in (i+1)..s.natom
            {
                delta_x = s.coord[[i, 0]] - s.coord[[j, 0]];
                delta_y = s.coord[[i, 1]] - s.coord[[j, 1]];
                delta_z = s.coord[[i, 2]] - s.coord[[j, 2]];
                dist = (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z).sqrt();
                bond = Coordinates::cartesian(delta_x, delta_y, delta_z);

                weight = quintic_smooth.f(dist);
                dweight = quintic_smooth.df(dist);

                v += weight;
                dv[[i, 0]] += dweight * delta_x / dist;
                dv[[i, 1]] += dweight * delta_y / dist;
                dv[[i, 2]] += dweight * delta_z / dist;
                dv[[j, 0]] -= dweight * delta_x / dist;
                dv[[j, 1]] -= dweight * delta_y / dist;
                dv[[j, 2]] -= dweight * delta_z / dist;

                for m in (-self.l)..(self.l+1)
                {
                    index = (m + self.l) as usize;

                    y_lm = sh(self.l, m, &bond);
                    (dy_lm_dx, dy_lm_dy, dy_lm_dz) = dsh_dxyz(self.l, m, &bond);

                    u[index] += weight * y_lm;
                    du[index][[i, 0]] += y_lm * (dweight * delta_x / dist) + weight * dy_lm_dx;
                    du[index][[i, 1]] += y_lm * (dweight * delta_y / dist) + weight * dy_lm_dy;
                    du[index][[i, 2]] += y_lm * (dweight * delta_z / dist) + weight * dy_lm_dz;
                    du[index][[j, 0]] -= y_lm * (dweight * delta_x / dist) + weight * dy_lm_dx;
                    du[index][[j, 1]] -= y_lm * (dweight * delta_y / dist) + weight * dy_lm_dy;
                    du[index][[j, 2]] -= y_lm * (dweight * delta_z / dist) + weight * dy_lm_dz;
                }
            }
        }


        // Calculate Q_lm, along with the derivative with respect to atomic coordinates
        let mut q_lm: Vec< Complex<f64> > = Vec::with_capacity(num_order);                // Q_lm (2l+1)
        let mut dq_lm: Vec< Array2<Complex<f64>> > = Vec::with_capacity(num_order);                // Derivative of Q_lm (2l+1)
        for k in 0..num_order
        {
            q_lm.push(u[k] / v);
            dq_lm.push( (Complex::new(1.0, 0.0) / v) * &du[k] - (u[k] / v / v) * &dv );
        }


        // Calculate Q_l, along with the derivative with respect to atomic coordinates
        let mut q_l: f64 = 0.0;
        let mut dq_l: Array2<f64> = Array2::zeros((s.natom, 3));
        for k in 0..num_order
        {
            q_l += q_lm[k].norm_sqr();
            dq_l += &(q_lm[k].conj() * &dq_lm[k]).map(|x| x.re * 2.0);
        }
        q_l = (4.0 * PI * q_l / num_order as f64).sqrt();
        dq_l *= 2.0 * PI / (q_l * num_order as f64);


        (q_l, dq_l)
    }
}










