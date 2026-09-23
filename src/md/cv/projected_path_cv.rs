//! About the projected path collective variable (CV) for enhanced sampling

use std::path::Path;
use std::fmt::Debug;
use crate::common::constants::*;
use crate::common::traits::ColVar;
use crate::common::error::*;
use crate::io::input::Para;
use crate::pes_exploration::system::System;
use crate::pes_exploration::rtip::{rti_dist, rti_dist_coord, rti_dist_vec, rti_dist_rot_tran};
use crate::md::eabf_atf::BiasingForce1D;
use ndarray::{Array1, Array2};





/// The basic structure describing the reference path for construction of projected path CV.
///
/// # Fields
/// ```
/// lambda: exponential decay factor
/// ini_p: the initial point (initial state), which is only used for output (write_pdb and write_xyz)
/// way_p: the waypoints along the reference path with optimal alignments (n Vector containing natom*3 Array, Unit: Bohr)
/// mid_p: the midpoints of the adjacent waypoints along the reference path (n-1 Vector containing natom*3 Array, Unit: Bohr)
/// adj_dist: the distances of the adjacent waypoints along the reference path (n-1 Vector, Unit: Bohr)
/// accum_dist: the accumulated distances of the midpoints along the reference path (n-1 Vector, Unit: Bohr)
/// ```
#[derive(Debug)]
pub struct PPCV
{
    pub lambda: f64,
    pub ini_p: System,
    pub way_p: Vec< Array2<f64> >,
    pub mid_p: Vec< Array2<f64> >,
    pub way_p_fragment: Option< Vec<Array2<f64>> >,
    pub mid_p_fragment: Option< Vec<Array2<f64>> >,
    pub adj_dist: Vec<f64>,
    pub accum_dist: Vec<f64>,
}





#[derive(Debug)]
pub enum OutPoint
{
    Waypoint,
    Midpoint,
    Both,
}










impl PPCV
{
    /// Input a vector of Systems containing the original waypoint structures,
    /// align the structures and construct the PPCV
    ///
    /// # Parameters
    /// ```
    /// s: the input systems containing the structures along the reference path
    /// ppcv: the output PPCV
    /// ```
    ///
    /// # Examples
    /// ```
    /// let mut path: Vec<System> = Vec::with_capacity(8);
    /// path.push(System::read_xyz("IS.xyz"));
    /// path.push(System::read_xyz("FS.xyz"));
    /// let ppcv: PPCV = PPCV::from_system(&path, 10.0);
    /// ```
    pub fn from_system(s: &Vec<System>, lambda: f64) -> Self
    {
        let n: usize = s.len();                // Number of waypoints along the path
        if n < 2
        {
            panic!("{}", error_ref_path());                // There are at least two waypoints, one for IS, the other for FS
        }
        for i in 1..n
        {
            if s[i].natom != s[0].natom || s[i].atom_type != s[0].atom_type
            {
                panic!("{}", error_ref_path());                // All waypoints should have the same number of atoms
            }
        }


        let mut way_p: Vec<Array2<f64>> = Vec::with_capacity(n);
        let mut mid_p: Vec<Array2<f64>> = Vec::with_capacity(n-1);
        let mut adj_dist: Vec<f64> = Vec::with_capacity(n-1);
        let mut accum_dist: Vec<f64> = Vec::with_capacity(n-1);
        way_p.push(s[0].coord.clone());


        match &s[0].atom_add_pot
        {
            None =>
            {
                for i in 0..(n-1)
                {
                    let (dist, coord): (f64, Array2<f64>) = rti_dist_coord(&s[i+1].coord, &way_p[i]);
                    adj_dist.push(dist);
                    way_p.push(coord);
                    mid_p.push( 0.5 * (&way_p[i] + &way_p[i+1]) );
                    if i == 0
                    {
                        accum_dist.push( 0.5 * adj_dist[i] );
                    }
                    else
                    {
                        accum_dist.push( accum_dist[i-1] + 0.5 * (adj_dist[i-1] + adj_dist[i]) );
                    }
                }

                PPCV
                {
                    lambda,
                    ini_p: s[0].clone(),
                    way_p,
                    mid_p,
                    way_p_fragment: None,
                    mid_p_fragment: None,
                    adj_dist,
                    accum_dist,
                }
            },


            Some(atom_add_pot) =>
            {
                let mut fragment: Array2<f64> = Array2::zeros((atom_add_pot.len(), 3));
                let mut way_p_fragment: Vec<Array2<f64>> = Vec::with_capacity(n);
                let mut mid_p_fragment: Vec<Array2<f64>> = Vec::with_capacity(n-1);
                for j in 0..atom_add_pot.len()
                {
                    fragment[[j, 0]] = way_p[0][[atom_add_pot[j], 0]];
                    fragment[[j, 1]] = way_p[0][[atom_add_pot[j], 1]];
                    fragment[[j, 2]] = way_p[0][[atom_add_pot[j], 2]];
                }
                way_p_fragment.push(fragment.clone());

                for i in 0..(n-1)
                {
                    for j in 0..atom_add_pot.len()
                    {
                        fragment[[j, 0]] = s[i+1].coord[[atom_add_pot[j], 0]];
                        fragment[[j, 1]] = s[i+1].coord[[atom_add_pot[j], 1]];
                        fragment[[j, 2]] = s[i+1].coord[[atom_add_pot[j], 2]];
                    }

                    let (dist, rot, tran): (f64, Array2<f64>, Array1<f64>) = rti_dist_rot_tran(&fragment, &way_p_fragment[i]);
                    adj_dist.push(dist);
                    way_p.push(s[i+1].coord.dot(&rot) + &tran);
                    mid_p.push( 0.5 * (&way_p[i] + &way_p[i+1]) );
                    if i == 0
                    {
                        accum_dist.push( 0.5 * adj_dist[i] );
                    }
                    else
                    {
                        accum_dist.push( accum_dist[i-1] + 0.5 * (adj_dist[i-1] + adj_dist[i]) );
                    }

                    way_p_fragment.push(fragment.dot(&rot) + &tran);
                    mid_p_fragment.push( 0.5 * (&way_p_fragment[i] + &way_p_fragment[i+1]) );
                }

                PPCV
                {
                    lambda,
                    ini_p: s[0].clone(),
                    way_p,
                    mid_p,
                    way_p_fragment: Some(way_p_fragment),
                    mid_p_fragment: Some(mid_p_fragment),
                    adj_dist,
                    accum_dist,
                }
            },
        }
    }





    /// Create a new PDB file in Angstrom (if already existed, truncate it)
    ///
    /// # Parameters
    /// ```
    /// filename: name of the PDB file to be writen
    /// out_point: the output points along the path
    /// ```
    ///
    /// # Examples
    /// ```
    /// ppcv.write_pdb("filename.pdb", OutPoint::Both);
    /// ```
    pub fn write_pdb<P: AsRef<Path> + Debug>(&self, filename: P, out_point: OutPoint)
    {
        let mut s: System = self.ini_p.clone();
        match out_point
        {
            OutPoint::Waypoint =>
            {
                s.coord = self.way_p[0].clone();
                s.write_pdb(&filename, true, 0);
                for i in 1..self.way_p.len()
                {
                    s.coord = self.way_p[i].clone();
                    s.write_pdb(&filename, false, i);
                }
            },

            OutPoint::Midpoint =>
            {
                s.coord = self.mid_p[0].clone();
                s.write_pdb(&filename, true, 0);
                for i in 1..self.mid_p.len()
                {
                    s.coord = self.mid_p[i].clone();
                    s.write_pdb(&filename, false, i);
                }
            },

            OutPoint::Both =>
            {
                s.coord = self.way_p[0].clone();
                s.write_pdb(&filename, true, 0);
                for i in 0..self.mid_p.len()
                {
                    s.coord = self.mid_p[i].clone();
                    s.write_pdb(&filename, false, i);
                    s.coord = self.way_p[i+1].clone();
                    s.write_pdb(&filename, false, i+1);
                }
            },
        }
    }





    /// Create a new XYZ file in Angstrom (if already existed, truncate it)
    ///
    /// # Parameters
    /// ```
    /// filename: name of the XYZ file to be writen
    /// out_point: the output points along the path
    /// ```
    ///
    /// # Examples
    /// ```
    /// ppcv.write_xyz("filename.xyz", OutPoint::Both);
    /// ```
    pub fn write_xyz<P: AsRef<Path> + Debug>(&self, filename: P, out_point: OutPoint)
    {
        let mut s: System = self.ini_p.clone();
        match out_point
        {
            OutPoint::Waypoint =>
            {
                s.coord = self.way_p[0].clone();
                s.write_xyz(&filename, true, 0);
                for i in 1..self.way_p.len()
                {
                    s.coord = self.way_p[i].clone();
                    s.write_xyz(&filename, false, i);
                }
            },

            OutPoint::Midpoint =>
            {
                s.coord = self.mid_p[0].clone();
                s.write_xyz(&filename, true, 0);
                for i in 1..self.mid_p.len()
                {
                    s.coord = self.mid_p[i].clone();
                    s.write_xyz(&filename, false, i);
                }
            },

            OutPoint::Both =>
            {
                s.coord = self.way_p[0].clone();
                s.write_xyz(&filename, true, 0);
                for i in 0..self.mid_p.len()
                {
                    s.coord = self.mid_p[i].clone();
                    s.write_xyz(&filename, false, i);
                    s.coord = self.way_p[i+1].clone();
                    s.write_xyz(&filename, false, i+1);
                }
            },
        }
    }
}










impl ColVar for PPCV
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
    /// let s: System = ppcv.get_ini_str();
    /// ```
    fn get_ini_str(&self) -> System
    {
        self.ini_p.clone()
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
    /// let biasing_force_1d: BiasingForce1D = ppcv.initialize_biasing_force(&para);
    /// ```
    fn initialize_biasing_force(&self, para: &Para) -> BiasingForce1D
    {
        // Read in the needed parameters from para
        let cv_bin: f64 = para.md_para.eabf_atf_para.cv_bin * ANGSTROM_TO_BOHR;                // reference CV bin in eABF-ATF (Unit: bohr)
        let cv_ext: f64 = para.md_para.eabf_atf_para.cv_ext * ANGSTROM_TO_BOHR;                // extended CV in PPCV (Unit: bohr)

        // Get the minimum and maximum CV value
        let mut s: System = self.get_ini_str();
        s.coord = self.way_p[0].clone();
        let cv_min: f64 = self.get_cv(&s) - cv_bin - cv_ext;                // Minimum value of CV (Unit: bohr)
        s.coord = self.way_p[self.way_p.len()-1].clone();
        let cv_max: f64 = self.get_cv(&s) + cv_bin + cv_ext;                // Maximum value of CV (Unit: bohr)

        BiasingForce1D::from_range(&para, cv_min, cv_max)
    }





    /// Based on the reference path, calculate the projected path collective variable (PPCV) for a given structure.
    ///
    /// # Parameters
    /// ```
    /// s: the given structure
    /// cv: the output value of PPCV
    /// ```
    ///
    /// # Examples
    /// ```
    /// let cv: f64 = ppcv.get_cv(&s);
    /// ```
    fn get_cv(&self, s: &System) -> f64
    {
//        if s.natom != self.ini_p.natom
//        {
//            panic!("{}", error_ppcv());                // The input structure should have the same number of atoms with the reference path
//        }
        let n: usize = self.way_p.len();                // Number of waypoints along the path

        
        match &self.ini_p.atom_add_pot
        {
            None =>
            {
                // Align the waypoints
                let mut dist: f64;
                let mut sq_dist_to_way_p: Vec<f64> = Vec::with_capacity(n);                // Squared distances from the current structure to the waypoints
                for i in 0..n
                {
                    dist = rti_dist(&self.way_p[i], &s.coord);
                    sq_dist_to_way_p.push(dist*dist);
                }

                // Translate the exponents in global for numerical calculations
                let mut exponent_max: f64 = -1e300;                // Initialize to be a small value
                let mut weight: Vec<f64> = Vec::with_capacity(n-1);                // Weight fators of the path segments
                for i in 0..(n-1)
                {
                    dist = rti_dist(&self.mid_p[i], &s.coord);
                    weight.push(-self.lambda * dist * dist);
                    if weight[i] > exponent_max
                    {
                        exponent_max = weight[i];
                    }
                }
                exponent_max = 600.0 - exponent_max.round();

                // Calculate the important intermediate variables
                let mut lin_proj: f64;                // Linear projection
                let mut proj: Vec<f64> = Vec::with_capacity(n-1);                // Projections of the current structure on the path segments
                for i in 0..(n-1)
                {
                    weight[i] = (weight[i] + exponent_max).exp();

                    lin_proj = (sq_dist_to_way_p[i] - sq_dist_to_way_p[i+1]) / self.adj_dist[i].powi(2);
                    proj.push( self.accum_dist[i] + 0.5 * self.adj_dist[i] * lin_proj.tanh() );
                }

                // Calculate the numerator and denominator
                let mut u: f64 = 0.0;                // Numerator
                let mut v: f64 = 0.0;                // Denominator
                for i in 0..(n-1)
                {
                    u += proj[i] * weight[i];
                    v += weight[i];
                }

                u/v
            },


            Some(atom_add_pot) =>
            {
                // Abstract the fragments
                let way_p_fragment: &Vec<Array2<f64>> = self.way_p_fragment.as_ref().expect(&error_none_value("self.way_p_fragment"));
                let mid_p_fragment: &Vec<Array2<f64>> = self.mid_p_fragment.as_ref().expect(&error_none_value("self.mid_p_fragment"));
                let mut s_coord_fragment: Array2<f64> = Array2::zeros((atom_add_pot.len(), 3));
                for i in 0..atom_add_pot.len()
                {
                    s_coord_fragment[[i, 0]] = s.coord[[atom_add_pot[i], 0]];
                    s_coord_fragment[[i, 1]] = s.coord[[atom_add_pot[i], 1]];
                    s_coord_fragment[[i, 2]] = s.coord[[atom_add_pot[i], 2]];
                }

                // Align the waypoints
                let mut dist: f64;
                let mut sq_dist_to_way_p: Vec<f64> = Vec::with_capacity(n);                // Squared distances from the current structure to the waypoints
                for i in 0..n
                {
                    dist = rti_dist(&way_p_fragment[i], &s_coord_fragment);
                    sq_dist_to_way_p.push(dist*dist);
                }

                // Translate the exponents in global for numerical calculations
                let mut exponent_max: f64 = -1e300;                // Initialize to be a small value
                let mut weight: Vec<f64> = Vec::with_capacity(n-1);                // Weight fators of the path segments
                for i in 0..(n-1)
                {
                    dist = rti_dist(&mid_p_fragment[i], &s_coord_fragment);
                    weight.push(-self.lambda * dist * dist);
                    if weight[i] > exponent_max
                    {
                        exponent_max = weight[i];
                    }
                }
                exponent_max = 600.0 - exponent_max.round();

                // Calculate the important intermediate variables
                let mut lin_proj: f64;                // Linear projection
                let mut proj: Vec<f64> = Vec::with_capacity(n-1);                // Projections of the current structure on the path segments
                for i in 0..(n-1)
                {
                    weight[i] = (weight[i] + exponent_max).exp();

                    lin_proj = (sq_dist_to_way_p[i] - sq_dist_to_way_p[i+1]) / self.adj_dist[i].powi(2);
                    proj.push( self.accum_dist[i] + 0.5 * self.adj_dist[i] * lin_proj.tanh() );
                }

                // Calculate the numerator and denominator
                let mut u: f64 = 0.0;                // Numerator
                let mut v: f64 = 0.0;                // Denominator
                for i in 0..(n-1)
                {
                    u += proj[i] * weight[i];
                    v += weight[i];
                }

                u/v
            },
        }
    }





    /// Based on the reference path, calculate the projected path collective variable (PPCV) and its gradients for a given structure.
    ///
    /// # Parameters
    /// ```
    /// s: the given structure
    /// cv: the output value of PPCV
    /// grads: the output gradients of PPCV with respect to structural coordinates
    /// ```
    ///
    /// # Examples
    /// ```
    /// let (cv, grads): (f64, Array2<f64>) = ppcv.get_cv_grads(&s);
    /// ```
    fn get_cv_grads(&self, s: &System) -> (f64, Array2<f64>)
    {
//        if s.natom != self.ini_p.natom
//        {
//            panic!("{}", error_ppcv());                // The input structure should have the same number of atoms with the reference path
//        }
        let n: usize = self.way_p.len();                // Number of waypoints along the path


        match &self.ini_p.atom_add_pot
        {
            None =>
            {
                // Align the waypoints
                let mut sq_dist_to_way_p: Vec<f64> = Vec::with_capacity(n);                // Squared distances from the current structure to the waypoints
                let mut aligned_way_p: Vec<Array2<f64>> = Vec::with_capacity(n);                // The aligned waypoints with respect to the current structure
                for i in 0..n
                {
                    let (dist, coord): (f64, Array2<f64>) = rti_dist_coord(&self.way_p[i], &s.coord);
                    sq_dist_to_way_p.push(dist*dist);
                    aligned_way_p.push(coord);
                }

                // Translate the exponents in global for numerical calculations
                let mut exponent_max: f64 = -1e300;                // Initialize to be a small value
                let mut weight: Vec<f64> = Vec::with_capacity(n-1);                // Weight fators of the path segments
                let mut dweight: Vec<Array2<f64>> = Vec::with_capacity(n-1);                // Gradients of weight fators of the path segments
                for i in 0..(n-1)
                {
                    let (dist, vec): (f64, Array2<f64>) = rti_dist_vec(&self.mid_p[i], &s.coord);
                    weight.push(-self.lambda * dist * dist);
                    dweight.push(vec);
                    if weight[i] > exponent_max
                    {
                        exponent_max = weight[i];
                    }
                }
                exponent_max = 600.0 - exponent_max.round();

                // Calculate the important intermediate variables
                let mut a: f64;                // Temporary variable
                let mut lin_proj: f64;                // Linear projection
                let mut proj: Vec<f64> = Vec::with_capacity(n-1);                // Projections of the current structure on the path segments
                let mut dproj: Vec<Array2<f64>> = Vec::with_capacity(n-1);                // Gradients of projections of the current structure on the path segments
                for i in 0..(n-1)
                {
                    weight[i] = (weight[i] + exponent_max).exp();
                    dweight[i] *= -2.0 * self.lambda * weight[i];

                    lin_proj = (sq_dist_to_way_p[i] - sq_dist_to_way_p[i+1]) / self.adj_dist[i].powi(2);
                    proj.push( self.accum_dist[i] + 0.5 * self.adj_dist[i] * lin_proj.tanh() );
                    a = 1.0 / ( self.adj_dist[i] * lin_proj.cosh().powi(2) );
                    dproj.push( a * (&aligned_way_p[i+1] - &aligned_way_p[i]) );
                }

                // Calculate the numerator and denominator, and their gradients
                let mut u: f64 = 0.0;                // Numerator
                let mut v: f64 = 0.0;                // Denominator
                let mut du: Array2<f64> = Array2::zeros(s.coord.raw_dim());                // Gradients of numerator
                let mut dv: Array2<f64> = Array2::zeros(s.coord.raw_dim());                // Gradients of denominator
                for i in 0..(n-1)
                {
                    u += proj[i] * weight[i];
                    v += weight[i];
                    du = &du + weight[i] * &dproj[i] + proj[i] * &dweight[i];
                    dv = &dv + &dweight[i];
                }

                ( u/v, du/v - (u/v) * (dv/v) )
            },


            Some(atom_add_pot) =>
            {
                // Abstract the fragments
                let way_p_fragment: &Vec<Array2<f64>> = self.way_p_fragment.as_ref().expect(&error_none_value("self.way_p_fragment"));
                let mid_p_fragment: &Vec<Array2<f64>> = self.mid_p_fragment.as_ref().expect(&error_none_value("self.mid_p_fragment"));
                let mut s_coord_fragment: Array2<f64> = Array2::zeros((atom_add_pot.len(), 3));
                for i in 0..atom_add_pot.len()
                {
                    s_coord_fragment[[i, 0]] = s.coord[[atom_add_pot[i], 0]];
                    s_coord_fragment[[i, 1]] = s.coord[[atom_add_pot[i], 1]];
                    s_coord_fragment[[i, 2]] = s.coord[[atom_add_pot[i], 2]];
                }

                // Align the waypoints
                let mut sq_dist_to_way_p: Vec<f64> = Vec::with_capacity(n);                // Squared distances from the current structure to the waypoints
                let mut aligned_way_p: Vec<Array2<f64>> = Vec::with_capacity(n);                // The aligned waypoints with respect to the current structure
                for i in 0..n
                {
                    let (dist, coord): (f64, Array2<f64>) = rti_dist_coord(&way_p_fragment[i], &s_coord_fragment);
                    sq_dist_to_way_p.push(dist*dist);
                    aligned_way_p.push(coord);
                }

                // Translate the exponents in global for numerical calculations
                let mut exponent_max: f64 = -1e300;                // Initialize to be a small value
                let mut weight: Vec<f64> = Vec::with_capacity(n-1);                // Weight fators of the path segments
                let mut dweight: Vec<Array2<f64>> = Vec::with_capacity(n-1);                // Gradients of weight fators of the path segments
                for i in 0..(n-1)
                {
                    let (dist, vec): (f64, Array2<f64>) = rti_dist_vec(&mid_p_fragment[i], &s_coord_fragment);
                    weight.push(-self.lambda * dist * dist);
                    dweight.push(vec);
                    if weight[i] > exponent_max
                    {
                        exponent_max = weight[i];
                    }
                }
                exponent_max = 600.0 - exponent_max.round();

                // Calculate the important intermediate variables
                let mut a: f64;                // Temporary variable
                let mut lin_proj: f64;                // Linear projection
                let mut proj: Vec<f64> = Vec::with_capacity(n-1);                // Projections of the current structure on the path segments
                let mut dproj: Vec<Array2<f64>> = Vec::with_capacity(n-1);                // Gradients of projections of the current structure on the path segments
                for i in 0..(n-1)
                {
                    weight[i] = (weight[i] + exponent_max).exp();
                    dweight[i] *= -2.0 * self.lambda * weight[i];

                    lin_proj = (sq_dist_to_way_p[i] - sq_dist_to_way_p[i+1]) / self.adj_dist[i].powi(2);
                    proj.push( self.accum_dist[i] + 0.5 * self.adj_dist[i] * lin_proj.tanh() );
                    a = 1.0 / ( self.adj_dist[i] * lin_proj.cosh().powi(2) );
                    dproj.push( a * (&aligned_way_p[i+1] - &aligned_way_p[i]) );
                }

                // Calculate the numerator and denominator, and their gradients
                let mut u: f64 = 0.0;                // Numerator
                let mut v: f64 = 0.0;                // Denominator
                let mut du: Array2<f64> = Array2::zeros(s_coord_fragment.raw_dim());                // Gradients of numerator
                let mut dv: Array2<f64> = Array2::zeros(s_coord_fragment.raw_dim());                // Gradients of denominator
                for i in 0..(n-1)
                {
                    u += proj[i] * weight[i];
                    v += weight[i];
                    du = &du + weight[i] * &dproj[i] + proj[i] * &dweight[i];
                    dv = &dv + &dweight[i];
                }

                // Reconstruct the original gradients
                let grads_fragment: Array2<f64> = du/v - (u/v) * (dv/v);
                let mut grads: Array2<f64> = Array2::zeros(s.coord.raw_dim());
                for i in 0..atom_add_pot.len()
                {
                    grads[[atom_add_pot[i], 0]] = grads_fragment[[i, 0]];
                    grads[[atom_add_pot[i], 1]] = grads_fragment[[i, 1]];
                    grads[[atom_add_pot[i], 2]] = grads_fragment[[i, 2]];
                }

                (u/v, grads)
            },
        }
    }
}










