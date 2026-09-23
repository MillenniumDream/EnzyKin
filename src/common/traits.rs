//! About the traits
use std::path::PathBuf;
use crate::io::input::Para;
use crate::pes_exploration::system::System;
use crate::nn::protein::ProteinSystem;
use crate::md::eabf_atf::BiasingForce1D;
use ndarray::Array2;
use mpi::traits::Communicator;





pub trait PES
{
    fn get_energy(&self, s: &System) -> f64;
    fn get_energy_force(&self, s: &System) -> (f64, Array2<f64>);
}

pub trait ProteinPES
{
    fn get_energy(&self, s: &mut ProteinSystem) -> f64;
    fn get_energy_force(&self, s: &mut ProteinSystem) -> (f64, Array2<f64>);
}





pub trait ColVar
{
    fn get_ini_str(&self) -> System;
    fn initialize_biasing_force(&self, para: &Para) -> BiasingForce1D;
    fn get_cv(&self, s: &System) -> f64;
    fn get_cv_grads(&self, s: &System) -> (f64, Array2<f64>);
}





pub trait RtipPathSampling
{
    fn rtip_path_sampling<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf);
}

pub trait IdwmPathSampling
{
    fn idwm_path_sampling<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf);
}





pub trait BussiNVEMD
{
    fn bussi_nve_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf);
}

pub trait BussiNVTMD
{
    fn bussi_nvt_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf);
}





pub trait LangevinNVEMD
{
    fn langevin_nve_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf);
}

pub trait LangevinNVTMD
{
    fn langevin_nvt_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf);
}





pub trait BerendsenNVTMD
{
    fn berendsen_nvt_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf);
}

pub trait BerendsenNVTMD2
{
    fn berendsen_nvt_md2<C: Communicator, P: PES>(&self, comm: &C, real_pes1: &P, real_pes2: &P, para: &Para, output_path: &PathBuf);
}










