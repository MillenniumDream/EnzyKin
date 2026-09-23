//! About the neural network potential energy surface
use std::path::PathBuf;
use crate::common::constants::Device;
use crate::common::error::*;
use crate::common::traits::ProteinPES;
use crate::nn::protein::ProteinSystem;
use crate::nn::descriptor::{N_DES, truncate_cluster, get_descriptor};
use crate::long_ranged::aperiodic_system::{get_aperiodic_system_es_energy_serial, get_aperiodic_system_es_energy_force_serial};
use crate::nn::normalization::{FragmentNorm, GlobalNorm};
use crate::nn::global_nn::{GlobalNN, NN};
use ndarray::{Array1, Array2};
use dfdx::shapes::{Const, Rank0, Rank1};
use dfdx::tensor::{Tensor, TensorFromVec, Gradients, Trace, NoneTape, OwnedTape, AsArray};
use dfdx::tensor_ops::{SumTo, Backward};
use dfdx::nn::{Module, ZeroGrads};










/// The structure containing all the information about the neural network potential energy surface
///
/// # Fields
/// ```
/// dev: the device for tensor construction (CPU or Cuda)
/// global_nn: the global neural network potential
/// global_norm: the global normalization parameters for descriptors
/// long_ranged_dir: the directory containing the long-ranged interaction parameters
/// output_path: the path for output file
/// ```
pub struct NNPot
{
    pub dev: Device,
    pub global_nn: GlobalNN,
    pub global_norm: GlobalNorm,
    pub long_ranged_dir: Option<PathBuf>,
    pub output_path: PathBuf,
}





impl ProteinPES for NNPot
{
    /// Input a protein system, calculate and output its potential energy according to the NN potential energy surface (PES)
    ///
    /// # Parameters
    /// ```
    /// s: the input protein system
    /// pot_nn_es: the output potential energy for neural network and electrostatics
    /// ```
    fn get_energy(&self, s: &mut ProteinSystem) -> f64
    {
        if (s.long_ranged_initialized == false) || (s.long_ranged_dir != self.long_ranged_dir)
        {
            s.long_ranged_dir = self.long_ranged_dir.clone();
            s.get_fixed_long_ranged_para();
//            *s = s.to_atom_fragment();
        }

        get_energy_serial(&self.dev, &self.global_nn, &self.global_norm, &s)
    }



    /// Input a protein system,
    /// calculate and output its potential energy and atomic forces according to the NN potential energy surface (PES)
    ///
    /// # Parameters
    /// ```
    /// s: the input protein system
    /// pot_nn_es: the output total potential energy for neural network and electrostatics
    /// force_nn_es: the output total atomic forces (s.natom * 3) for neural network and electrostatics
    /// ```
    fn get_energy_force(&self, s: &mut ProteinSystem) -> (f64, Array2<f64>)
    {
        if (s.long_ranged_initialized == false) || (s.long_ranged_dir != self.long_ranged_dir)
        {
            s.long_ranged_dir = self.long_ranged_dir.clone();
            s.get_fixed_long_ranged_para();
//            *s = s.to_atom_fragment();
        }

        get_energy_force_serial(&self.dev, &self.global_nn, &self.global_norm, &s)
    }
}










/// Calculate the potentail energy for a given protein system according to the global neural network (serial version)
/// Of note, the long-range interaction parameters should be assigned to each atom in advance for the ES calculations
///
/// # Parameters
/// ```
/// dev: the input device for tensor construction (CPU or Cuda)
/// global_nn: the input global neural network potential energy surface
/// global_norm: the input global normalization parameters for descriptors
/// s: the input protein system
/// pot_nn_es: the output potential energy for neural network and electrostatics
/// ```
///
/// # Examples
/// ```
/// ```
pub fn get_energy_serial(dev: &Device, global_nn: &GlobalNN, global_norm: &GlobalNorm, s: &ProteinSystem) -> f64
{
    let mut pot_nn: f64 = 0.0;
    let mut pot_nn_i: f64;
    let mut pot_mean: f64 = 0.0;
    let mut fragment_norm: &FragmentNorm;

    // Calculate the potential energy for Fragment i
    for i in 0..s.fragment.len()
    {
        // Accumulate the pot_mean
        fragment_norm = global_norm.get_fragment_norm(&s.fragment[i].fragment_type);
        pot_mean += fragment_norm.pot_mean;

        // Get the fragment descriptor
        let (coord_within_rcut, _list_within_rcut, atomic_number_within_rcut): (Array2<f64>, Vec<usize>, Vec<usize>) = truncate_cluster(&s, i);
        let (mut descriptor, _gradient): (Vec<f64>, Array2<f64>) = get_descriptor(coord_within_rcut, atomic_number_within_rcut, s.fragment[i].natom);
        for j in 0..descriptor.len()
        {
            descriptor[j] = (descriptor[j] - fragment_norm.mean[j]) / fragment_norm.dev[j];
        }
        let descriptor: Tensor<Rank1<N_DES>, f64, Device, NoneTape> = dev.tensor_from_vec(descriptor, (Const,));

        // Calculate the potential energy for Fragment i
        pot_nn_i = match global_nn.get_fragment_nn(&s.fragment[i].fragment_type)
        {
            NN::Small(nn_small) => nn_small.forward(descriptor).sum().array(),
            NN::Middle(nn_middle) => nn_middle.forward(descriptor).sum().array(),
            NN::Large(nn_large) => nn_large.forward(descriptor).sum().array(),
        };

        // Accumulate the potential energy
        pot_nn += pot_nn_i;
    }

    // Calculate the long-ranged interaction potential
    let pot_es: f64 = match &s.cell
    {
        Some(_cell) => panic!("\n\n\n Have not implemented the long-ranged calculations for the periodic system, check it. \n\n\n"),
        None => get_aperiodic_system_es_energy_serial(&s),
    };

    pot_nn + pot_mean + pot_es
}





/// Calculate the potentail energy and force for a given protein system according to the global neural network (serial version)
/// Of note, the long-range interaction parameters should be assigned to each atom in advance for the ES calculations
///
/// # Parameters
/// ```
/// dev: the input device for tensor construction (CPU or Cuda)
/// global_nn: the input global neural network potential energy surface
/// global_norm: the input global normalization parameters for descriptors
/// s: the input protein system
/// pot_nn_es: the output potential energy for neural network and electrostatics
/// force_nn_es: the output atomic force for neural network and electrostatics
/// ```
///
/// # Examples
/// ```
/// ```
pub fn get_energy_force_serial(dev: &Device, global_nn: &GlobalNN, global_norm: &GlobalNorm, s: &ProteinSystem) -> (f64, Array2<f64>)
{
    let mut pot_nn: f64 = 0.0;
    let mut force_nn: Array2<f64> = Array2::zeros((s.natom, 3));
    let mut pot_nn_i: f64;
    let mut force_nn_i: Array2<f64>;
    let mut pot_mean: f64 = 0.0;
    let mut fragment_norm: &FragmentNorm;
    let mut derivative1: [f64; N_DES];
    let mut derivative2: Array1<f64> = Array1::zeros(N_DES);


    // Calculate the potential energy and atomic force for Fragment i
    for i in 0..s.fragment.len()
    {
        // Accumulate the pot_mean
        fragment_norm = global_norm.get_fragment_norm(&s.fragment[i].fragment_type);
        pot_mean += fragment_norm.pot_mean;

        // Get the fragment descriptor
        let (coord_within_rcut, list_within_rcut, atomic_number_within_rcut): (Array2<f64>, Vec<usize>, Vec<usize>) = truncate_cluster(&s, i);
        let (mut descriptor, gradient): (Vec<f64>, Array2<f64>) = get_descriptor(coord_within_rcut, atomic_number_within_rcut, s.fragment[i].natom);
        for j in 0..descriptor.len()
        {
            descriptor[j] = (descriptor[j] - fragment_norm.mean[j]) / fragment_norm.dev[j];
        }
        let descriptor: Tensor<Rank1<N_DES>, f64, Device, NoneTape> = dev.tensor_from_vec(descriptor, (Const,));
        let mut grads: Gradients<f64, Device> = descriptor.alloc_grads();


        // Calculate the potential energy and atomic force for Fragment i
        let pot: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = match global_nn.get_fragment_nn(&s.fragment[i].fragment_type)
        {
            NN::Small(nn_small) => nn_small.forward(descriptor.trace(grads)).sum(),

            NN::Middle(nn_middle) => nn_middle.forward(descriptor.trace(grads)).sum(),

            NN::Large(nn_large) => nn_large.forward(descriptor.trace(grads)).sum(),
        };
        pot_nn_i = pot.array();
        grads = pot.backward();
        derivative1 = grads.get(&descriptor).array();
        for j in 0..N_DES
        {
            derivative2[j] = -derivative1[j] / fragment_norm.dev[j];
        }
        force_nn_i = gradient.dot(&derivative2).into_shape((list_within_rcut.len(), 3)).expect(&error_none_value("derivative3"));


        // Accumulate the potential energy and atomic force
        pot_nn += pot_nn_i;
        for j in 0..list_within_rcut.len()
        {
            force_nn[[list_within_rcut[j], 0]] += force_nn_i[[j, 0]];
            force_nn[[list_within_rcut[j], 1]] += force_nn_i[[j, 1]];
            force_nn[[list_within_rcut[j], 2]] += force_nn_i[[j, 2]];
        }
    }


    // Calculate the long-ranged interaction potential and force
    let (pot_es, force_es): (f64, Array2<f64>) = match &s.cell
    {
        Some(_cell) => panic!("\n\n\n Have not implemented the long-ranged calculations for the periodic system, check it. \n\n\n"),
        None => get_aperiodic_system_es_energy_force_serial(&s),
    };


    (pot_nn + pot_mean + pot_es, force_nn + force_es)
}










