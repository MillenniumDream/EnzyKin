//! The module for training data processing
use std::path::{Path, PathBuf};
use std::fmt::Debug;
use crate::common::constants::Element;
use crate::common::error::*;
use crate::common::file::traverse_leaf_dir;
use crate::nn::fragment_definition::FragmentType;
use crate::nn::protein::ProteinSystem;
use crate::nn::descriptor::{N_DES, truncate_cluster, get_descriptor};
use crate::nn::normalization::{FragmentNorm, GlobalNorm};
use crate::long_ranged::aperiodic_system::get_aperiodic_system_es_energy_force_serial;
use ndarray::{Array1, Array2, Array3, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use savefile::load_file;
use savefile_derive::Savefile;





pub const EPS: f64 = 0.0000001;





/// The basic structure reserving the DFT data for a series of structures of a molecular system, which is going to be saved into the disk
///
/// # Fields
/// ```
/// nstruct: the number of structures of the molecular system
/// cell: the fixed cell of the molecular system (3*3 Vec, Unit: Bohr)
/// natom: the number of atoms of the molecular system
/// atom_type: the types of the atoms within the molecular system (natom Vec)
/// coord: the atomic coordinates of all the structures of the molecular system (nstruct*natom*3 Vec, Unit: Bohr)
/// pot: the potential energies of all the structures of the molecular system (nstruct Vec, Unit: Hartree)
/// force: the atomic forces of all the structures of the molecular system (nstruct*natom*3 Vec, Unit: Hartree/Bohr)
/// ```
#[derive(Debug, Savefile)]
pub struct DataSaved
{
    pub nstruct: usize,
    pub cell: Option< Vec<f64> >,
    pub natom: usize,
    pub atom_type: Vec<Element>,
    pub coord: Vec<f64>,
    pub pot: Vec<f64>,
    pub force: Vec<f64>,
}





/// Basic structure for the structural descriptor of a fragment,
/// derived from the fragment structure and its finite difference perturbation (only for training)
///
/// # Fields
/// ```
/// fragment_type: the type of the fragment (Atom, Residue, or Molecule)
/// descriptor: the descriptor (N_DES), derived from the fragment structure
/// descriptor_diff: the finite difference descriptor (6*natom*N_DES), derived from the finite difference perturbation of the fragment structure
/// 
/// ```
#[derive(Debug)]
pub struct FragmentDescriptor
{
    pub fragment_type: FragmentType,
    pub descriptor: Vec<f64>,
    pub descriptor_diff: Vec<f64>,
}





/// Basic structure for the structural descriptor of a protein system,
/// derived from the protein system structure and its finite difference perturbation (only for training)
///
/// # Fields
/// ```
/// n_diff: the number of finite difference perturbation structures
/// fragment_descriptor: Assembly of structural descriptors of all the fragments in a protein system
/// pot: DFT potential energy of the protein system
/// pot_diff: DFT potential energy of the finite difference perturbation of the protein system (6*natom)
/// ```
#[derive(Debug)]
pub struct ProteinSystemDescriptor
{
    pub n_diff: usize,
    pub fragment_descriptor: Vec<FragmentDescriptor>,
    pub pot: f64,
    pub pot_diff: Vec<f64>,
}










impl ProteinSystemDescriptor
{
    /// Derive the normalized structural descriptor and finite difference perturbation for the input ProteinSystem.
    /// Note that the long-ranged interactions have been included.
    ///
    /// # Parameters
    /// ```
    /// s: the input ProteinSystem, specially containing the potential energy and atomic forces from DFT calculations
    /// global_norm: the normalization parameters containing mean values and standard deviations of all the fragments
    /// des: the output descriptors for the ProteinSystem and its finite difference perturbation
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn from_struct(s: &ProteinSystem, global_norm: &GlobalNorm) -> Self
    {
        let n_diff: usize = s.natom * 6;
        let mut within_list: bool;                // To reserve if the atom is within the list or not
        let mut index: usize = 0;                // To reserve the index of the atom in the list
        let mut row: usize;
        let mut pot_mean: f64 = 0.0;
        let mut fragment_norm: &FragmentNorm;



        // Achieve the fragment descriptors for all the fragments in the protein
        let mut fragment_descriptor: Vec<FragmentDescriptor> = Vec::with_capacity(s.fragment.len());
        // For each fragment in the protein system
        for i in 0..s.fragment.len()
        {
            // Get the normalization parameters for fragment i, and accumulate its potential mean value
            fragment_norm = global_norm.get_fragment_norm(&s.fragment[i].fragment_type);
            pot_mean += fragment_norm.pot_mean;


            // Achieve the descriptor for Fragment i
            let (coord_within_rcut, list_within_rcut, atomic_number_within_rcut): (Array2<f64>, Vec<usize>, Vec<usize>) = truncate_cluster(&s, i);
            let (mut descriptor, gradient): (Vec<f64>, Array2<f64>) = get_descriptor(coord_within_rcut, atomic_number_within_rcut, s.fragment[i].natom);
//            assert_eq!(descriptor.len(), fragment_norm.mean.len());
            for j in 0..descriptor.len()
            {
                descriptor[j] = (descriptor[j] - fragment_norm.mean[j]) / fragment_norm.dev[j];
            }


            // Achieve the descriptor_diff for Fragment i
            let mut descriptor_diff: Vec<f64> = Vec::with_capacity(n_diff * N_DES);
            // For each atom in the protein system
            for j in 0..s.natom
            {
                // Jubge if the atom is within the list or not. If within the list, reserve its index in the list
                within_list = false;
                for k in 0..list_within_rcut.len()
                {
                    if j == list_within_rcut[k]
                    {
                        within_list = true;
                        index = k;
                        break
                    }
                }

                // Obtain the descriptor for the finite difference perturbation on the atom
                if within_list                // If the atom is within the list, induce the finite difference perturbation on the descriptor
                {
                    // For x, y, z axis
                    for axis in 0..3
                    {
                        row = index * 3 + axis;
                        for k in 0..N_DES
                        {
                            descriptor_diff.push(descriptor[k] + EPS * gradient[[row, k]] / fragment_norm.dev[k]);
                        }
                        for k in 0..N_DES
                        {
                            descriptor_diff.push(descriptor[k] - EPS * gradient[[row, k]] / fragment_norm.dev[k]);
                        }
                    }
                }
                else                // If the atom is outside the list, its finite difference perturbation doesn't affect the descriptor
                {
                    for _ in 0..6
                    {
                        for k in 0..N_DES
                        {
                            descriptor_diff.push(descriptor[k]);
                        }
                    }
                }
            }


            // Save the descriptors of Fragment i
            fragment_descriptor.push
            (
                FragmentDescriptor
                {
                    fragment_type: s.fragment[i].fragment_type.clone(),
                    descriptor,
                    descriptor_diff,
                }
            );
        }



        // Obtain the long-ranged interaction potential and force
        let (pot_es, force_es): (f64, Array2<f64>) = match &s.cell
        {
            Some(_cell) => panic!("\n\n\n Have not implemented the long-ranged calculations for the periodic system, check it. \n\n\n"),
            None => get_aperiodic_system_es_energy_force_serial(&s),
        };
        // Achieve the potential energies (excluding the long-ranged interactions) for the protein system and its finite difference perturbation
        let pot_nn: f64 = s.pot - pot_es - pot_mean;
        let force_nn: Array2<f64> = s.force.clone().expect(&error_none_value("s.force")) - force_es;
        let mut pot_nn_diff: Vec<f64> = Vec::with_capacity(n_diff);
        // For each atom in the protein system
        for i in 0..s.natom
        {
            for j in 0..3
            {
                pot_nn_diff.push(-EPS * force_nn[[i, j]]);
                pot_nn_diff.push(EPS * force_nn[[i, j]]);
            }
        }



        // Return the whole structure
        ProteinSystemDescriptor
        {
            n_diff,
            fragment_descriptor,
            pot: pot_nn,
            pot_diff: pot_nn_diff,
        }
    }





    /// Load a single structural data for a specific protein system, and calculate the descriptors
    ///
    /// # Parameters
    /// ```
    /// input_data_dir: the directory containing the input data files
    /// norm_dir: the directory containing the global normalization parameters
    /// long_ranged_dir: the directory containing the long-ranged interaction parameters
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn from_single_data<P: AsRef<Path> + Debug>(input_data_dir: P, norm_dir: P, long_ranged_dir: Option<PathBuf>) -> Vec<Self>
    {
        // Obtain the input filenames
        let str_input_file: PathBuf = input_data_dir.as_ref().join("str.pdb");
        let data_input_file: PathBuf = input_data_dir.as_ref().join("data.bin");

        // Load the input files
        let mut s: ProteinSystem = ProteinSystem::read_pdb(&str_input_file);
        let data: DataSaved = load_file(&data_input_file, 0).expect(&error_file("reading", &data_input_file));

        // Assert if the input protein system and the input DFT data is matching
        assert_eq!(s.natom, data.natom);
        assert_eq!(s.atom_type, data.atom_type);

        // Extract the atomic coordinates, potentials, and forces from the data
        let coord: Array3<f64> = Array3::from_shape_vec((data.nstruct, data.natom, 3), data.coord).expect(&error_none_value("data.coord"));
        let pot: Vec<f64> = data.pot;
        let force: Array3<f64> = Array3::from_shape_vec((data.nstruct, data.natom, 3), data.force).expect(&error_none_value("data.force"));

        // Load the global normalization parameters
        let global_norm: GlobalNorm = GlobalNorm::load(norm_dir);

        // Initialize the long-ranged interaction parameters
        s.long_ranged_dir = long_ranged_dir;
        s.get_fixed_long_ranged_para();
//        s = s.to_atom_fragment();



        // Define the Vec to contain the protein descriptors
        let mut protein_descriptor: Vec<ProteinSystemDescriptor> = Vec::with_capacity(data.nstruct);
        // Calculate the descriptors for all the structures
        for i in 0..data.nstruct                // For each structure
        {
            // Copy the atomic coordinates, potential, forces to the ProteinSystem
            s.coord = coord.slice(s![i, .., ..]).to_owned();
            for j in 0..s.fragment.len()                // For each fragment in the structure
            {
                let atom_list: Vec<usize> = s.fragment[j].atom_list.clone().expect(&error_none_value("s.fragment[j].atom_list"));
                for k in 0..atom_list.len()                // For each atom in the fragment
                {
                    s.fragment[j].coord[[k, 0]] = s.coord[[atom_list[k], 0]];
                    s.fragment[j].coord[[k, 1]] = s.coord[[atom_list[k], 1]];
                    s.fragment[j].coord[[k, 2]] = s.coord[[atom_list[k], 2]];
                }
            }
            s.pot = pot[i];
            s.force = Some(force.slice(s![i, .., ..]).to_owned());

            // Calculate the descriptors
            protein_descriptor.push(Self::from_struct(&s, &global_norm));
        }

        protein_descriptor
    }





    /// Randomly select a structural data set, and randomly select a batch size of structures, calculate the descriptors for a specific protein system
    ///
    /// # Parameters
    /// ```
    /// input_data_dir: the parent directories containing all the leaf directories and the corresponding training data
    /// norm_dir: the directory containing the global normalization parameters
    /// long_ranged_dir: the directory containing the long-ranged interaction parameters
    /// training_batch_size: the number of structure to be select for calculation
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn from_random_data<P: AsRef<Path> + Debug>(input_data_dir: &Vec<PathBuf>, norm_dir: P, long_ranged_dir: Option<PathBuf>, training_batch_size: usize) -> Vec<Self>
    {
        // Randomly select a structural data set
        let leaf_dir: Vec<PathBuf> = traverse_leaf_dir(input_data_dir);
        let random_index: usize = Array1::random(1, Uniform::new(0, leaf_dir.len()))[0];

        // Obtain the input filenames
        let str_input_file: PathBuf = leaf_dir[random_index].join("str.pdb");
        let data_input_file: PathBuf = leaf_dir[random_index].join("data.bin");

        // Load the input files
        let mut s: ProteinSystem = ProteinSystem::read_pdb(&str_input_file);
        let data: DataSaved = load_file(&data_input_file, 0).expect(&error_file("reading", &data_input_file));

        // Assert if the input protein system and the input DFT data is matching
        assert_eq!(s.natom, data.natom);
        assert_eq!(s.atom_type, data.atom_type);

        // Extract the atomic coordinates, potentials, and forces from the data
        let coord: Array3<f64> = Array3::from_shape_vec((data.nstruct, data.natom, 3), data.coord).expect(&error_none_value("data.coord"));
        let pot: Vec<f64> = data.pot;
        let force: Array3<f64> = Array3::from_shape_vec((data.nstruct, data.natom, 3), data.force).expect(&error_none_value("data.force"));

        // Load the global normalization parameters
        let global_norm: GlobalNorm = GlobalNorm::load(norm_dir);

        // Initialize the long-ranged interaction parameters
        s.long_ranged_dir = long_ranged_dir;
        s.get_fixed_long_ranged_para();
//        s = s.to_atom_fragment();



        // If the data set has less structures than the training_batch_size, select all its structures
        if data.nstruct <= training_batch_size
        {
            // Define the Vec to contain the protein descriptors
            let mut protein_descriptor: Vec<ProteinSystemDescriptor> = Vec::with_capacity(data.nstruct);
            // Calculate the descriptors for all the structures
            for i in 0..data.nstruct                // For each structure
            {
                // Copy the atomic coordinates, potential, forces to the ProteinSystem
                s.coord = coord.slice(s![i, .., ..]).to_owned();
                for j in 0..s.fragment.len()                // For each fragment in the structure
                {
                    let atom_list: Vec<usize> = s.fragment[j].atom_list.clone().expect(&error_none_value("s.fragment[j].atom_list"));
                    for k in 0..atom_list.len()                // For each atom in the fragment
                    {
                        s.fragment[j].coord[[k, 0]] = s.coord[[atom_list[k], 0]];
                        s.fragment[j].coord[[k, 1]] = s.coord[[atom_list[k], 1]];
                        s.fragment[j].coord[[k, 2]] = s.coord[[atom_list[k], 2]];
                    }
                }
                s.pot = pot[i];
                s.force = Some(force.slice(s![i, .., ..]).to_owned());
                // Calculate the descriptors
                protein_descriptor.push(Self::from_struct(&s, &global_norm));
            }

            return protein_descriptor;
        }
        // Otherwise, select training_batch_size structures from it
        else
        {
            // Randomly select training_batch_size structures from the data set
            let random_index: Array1<usize> = Array1::random(training_batch_size, Uniform::new(0, data.nstruct));
            // Define the Vec to contain the protein descriptors
            let mut protein_descriptor: Vec<ProteinSystemDescriptor> = Vec::with_capacity(training_batch_size);
            // Calculate the descriptors for all the structures
            for i in 0..training_batch_size                // For each structure
            {
                // Copy the atomic coordinates, potential, forces to the ProteinSystem
                s.coord = coord.slice(s![random_index[i], .., ..]).to_owned();
                for j in 0..s.fragment.len()                // For each fragment in the structure
                {
                    let atom_list: Vec<usize> = s.fragment[j].atom_list.clone().expect(&error_none_value("s.fragment[j].atom_list"));
                    for k in 0..atom_list.len()                // For each atom in the fragment
                    {
                        s.fragment[j].coord[[k, 0]] = s.coord[[atom_list[k], 0]];
                        s.fragment[j].coord[[k, 1]] = s.coord[[atom_list[k], 1]];
                        s.fragment[j].coord[[k, 2]] = s.coord[[atom_list[k], 2]];
                    }
                }
                s.pot = pot[random_index[i]];
                s.force = Some(force.slice(s![random_index[i], .., ..]).to_owned());
                // Calculate the descriptors
                protein_descriptor.push(Self::from_struct(&s, &global_norm));
            }

            return protein_descriptor;
        }
    }
}










