//! Normalization of the descriptor and potential energy
use std::fs;
use std::path::{Path, PathBuf};
use std::fmt::Debug;
use crate::common::constants::Element;
use crate::common::error::*;
use crate::common::file::traverse_leaf_dir;
use crate::nn::fragment_definition::{AminoAcid, Molecule, Cap, FragmentType};
use crate::nn::protein::ProteinSystem;
use crate::nn::descriptor::{N_DES, truncate_cluster, get_descriptor};
use crate::nn::training_data::DataSaved;
use ndarray::{Array2, Array3, s};
use savefile::{save_file, load_file};
use savefile_derive::Savefile;





/// The basic structure containing mean value and standard deviation of the descriptor and potential energy for a fragment.
///
/// # Fields
/// ```
/// nfragment: number of fragments for the calculations
/// mean: mean value of the descriptor
/// dev: standard deviation of the descriptor
/// pot_mean: mean value of the potential energy
/// ```
#[derive(Debug, Savefile)]
pub struct FragmentNorm
{
    pub nfragment: usize,
    pub mean: [f64; N_DES],
    pub dev: [f64; N_DES],
    pub pot_mean: f64,
}





/// The basic structure containing mean value and standard deviation of all the fragments
///
/// # Fields
/// ```
/// ```
#[derive(Debug, Savefile)]
pub struct GlobalNorm
{
    pub element_h_norm: FragmentNorm,
    pub element_b_norm: FragmentNorm,
    pub element_c_norm: FragmentNorm,
    pub element_n_norm: FragmentNorm,
    pub element_o_norm: FragmentNorm,
    pub element_f_norm: FragmentNorm,
    pub element_na_norm: FragmentNorm,
    pub element_mg_norm: FragmentNorm,
    pub element_si_norm: FragmentNorm,
    pub element_p_norm: FragmentNorm,
    pub element_s_norm: FragmentNorm,
    pub element_cl_norm: FragmentNorm,
    pub element_k_norm: FragmentNorm,
    pub element_ca_norm: FragmentNorm,
    pub element_v_norm: FragmentNorm,
    pub element_cr_norm: FragmentNorm,
    pub element_mn_norm: FragmentNorm,
    pub element_fe_norm: FragmentNorm,
    pub element_co_norm: FragmentNorm,
    pub element_ni_norm: FragmentNorm,
    pub element_cu_norm: FragmentNorm,
    pub element_zn_norm: FragmentNorm,
    pub element_as_norm: FragmentNorm,
    pub element_se_norm: FragmentNorm,
    pub element_br_norm: FragmentNorm,
    pub element_mo_norm: FragmentNorm,
    pub element_cd_norm: FragmentNorm,
    pub element_sn_norm: FragmentNorm,
    pub element_i_norm: FragmentNorm,

    pub amino_acid_gly_norm: FragmentNorm,
    pub amino_acid_ala_norm: FragmentNorm,
    pub amino_acid_val_norm: FragmentNorm,
    pub amino_acid_leu_norm: FragmentNorm,
    pub amino_acid_ile_norm: FragmentNorm,
    pub amino_acid_ser_norm: FragmentNorm,
    pub amino_acid_thr_norm: FragmentNorm,
    pub amino_acid_asp_norm: FragmentNorm,
    pub amino_acid_ash_norm: FragmentNorm,
    pub amino_acid_asn_norm: FragmentNorm,
    pub amino_acid_glu_norm: FragmentNorm,
    pub amino_acid_glh_norm: FragmentNorm,
    pub amino_acid_gln_norm: FragmentNorm,
    pub amino_acid_lys_norm: FragmentNorm,
    pub amino_acid_lyn_norm: FragmentNorm,
    pub amino_acid_arg_norm: FragmentNorm,
//    pub amino_acid_arn_norm: FragmentNorm,
    pub amino_acid_cys_norm: FragmentNorm,
    pub amino_acid_cyx_norm: FragmentNorm,
//    pub amino_acid_cym_norm: FragmentNorm,
    pub amino_acid_met_norm: FragmentNorm,
    pub amino_acid_hid_norm: FragmentNorm,
    pub amino_acid_hie_norm: FragmentNorm,
    pub amino_acid_hip_norm: FragmentNorm,
    pub amino_acid_phe_norm: FragmentNorm,
    pub amino_acid_tyr_norm: FragmentNorm,
    pub amino_acid_trp_norm: FragmentNorm,
    pub amino_acid_pro_norm: FragmentNorm,

    pub head_h_norm: FragmentNorm,
    pub tail_o_norm: FragmentNorm,
    pub tail_h_norm: FragmentNorm,
    pub head_ace_norm: FragmentNorm,
    pub tail_nme_norm: FragmentNorm,
    pub tail_nhe_norm: FragmentNorm,

    pub molecule_wat_norm: FragmentNorm,
}










impl FragmentNorm
{
    /// Initialize the normalization parameters for a given fragment
    ///
    /// # Parameters
    /// ```
    /// pot_mean: the input mean value of potential energy
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn initialize(pot_mean: f64) -> Self
    {
        FragmentNorm
        {
            nfragment: 0,
            mean: [0.0; N_DES],
            dev: [1.0; N_DES],
            pot_mean,
        }
    }



    /// Accumulate the input descriptors to the mean values
    ///
    /// # Parameters
    /// ```
    /// descriptor: the input descriptor
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn mean_accumulation(&mut self, descriptor: &Vec<f64>)
    {
        // Update the accumulated number of fragments
        self.nfragment += 1;

        // Accumulate the descriptors to the mean values
//        assert_eq!(self.mean.len(), descriptor.len());
        for i in 0..self.mean.len()
        {
            self.mean[i] += descriptor[i];
        }
    }



    /// Accumulate the input descriptors to the standard deviations
    ///
    /// # Parameters
    /// ```
    /// descriptor: the input descriptor
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn dev_accumulation(&mut self, descriptor: &Vec<f64>)
    {
        // Update the accumulated number of fragments
        self.nfragment += 1;

        // Accumulate the descriptors to the standard deviations
//        assert_eq!(self.dev.len(), descriptor.len());
        for i in 0..self.dev.len()
        {
            self.dev[i] += (descriptor[i] - self.mean[i]).powi(2);
        }
    }



    /// According to the accumulated sum, calculate the mean values of the descriptor for a given fragment
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn mean_calculation(&mut self)
    {
        match self.nfragment
        {
            // If none of the fragment has been accumulated, do nothing.
            // Remain the mean values to be the default 0, and remain the standard deviations to be the default 1.
            0 => (),

            // If only one fragment has been accumulated, remain the mean values to be the accumulated value,
            // and remain the standard deviations to be the default 1.
            1 => self.nfragment = 0,

            // If the accumulated number of fragment is larger than 2, calculate the mean values by average,
            // and reset the standard deviation to be 0 for its following accumulations
            _ =>
            {
                // Calculate the mean values by average
                for i in 0..self.mean.len()
                {
                    self.mean[i] /= self.nfragment as f64;
                }

                // Reset the standard deviations to be 0
                for i in 0..self.dev.len()
                {
                    self.dev[i] = 0.0;
                }

                // Reset the number of accumulated fragments
                self.nfragment = 0;
            },
        }
    }



    /// According to the accumulated sum, calculate the standard deviations of the descriptor for a given fragment
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn dev_calculation(&mut self)
    {
        match self.nfragment
        {
            // If none of the fragment has been accumulated, do nothing.
            // Remain the mean values to be the default 0, and remain the standard deviations to be the default 1.
            0 => (),

            // If only one fragment has been accumulated, remain the mean values to be the accumulated value,
            // and remain the standard deviations to be the default 1.
            1 => (),

            // If the accumulated number of fragment is larger than 2, calculate the standard deviations by average.
            _ =>
            {
                // Calculate the standard deviations by average
                for i in 0..self.dev.len()
                {
                    self.dev[i] = (self.dev[i] / self.nfragment as f64).sqrt();
                }
            },
        }
    }
}










impl GlobalNorm
{
    /// Initialize the global normalization parameters for all the fragments
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn initialize() -> Self
    {
        GlobalNorm
        {
            element_h_norm: FragmentNorm::initialize(-0.5309),
            element_b_norm: FragmentNorm::initialize(0.0),
            element_c_norm: FragmentNorm::initialize(-2.1385),
            element_n_norm: FragmentNorm::initialize(-3.1951),
            element_o_norm: FragmentNorm::initialize(-4.6977),
            element_f_norm: FragmentNorm::initialize(0.0),
            element_na_norm: FragmentNorm::initialize(0.0),
            element_mg_norm: FragmentNorm::initialize(0.0),
            element_si_norm: FragmentNorm::initialize(0.0),
            element_p_norm: FragmentNorm::initialize(0.0),
            element_s_norm: FragmentNorm::initialize(-3.6834),
            element_cl_norm: FragmentNorm::initialize(0.0),
            element_k_norm: FragmentNorm::initialize(0.0),
            element_ca_norm: FragmentNorm::initialize(0.0),
            element_v_norm: FragmentNorm::initialize(0.0),
            element_cr_norm: FragmentNorm::initialize(0.0),
            element_mn_norm: FragmentNorm::initialize(0.0),
            element_fe_norm: FragmentNorm::initialize(0.0),
            element_co_norm: FragmentNorm::initialize(0.0),
            element_ni_norm: FragmentNorm::initialize(0.0),
            element_cu_norm: FragmentNorm::initialize(0.0),
            element_zn_norm: FragmentNorm::initialize(0.0),
            element_as_norm: FragmentNorm::initialize(0.0),
            element_se_norm: FragmentNorm::initialize(0.0),
            element_br_norm: FragmentNorm::initialize(0.0),
            element_mo_norm: FragmentNorm::initialize(0.0),
            element_cd_norm: FragmentNorm::initialize(0.0),
            element_sn_norm: FragmentNorm::initialize(0.0),
            element_i_norm: FragmentNorm::initialize(0.0),

            amino_acid_gly_norm: FragmentNorm::initialize(-13.7625),
            amino_acid_ala_norm: FragmentNorm::initialize(-16.9628),
            amino_acid_val_norm: FragmentNorm::initialize(-23.3627),
            amino_acid_leu_norm: FragmentNorm::initialize(-26.5610),
            amino_acid_ile_norm: FragmentNorm::initialize(-26.5607),
            amino_acid_ser_norm: FragmentNorm::initialize(-21.6605),
            amino_acid_thr_norm: FragmentNorm::initialize(-24.8609),
            amino_acid_asp_norm: FragmentNorm::initialize(-28.2390),
            amino_acid_ash_norm: FragmentNorm::initialize(-28.4961),
            amino_acid_asn_norm: FragmentNorm::initialize(-27.5457),
            amino_acid_glu_norm: FragmentNorm::initialize(-31.4380),
            amino_acid_glh_norm: FragmentNorm::initialize(-31.6976),
            amino_acid_gln_norm: FragmentNorm::initialize(0.0),
            amino_acid_lys_norm: FragmentNorm::initialize(-30.4420),
            amino_acid_lyn_norm: FragmentNorm::initialize(0.0),
            amino_acid_arg_norm: FragmentNorm::initialize(-36.8741),
//            amino_acid_arn_norm: FragmentNorm::initialize(0.0),
            amino_acid_cys_norm: FragmentNorm::initialize(-20.6562),
            amino_acid_cyx_norm: FragmentNorm::initialize(0.0),
//            amino_acid_cym_norm: FragmentNorm::initialize(0.0),
            amino_acid_met_norm: FragmentNorm::initialize(-27.0437),
            amino_acid_hid_norm: FragmentNorm::initialize(0.0),
            amino_acid_hie_norm: FragmentNorm::initialize(0.0),
            amino_acid_hip_norm: FragmentNorm::initialize(-30.9140),
            amino_acid_phe_norm: FragmentNorm::initialize(0.0),
            amino_acid_tyr_norm: FragmentNorm::initialize(0.0),
            amino_acid_trp_norm: FragmentNorm::initialize(0.0),
            amino_acid_pro_norm: FragmentNorm::initialize(0.0),

            head_h_norm: FragmentNorm::initialize(0.0),
            tail_o_norm: FragmentNorm::initialize(0.0),
            tail_h_norm: FragmentNorm::initialize(0.0),
            head_ace_norm: FragmentNorm::initialize(-10.5674),
            tail_nme_norm: FragmentNorm::initialize(-7.4572),
            tail_nhe_norm: FragmentNorm::initialize(0.0),

            molecule_wat_norm: FragmentNorm::initialize(0.0),
        }
    }



    /// Save the global normalization parameters
    ///
    /// # Parameters
    /// ```
    /// dir: the directory for global normalization parameters saving
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn save<P: AsRef<Path> + Debug>(&self, dir: P)
    {
        // If directory 'dir' already exist, do nothing; otherwise, create the directory
        let dir_exist = fs::metadata(&dir);
        match dir_exist
        {
            Ok(_) => (),
            Err(_) => fs::create_dir_all(&dir).expect(&error_dir("creating", &dir)),
        }

        // Save the whole structure directly
        let norm_output_file: PathBuf = dir.as_ref().join("NormPara");
        save_file(&norm_output_file, 0, self).expect(&error_file("creating", &norm_output_file));
    }



    /// Load the global normalization parameters
    ///
    /// # Parameters
    /// ```
    /// dir: the directory for global normalization parameters loading
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn load<P: AsRef<Path> + Debug>(dir: P) -> Self
    {
        // Read the whole structure directly
        let norm_input_file: PathBuf = dir.as_ref().join("NormPara");
        let global_norm: GlobalNorm = load_file(&norm_input_file, 0).expect(&error_file("reading", &norm_input_file));

        global_norm
    }



    /// According to the input protein system, calculate the descriptors and accumulate to the mean values
    ///
    /// # Parameters
    /// ```
    /// s: the input protein system
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn mean_accumulation(&mut self, s: &ProteinSystem)
    {
        // For each fragment in the protein system
        for i in 0..s.fragment.len()
        {
            // Achieve the descriptors for Fragment i
            let (coord_within_rcut, _list_within_rcut, atomic_number_within_rcut): (Array2<f64>, Vec<usize>, Vec<usize>) = truncate_cluster(&s, i);
            let (descriptor, _gradient): (Vec<f64>, Array2<f64>) = get_descriptor(coord_within_rcut, atomic_number_within_rcut, s.fragment[i].natom);
            match &s.fragment[i].fragment_type
            {
                FragmentType::Atom(element) =>
                {
                    match element
                    {
                        Element::H => self.element_h_norm.mean_accumulation(&descriptor),
                        Element::C => self.element_c_norm.mean_accumulation(&descriptor),
                        Element::O => self.element_o_norm.mean_accumulation(&descriptor),
                        Element::N => self.element_n_norm.mean_accumulation(&descriptor),
                        Element::S => self.element_s_norm.mean_accumulation(&descriptor),
                        Element::P => self.element_p_norm.mean_accumulation(&descriptor),
                        Element::Na => self.element_na_norm.mean_accumulation(&descriptor),
                        Element::Cl => self.element_cl_norm.mean_accumulation(&descriptor),
                        Element::K => self.element_k_norm.mean_accumulation(&descriptor),
                        Element::Ca => self.element_ca_norm.mean_accumulation(&descriptor),
                        Element::Mg => self.element_mg_norm.mean_accumulation(&descriptor),
                        Element::F => self.element_f_norm.mean_accumulation(&descriptor),
                        Element::Fe => self.element_fe_norm.mean_accumulation(&descriptor),
                        Element::Cu => self.element_cu_norm.mean_accumulation(&descriptor),
                        Element::Zn => self.element_zn_norm.mean_accumulation(&descriptor),
                        Element::Mn => self.element_mn_norm.mean_accumulation(&descriptor),
                        Element::Mo => self.element_mo_norm.mean_accumulation(&descriptor),
                        Element::Co => self.element_co_norm.mean_accumulation(&descriptor),
                        Element::Cr => self.element_cr_norm.mean_accumulation(&descriptor),
                        Element::V => self.element_v_norm.mean_accumulation(&descriptor),
                        Element::Sn => self.element_sn_norm.mean_accumulation(&descriptor),
                        Element::Ni => self.element_ni_norm.mean_accumulation(&descriptor),
                        Element::Si => self.element_si_norm.mean_accumulation(&descriptor),
                        Element::Se => self.element_se_norm.mean_accumulation(&descriptor),
                        Element::I => self.element_i_norm.mean_accumulation(&descriptor),
                        Element::Br => self.element_br_norm.mean_accumulation(&descriptor),
                        Element::As => self.element_as_norm.mean_accumulation(&descriptor),
                        Element::B => self.element_b_norm.mean_accumulation(&descriptor),
                        Element::Cd => self.element_cd_norm.mean_accumulation(&descriptor),
                        _ => panic!("{}", error_non_bioelement_nn(element)),
                    }
                },

                FragmentType::Residue(amino_acid) =>
                {
                    match amino_acid
                    {
                        AminoAcid::GLY => self.amino_acid_gly_norm.mean_accumulation(&descriptor),
                        AminoAcid::ALA => self.amino_acid_ala_norm.mean_accumulation(&descriptor),
                        AminoAcid::VAL => self.amino_acid_val_norm.mean_accumulation(&descriptor),
                        AminoAcid::LEU => self.amino_acid_leu_norm.mean_accumulation(&descriptor),
                        AminoAcid::ILE => self.amino_acid_ile_norm.mean_accumulation(&descriptor),
                        AminoAcid::SER => self.amino_acid_ser_norm.mean_accumulation(&descriptor),
                        AminoAcid::THR => self.amino_acid_thr_norm.mean_accumulation(&descriptor),
                        AminoAcid::ASP => self.amino_acid_asp_norm.mean_accumulation(&descriptor),
                        AminoAcid::ASH => self.amino_acid_ash_norm.mean_accumulation(&descriptor),
                        AminoAcid::ASN => self.amino_acid_asn_norm.mean_accumulation(&descriptor),
                        AminoAcid::GLU => self.amino_acid_glu_norm.mean_accumulation(&descriptor),
                        AminoAcid::GLH => self.amino_acid_glh_norm.mean_accumulation(&descriptor),
                        AminoAcid::GLN => self.amino_acid_gln_norm.mean_accumulation(&descriptor),
                        AminoAcid::LYS => self.amino_acid_lys_norm.mean_accumulation(&descriptor),
                        AminoAcid::LYN => self.amino_acid_lyn_norm.mean_accumulation(&descriptor),
                        AminoAcid::ARG => self.amino_acid_arg_norm.mean_accumulation(&descriptor),
//                        AminoAcid::ARN => self.amino_acid_arn_norm.mean_accumulation(&descriptor),
                        AminoAcid::CYS => self.amino_acid_cys_norm.mean_accumulation(&descriptor),
                        AminoAcid::CYX => self.amino_acid_cyx_norm.mean_accumulation(&descriptor),
//                        AminoAcid::CYM => self.amino_acid_cym_norm.mean_accumulation(&descriptor),
                        AminoAcid::MET => self.amino_acid_met_norm.mean_accumulation(&descriptor),
                        AminoAcid::HID => self.amino_acid_hid_norm.mean_accumulation(&descriptor),
                        AminoAcid::HIE => self.amino_acid_hie_norm.mean_accumulation(&descriptor),
                        AminoAcid::HIP => self.amino_acid_hip_norm.mean_accumulation(&descriptor),
                        AminoAcid::PHE => self.amino_acid_phe_norm.mean_accumulation(&descriptor),
                        AminoAcid::TYR => self.amino_acid_tyr_norm.mean_accumulation(&descriptor),
                        AminoAcid::TRP => self.amino_acid_trp_norm.mean_accumulation(&descriptor),
                        AminoAcid::PRO => self.amino_acid_pro_norm.mean_accumulation(&descriptor),
                    }
                },

                FragmentType::Head(cap) =>
                {
                    match cap
                    {
                        Cap::H => self.head_h_norm.mean_accumulation(&descriptor),
                        Cap::ACE => self.head_ace_norm.mean_accumulation(&descriptor),
                        _ => panic!("{}", error_type("Head", &format!("{:?}", cap))),
                    }
                },

                FragmentType::Tail(cap) =>
                {
                    match cap
                    {
                        Cap::O => self.tail_o_norm.mean_accumulation(&descriptor),
                        Cap::H => self.tail_h_norm.mean_accumulation(&descriptor),
                        Cap::NME => self.tail_nme_norm.mean_accumulation(&descriptor),
                        Cap::NHE => self.tail_nhe_norm.mean_accumulation(&descriptor),
                        _ => panic!("{}", error_type("Tail", &format!("{:?}", cap))),
                    }
                },

                FragmentType::Molecule(molecule) =>
                {
                    match molecule
                    {
                        Molecule::WAT => self.molecule_wat_norm.mean_accumulation(&descriptor),
                    }
                },
            }
        }
    }



    /// According to the input protein system, calculate the descriptors and accumulate to the standard deviations
    ///
    /// # Parameters
    /// ```
    /// s: the input protein system
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn dev_accumulation(&mut self, s: &ProteinSystem)
    {
        // For each fragment in the protein system
        for i in 0..s.fragment.len()
        {
            // Achieve the descriptors for Fragment i
            let (coord_within_rcut, _list_within_rcut, atomic_number_within_rcut): (Array2<f64>, Vec<usize>, Vec<usize>) = truncate_cluster(&s, i);
            let (descriptor, _gradient): (Vec<f64>, Array2<f64>) = get_descriptor(coord_within_rcut, atomic_number_within_rcut, s.fragment[i].natom);
            match &s.fragment[i].fragment_type
            {
                FragmentType::Atom(element) =>
                {
                    match element
                    {
                        Element::H => self.element_h_norm.dev_accumulation(&descriptor),
                        Element::C => self.element_c_norm.dev_accumulation(&descriptor),
                        Element::O => self.element_o_norm.dev_accumulation(&descriptor),
                        Element::N => self.element_n_norm.dev_accumulation(&descriptor),
                        Element::S => self.element_s_norm.dev_accumulation(&descriptor),
                        Element::P => self.element_p_norm.dev_accumulation(&descriptor),
                        Element::Na => self.element_na_norm.dev_accumulation(&descriptor),
                        Element::Cl => self.element_cl_norm.dev_accumulation(&descriptor),
                        Element::K => self.element_k_norm.dev_accumulation(&descriptor),
                        Element::Ca => self.element_ca_norm.dev_accumulation(&descriptor),
                        Element::Mg => self.element_mg_norm.dev_accumulation(&descriptor),
                        Element::F => self.element_f_norm.dev_accumulation(&descriptor),
                        Element::Fe => self.element_fe_norm.dev_accumulation(&descriptor),
                        Element::Cu => self.element_cu_norm.dev_accumulation(&descriptor),
                        Element::Zn => self.element_zn_norm.dev_accumulation(&descriptor),
                        Element::Mn => self.element_mn_norm.dev_accumulation(&descriptor),
                        Element::Mo => self.element_mo_norm.dev_accumulation(&descriptor),
                        Element::Co => self.element_co_norm.dev_accumulation(&descriptor),
                        Element::Cr => self.element_cr_norm.dev_accumulation(&descriptor),
                        Element::V => self.element_v_norm.dev_accumulation(&descriptor),
                        Element::Sn => self.element_sn_norm.dev_accumulation(&descriptor),
                        Element::Ni => self.element_ni_norm.dev_accumulation(&descriptor),
                        Element::Si => self.element_si_norm.dev_accumulation(&descriptor),
                        Element::Se => self.element_se_norm.dev_accumulation(&descriptor),
                        Element::I => self.element_i_norm.dev_accumulation(&descriptor),
                        Element::Br => self.element_br_norm.dev_accumulation(&descriptor),
                        Element::As => self.element_as_norm.dev_accumulation(&descriptor),
                        Element::B => self.element_b_norm.dev_accumulation(&descriptor),
                        Element::Cd => self.element_cd_norm.dev_accumulation(&descriptor),
                        _ => panic!("{}", error_non_bioelement_nn(element)),
                    }
                },

                FragmentType::Residue(amino_acid) =>
                {
                    match amino_acid
                    {
                        AminoAcid::GLY => self.amino_acid_gly_norm.dev_accumulation(&descriptor),
                        AminoAcid::ALA => self.amino_acid_ala_norm.dev_accumulation(&descriptor),
                        AminoAcid::VAL => self.amino_acid_val_norm.dev_accumulation(&descriptor),
                        AminoAcid::LEU => self.amino_acid_leu_norm.dev_accumulation(&descriptor),
                        AminoAcid::ILE => self.amino_acid_ile_norm.dev_accumulation(&descriptor),
                        AminoAcid::SER => self.amino_acid_ser_norm.dev_accumulation(&descriptor),
                        AminoAcid::THR => self.amino_acid_thr_norm.dev_accumulation(&descriptor),
                        AminoAcid::ASP => self.amino_acid_asp_norm.dev_accumulation(&descriptor),
                        AminoAcid::ASH => self.amino_acid_ash_norm.dev_accumulation(&descriptor),
                        AminoAcid::ASN => self.amino_acid_asn_norm.dev_accumulation(&descriptor),
                        AminoAcid::GLU => self.amino_acid_glu_norm.dev_accumulation(&descriptor),
                        AminoAcid::GLH => self.amino_acid_glh_norm.dev_accumulation(&descriptor),
                        AminoAcid::GLN => self.amino_acid_gln_norm.dev_accumulation(&descriptor),
                        AminoAcid::LYS => self.amino_acid_lys_norm.dev_accumulation(&descriptor),
                        AminoAcid::LYN => self.amino_acid_lyn_norm.dev_accumulation(&descriptor),
                        AminoAcid::ARG => self.amino_acid_arg_norm.dev_accumulation(&descriptor),
//                        AminoAcid::ARN => self.amino_acid_arn_norm.dev_accumulation(&descriptor),
                        AminoAcid::CYS => self.amino_acid_cys_norm.dev_accumulation(&descriptor),
                        AminoAcid::CYX => self.amino_acid_cyx_norm.dev_accumulation(&descriptor),
//                        AminoAcid::CYM => self.amino_acid_cym_norm.dev_accumulation(&descriptor),
                        AminoAcid::MET => self.amino_acid_met_norm.dev_accumulation(&descriptor),
                        AminoAcid::HID => self.amino_acid_hid_norm.dev_accumulation(&descriptor),
                        AminoAcid::HIE => self.amino_acid_hie_norm.dev_accumulation(&descriptor),
                        AminoAcid::HIP => self.amino_acid_hip_norm.dev_accumulation(&descriptor),
                        AminoAcid::PHE => self.amino_acid_phe_norm.dev_accumulation(&descriptor),
                        AminoAcid::TYR => self.amino_acid_tyr_norm.dev_accumulation(&descriptor),
                        AminoAcid::TRP => self.amino_acid_trp_norm.dev_accumulation(&descriptor),
                        AminoAcid::PRO => self.amino_acid_pro_norm.dev_accumulation(&descriptor),
                    }
                },

                FragmentType::Head(cap) =>
                {
                    match cap
                    {
                        Cap::H => self.head_h_norm.dev_accumulation(&descriptor),
                        Cap::ACE => self.head_ace_norm.dev_accumulation(&descriptor),
                        _ => panic!("{}", error_type("Head", &format!("{:?}", cap))),
                    }
                },

                FragmentType::Tail(cap) =>
                {
                    match cap
                    {
                        Cap::O => self.tail_o_norm.dev_accumulation(&descriptor),
                        Cap::H => self.tail_h_norm.dev_accumulation(&descriptor),
                        Cap::NME => self.tail_nme_norm.dev_accumulation(&descriptor),
                        Cap::NHE => self.tail_nhe_norm.dev_accumulation(&descriptor),
                        _ => panic!("{}", error_type("Tail", &format!("{:?}", cap))),
                    }
                },

                FragmentType::Molecule(molecule) =>
                {
                    match molecule
                    {
                        Molecule::WAT => self.molecule_wat_norm.dev_accumulation(&descriptor),
                    }
                },
            }
        }
    }



    /// According to the accumulated sum, calculate the mean values of descriptor for all the fragments
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn mean_calculation(&mut self)
    {
        self.element_h_norm.mean_calculation();
        self.element_b_norm.mean_calculation();
        self.element_c_norm.mean_calculation();
        self.element_n_norm.mean_calculation();
        self.element_o_norm.mean_calculation();
        self.element_f_norm.mean_calculation();
        self.element_na_norm.mean_calculation();
        self.element_mg_norm.mean_calculation();
        self.element_si_norm.mean_calculation();
        self.element_p_norm.mean_calculation();
        self.element_s_norm.mean_calculation();
        self.element_cl_norm.mean_calculation();
        self.element_k_norm.mean_calculation();
        self.element_ca_norm.mean_calculation();
        self.element_v_norm.mean_calculation();
        self.element_cr_norm.mean_calculation();
        self.element_mn_norm.mean_calculation();
        self.element_fe_norm.mean_calculation();
        self.element_co_norm.mean_calculation();
        self.element_ni_norm.mean_calculation();
        self.element_cu_norm.mean_calculation();
        self.element_zn_norm.mean_calculation();
        self.element_as_norm.mean_calculation();
        self.element_se_norm.mean_calculation();
        self.element_br_norm.mean_calculation();
        self.element_mo_norm.mean_calculation();
        self.element_cd_norm.mean_calculation();
        self.element_sn_norm.mean_calculation();
        self.element_i_norm.mean_calculation();

        self.amino_acid_gly_norm.mean_calculation();
        self.amino_acid_ala_norm.mean_calculation();
        self.amino_acid_val_norm.mean_calculation();
        self.amino_acid_leu_norm.mean_calculation();
        self.amino_acid_ile_norm.mean_calculation();
        self.amino_acid_ser_norm.mean_calculation();
        self.amino_acid_thr_norm.mean_calculation();
        self.amino_acid_asp_norm.mean_calculation();
        self.amino_acid_ash_norm.mean_calculation();
        self.amino_acid_asn_norm.mean_calculation();
        self.amino_acid_glu_norm.mean_calculation();
        self.amino_acid_glh_norm.mean_calculation();
        self.amino_acid_gln_norm.mean_calculation();
        self.amino_acid_lys_norm.mean_calculation();
        self.amino_acid_lyn_norm.mean_calculation();
        self.amino_acid_arg_norm.mean_calculation();
//        self.amino_acid_arn_norm.mean_calculation();
        self.amino_acid_cys_norm.mean_calculation();
        self.amino_acid_cyx_norm.mean_calculation();
//        self.amino_acid_cym_norm.mean_calculation();
        self.amino_acid_met_norm.mean_calculation();
        self.amino_acid_hid_norm.mean_calculation();
        self.amino_acid_hie_norm.mean_calculation();
        self.amino_acid_hip_norm.mean_calculation();
        self.amino_acid_phe_norm.mean_calculation();
        self.amino_acid_tyr_norm.mean_calculation();
        self.amino_acid_trp_norm.mean_calculation();
        self.amino_acid_pro_norm.mean_calculation();

        self.head_h_norm.mean_calculation();
        self.tail_o_norm.mean_calculation();
        self.tail_h_norm.mean_calculation();
        self.head_ace_norm.mean_calculation();
        self.tail_nme_norm.mean_calculation();
        self.tail_nhe_norm.mean_calculation();

        self.molecule_wat_norm.mean_calculation();
    }



    /// According to the accumulated sum, calculate the standard deviations of descriptor for all the fragments
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn dev_calculation(&mut self)
    {
        self.element_h_norm.dev_calculation();
        self.element_b_norm.dev_calculation();
        self.element_c_norm.dev_calculation();
        self.element_n_norm.dev_calculation();
        self.element_o_norm.dev_calculation();
        self.element_f_norm.dev_calculation();
        self.element_na_norm.dev_calculation();
        self.element_mg_norm.dev_calculation();
        self.element_si_norm.dev_calculation();
        self.element_p_norm.dev_calculation();
        self.element_s_norm.dev_calculation();
        self.element_cl_norm.dev_calculation();
        self.element_k_norm.dev_calculation();
        self.element_ca_norm.dev_calculation();
        self.element_v_norm.dev_calculation();
        self.element_cr_norm.dev_calculation();
        self.element_mn_norm.dev_calculation();
        self.element_fe_norm.dev_calculation();
        self.element_co_norm.dev_calculation();
        self.element_ni_norm.dev_calculation();
        self.element_cu_norm.dev_calculation();
        self.element_zn_norm.dev_calculation();
        self.element_as_norm.dev_calculation();
        self.element_se_norm.dev_calculation();
        self.element_br_norm.dev_calculation();
        self.element_mo_norm.dev_calculation();
        self.element_cd_norm.dev_calculation();
        self.element_sn_norm.dev_calculation();
        self.element_i_norm.dev_calculation();

        self.amino_acid_gly_norm.dev_calculation();
        self.amino_acid_ala_norm.dev_calculation();
        self.amino_acid_val_norm.dev_calculation();
        self.amino_acid_leu_norm.dev_calculation();
        self.amino_acid_ile_norm.dev_calculation();
        self.amino_acid_ser_norm.dev_calculation();
        self.amino_acid_thr_norm.dev_calculation();
        self.amino_acid_asp_norm.dev_calculation();
        self.amino_acid_ash_norm.dev_calculation();
        self.amino_acid_asn_norm.dev_calculation();
        self.amino_acid_glu_norm.dev_calculation();
        self.amino_acid_glh_norm.dev_calculation();
        self.amino_acid_gln_norm.dev_calculation();
        self.amino_acid_lys_norm.dev_calculation();
        self.amino_acid_lyn_norm.dev_calculation();
        self.amino_acid_arg_norm.dev_calculation();
//        self.amino_acid_arn_norm.dev_calculation();
        self.amino_acid_cys_norm.dev_calculation();
        self.amino_acid_cyx_norm.dev_calculation();
//        self.amino_acid_cym_norm.dev_calculation();
        self.amino_acid_met_norm.dev_calculation();
        self.amino_acid_hid_norm.dev_calculation();
        self.amino_acid_hie_norm.dev_calculation();
        self.amino_acid_hip_norm.dev_calculation();
        self.amino_acid_phe_norm.dev_calculation();
        self.amino_acid_tyr_norm.dev_calculation();
        self.amino_acid_trp_norm.dev_calculation();
        self.amino_acid_pro_norm.dev_calculation();

        self.head_h_norm.dev_calculation();
        self.tail_o_norm.dev_calculation();
        self.tail_h_norm.dev_calculation();
        self.head_ace_norm.dev_calculation();
        self.tail_nme_norm.dev_calculation();
        self.tail_nhe_norm.dev_calculation();

        self.molecule_wat_norm.dev_calculation();
    }



    /// Calculate mean values and standard deviations of descriptor
    /// for all the fragments according to a series of data
    ///
    /// # Parameters
    /// ```
    /// dir: the input directories containing the data for the calculations
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn from_data(dir: &Vec<PathBuf>) -> Self
    {
        // Get a initialized global_norm with 0 mean values and 1 standard deviations
        let mut global_norm: GlobalNorm = GlobalNorm::initialize();

        // Traverse the input parent directories for all the leaf directories
        let leaf_dir: Vec<PathBuf> = traverse_leaf_dir(&dir);

        // Accumulate all the structural descriptors to mean values
        for n in 0..leaf_dir.len()
        {
            // Obtain the filenames
            let str_file: PathBuf = leaf_dir[n].join("str.pdb");
            let data_file: PathBuf = leaf_dir[n].join("data.bin");
            // Load the files
            let mut s: ProteinSystem = ProteinSystem::read_pdb(&str_file);
            s = s.to_atom_fragment();
            let data: DataSaved = load_file(&data_file, 0).expect(&error_file("reading", &data_file));
            // Assert if the protein system and the DFT data is matching
            assert_eq!(s.natom, data.natom);
            assert_eq!(s.atom_type, data.atom_type);
            // Extract the atomic coordinates from the data
            let coord: Array3<f64> = Array3::from_shape_vec((data.nstruct, data.natom, 3), data.coord).expect(&error_none_value("data.coord"));

            // Accumulate mean values for all the structures
            for i in 0..data.nstruct                // For each structure
            {
                // Copy the atomic coordinates to the ProteinSystem
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
                global_norm.mean_accumulation(&s);
            }
        }

        // Calculate the mean values
        global_norm.mean_calculation();

        // Accumulate all the structural descriptors to standard deviations
        for n in 0..leaf_dir.len()
        {
            // Obtain the filenames
            let str_file: PathBuf = leaf_dir[n].join("str.pdb");
            let data_file: PathBuf = leaf_dir[n].join("data.bin");
            // Load the files
            let mut s: ProteinSystem = ProteinSystem::read_pdb(&str_file);
            s = s.to_atom_fragment();
            let data: DataSaved = load_file(&data_file, 0).expect(&error_file("reading", &data_file));
            // Assert if the protein system and the DFT data is matching
            assert_eq!(s.natom, data.natom);
            assert_eq!(s.atom_type, data.atom_type);
            // Extract the atomic coordinates from the data
            let coord: Array3<f64> = Array3::from_shape_vec((data.nstruct, data.natom, 3), data.coord).expect(&error_none_value("data.coord"));

            // Accumulate standard deviations for all the structures
            for i in 0..data.nstruct                // For each structure
            {
                // Copy the atomic coordinates to the ProteinSystem
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
                global_norm.dev_accumulation(&s);
            }
        }

        // Calculate the standard deviations
        global_norm.dev_calculation();

        global_norm
    }



    /// Get the normalization parameters for a given fragment
    ///
    /// # Parameters
    /// ```
    /// fragment_type: the input fragment type
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn get_fragment_norm(&self, fragment_type: &FragmentType) -> &FragmentNorm
    {
        match fragment_type
        {
            FragmentType::Atom(element) =>
            {
                match element
                {
                    Element::H => &self.element_h_norm,
                    Element::C => &self.element_c_norm,
                    Element::O => &self.element_o_norm,
                    Element::N => &self.element_n_norm,
                    Element::S => &self.element_s_norm,
                    Element::P => &self.element_p_norm,
                    Element::Na => &self.element_na_norm,
                    Element::Cl => &self.element_cl_norm,
                    Element::K => &self.element_k_norm,
                    Element::Ca => &self.element_ca_norm,
                    Element::Mg => &self.element_mg_norm,
                    Element::F => &self.element_f_norm,
                    Element::Fe => &self.element_fe_norm,
                    Element::Cu => &self.element_cu_norm,
                    Element::Zn => &self.element_zn_norm,
                    Element::Mn => &self.element_mn_norm,
                    Element::Mo => &self.element_mo_norm,
                    Element::Co => &self.element_co_norm,
                    Element::Cr => &self.element_cr_norm,
                    Element::V => &self.element_v_norm,
                    Element::Sn => &self.element_sn_norm,
                    Element::Ni => &self.element_ni_norm,
                    Element::Si => &self.element_si_norm,
                    Element::Se => &self.element_se_norm,
                    Element::I => &self.element_i_norm,
                    Element::Br => &self.element_br_norm,
                    Element::As => &self.element_as_norm,
                    Element::B => &self.element_b_norm,
                    Element::Cd => &self.element_cd_norm,
                    _ => panic!("{}", error_non_bioelement_nn(element)),
                }
            },

            FragmentType::Residue(amino_acid) =>
            {
                match amino_acid
                {
                    AminoAcid::GLY => &self.amino_acid_gly_norm,
                    AminoAcid::ALA => &self.amino_acid_ala_norm,
                    AminoAcid::VAL => &self.amino_acid_val_norm,
                    AminoAcid::LEU => &self.amino_acid_leu_norm,
                    AminoAcid::ILE => &self.amino_acid_ile_norm,
                    AminoAcid::SER => &self.amino_acid_ser_norm,
                    AminoAcid::THR => &self.amino_acid_thr_norm,
                    AminoAcid::ASP => &self.amino_acid_asp_norm,
                    AminoAcid::ASH => &self.amino_acid_ash_norm,
                    AminoAcid::ASN => &self.amino_acid_asn_norm,
                    AminoAcid::GLU => &self.amino_acid_glu_norm,
                    AminoAcid::GLH => &self.amino_acid_glh_norm,
                    AminoAcid::GLN => &self.amino_acid_gln_norm,
                    AminoAcid::LYS => &self.amino_acid_lys_norm,
                    AminoAcid::LYN => &self.amino_acid_lyn_norm,
                    AminoAcid::ARG => &self.amino_acid_arg_norm,
//                    AminoAcid::ARN => &self.amino_acid_arn_norm,
                    AminoAcid::CYS => &self.amino_acid_cys_norm,
                    AminoAcid::CYX => &self.amino_acid_cyx_norm,
//                    AminoAcid::CYM => &self.amino_acid_cym_norm,
                    AminoAcid::MET => &self.amino_acid_met_norm,
                    AminoAcid::HID => &self.amino_acid_hid_norm,
                    AminoAcid::HIE => &self.amino_acid_hie_norm,
                    AminoAcid::HIP => &self.amino_acid_hip_norm,
                    AminoAcid::PHE => &self.amino_acid_phe_norm,
                    AminoAcid::TYR => &self.amino_acid_tyr_norm,
                    AminoAcid::TRP => &self.amino_acid_trp_norm,
                    AminoAcid::PRO => &self.amino_acid_pro_norm,
                }
            },

            FragmentType::Head(cap) =>
            {
                match cap
                {
                    Cap::H => &self.head_h_norm,
                    Cap::ACE => &self.head_ace_norm,
                    _ => panic!("{}", error_type("Head", &format!("{:?}", cap))),
                }
            },

            FragmentType::Tail(cap) =>
            {
                match cap
                {
                    Cap::O => &self.tail_o_norm,
                    Cap::H => &self.tail_h_norm,
                    Cap::NME => &self.tail_nme_norm,
                    Cap::NHE => &self.tail_nhe_norm,
                    _ => panic!("{}", error_type("Tail", &format!("{:?}", cap))),
                }
            },

            FragmentType::Molecule(molecule) =>
            {
                match molecule
                {
                    Molecule::WAT => &self.molecule_wat_norm,
                }
            },
        }
    }
}










