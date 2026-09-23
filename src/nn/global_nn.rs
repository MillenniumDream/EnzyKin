//! Initialization, saving, and loading of the neural network for each fragment in protein system
use std::fs;
use std::path::Path;
use std::fmt::Debug;
use crate::common::constants::{Element, Device};
use crate::common::error::*;
use crate::nn::fragment_definition::{AminoAcid, Molecule, Cap, FragmentType};
use crate::nn::descriptor::N_DES;
use dfdx::shapes::Rank0;
use dfdx::tensor::{Tensor, ZerosTensor, Gradients, NoneTape, OwnedTape, Trace};
use dfdx::tensor_ops::{Backward, AdamConfig};
use dfdx::nn::{BuildModule, SaveToSafetensors, LoadFromSafetensors, ZeroGrads};
use dfdx::nn::modules::{Linear, Tanh};
use dfdx::optim::{Adam, Optimizer};





// Define the NN modules
type NNSmall =
(
    (Linear<N_DES, N_DES, f64, Device>, Tanh),
    (Linear<N_DES, 256, f64, Device>, Tanh),
    (Linear<256, 64, f64, Device>, Tanh),
    (Linear<64, 16, f64, Device>, Tanh),
    (Linear<16, 4, f64, Device>, Tanh),
    Linear<4, 1, f64, Device>,
);

type NNMiddle =
(
    (Linear<N_DES, N_DES, f64, Device>, Tanh),
    (Linear<N_DES, N_DES, f64, Device>, Tanh),
    (Linear<N_DES, 256, f64, Device>, Tanh),
    (Linear<256, 64, f64, Device>, Tanh),
    (Linear<64, 16, f64, Device>, Tanh),
    (Linear<16, 4, f64, Device>, Tanh, Linear<4, 1, f64, Device>),
);

type NNLarge =
(
    (Linear<N_DES, N_DES, f64, Device>, Tanh),
    (Linear<N_DES, N_DES, f64, Device>, Tanh),
    (Linear<N_DES, N_DES, f64, Device>, Tanh),
    (Linear<N_DES, N_DES, f64, Device>, Tanh, Linear<N_DES, 256, f64, Device>, Tanh),
    (Linear<256, 64, f64, Device>, Tanh, Linear<64, 16, f64, Device>, Tanh),
    (Linear<16, 4, f64, Device>, Tanh, Linear<4, 1, f64, Device>),
);

pub enum NN<'a>
{
    Small(&'a NNSmall),
    Middle(&'a NNMiddle),
    Large(&'a NNLarge),
}

pub enum NNMut<'a>
{
    Small(&'a mut NNSmall),
    Middle(&'a mut NNMiddle),
    Large(&'a mut NNLarge),
}










/// The global neural network for the protein system, containing all the sub-nn for the framents (i.e. atoms, amino acids, and molecules)
///
/// # Fields
/// ```
/// ```
pub struct GlobalNN
{
    pub element_h_nn: NNLarge,
    pub element_b_nn: NNMiddle,
    pub element_c_nn: NNLarge,
    pub element_n_nn: NNLarge,
    pub element_o_nn: NNLarge,
    pub element_f_nn: NNMiddle,
    pub element_na_nn: NNMiddle,
    pub element_mg_nn: NNMiddle,
    pub element_si_nn: NNMiddle,
    pub element_p_nn: NNMiddle,
    pub element_s_nn: NNLarge,
    pub element_cl_nn: NNMiddle,
    pub element_k_nn: NNMiddle,
    pub element_ca_nn: NNMiddle,
    pub element_v_nn: NNMiddle,
    pub element_cr_nn: NNMiddle,
    pub element_mn_nn: NNMiddle,
    pub element_fe_nn: NNMiddle,
    pub element_co_nn: NNMiddle,
    pub element_ni_nn: NNMiddle,
    pub element_cu_nn: NNMiddle,
    pub element_zn_nn: NNMiddle,
    pub element_as_nn: NNMiddle,
    pub element_se_nn: NNMiddle,
    pub element_br_nn: NNMiddle,
    pub element_mo_nn: NNMiddle,
    pub element_cd_nn: NNMiddle,
    pub element_sn_nn: NNMiddle,
    pub element_i_nn: NNMiddle,

    pub amino_acid_gly_nn: NNLarge,
    pub amino_acid_ala_nn: NNLarge,
    pub amino_acid_val_nn: NNLarge,
    pub amino_acid_leu_nn: NNLarge,
    pub amino_acid_ile_nn: NNLarge,
    pub amino_acid_ser_nn: NNLarge,
    pub amino_acid_thr_nn: NNLarge,
    pub amino_acid_asp_nn: NNLarge,
    pub amino_acid_ash_nn: NNLarge,
    pub amino_acid_asn_nn: NNLarge,
    pub amino_acid_glu_nn: NNLarge,
    pub amino_acid_glh_nn: NNLarge,
    pub amino_acid_gln_nn: NNLarge,
    pub amino_acid_lys_nn: NNLarge,
    pub amino_acid_lyn_nn: NNLarge,
    pub amino_acid_arg_nn: NNLarge,
//    pub amino_acid_arn_nn: NNLarge,
    pub amino_acid_cys_nn: NNLarge,
    pub amino_acid_cyx_nn: NNLarge,
//    pub amino_acid_cym_nn: NNLarge,
    pub amino_acid_met_nn: NNLarge,
    pub amino_acid_hid_nn: NNLarge,
    pub amino_acid_hie_nn: NNLarge,
    pub amino_acid_hip_nn: NNLarge,
    pub amino_acid_phe_nn: NNLarge,
    pub amino_acid_tyr_nn: NNLarge,
    pub amino_acid_trp_nn: NNLarge,
    pub amino_acid_pro_nn: NNLarge,

    pub head_h_nn: NNMiddle,
    pub tail_o_nn: NNMiddle,
    pub tail_h_nn: NNMiddle,
    pub head_ace_nn: NNLarge,
    pub tail_nme_nn: NNLarge,
    pub tail_nhe_nn: NNLarge,

    pub molecule_wat_nn: NNSmall,
}





/// The global Adam optimizer for the global NN
///
/// # Fields
/// ```
/// ```
pub struct GlobalAdam
{
    element_h_adam: Adam<NNLarge, f64, Device>,
    element_b_adam: Adam<NNMiddle, f64, Device>,
    element_c_adam: Adam<NNLarge, f64, Device>,
    element_n_adam: Adam<NNLarge, f64, Device>,
    element_o_adam: Adam<NNLarge, f64, Device>,
    element_f_adam: Adam<NNMiddle, f64, Device>,
    element_na_adam: Adam<NNMiddle, f64, Device>,
    element_mg_adam: Adam<NNMiddle, f64, Device>,
    element_si_adam: Adam<NNMiddle, f64, Device>,
    element_p_adam: Adam<NNMiddle, f64, Device>,
    element_s_adam: Adam<NNLarge, f64, Device>,
    element_cl_adam: Adam<NNMiddle, f64, Device>,
    element_k_adam: Adam<NNMiddle, f64, Device>,
    element_ca_adam: Adam<NNMiddle, f64, Device>,
    element_v_adam: Adam<NNMiddle, f64, Device>,
    element_cr_adam: Adam<NNMiddle, f64, Device>,
    element_mn_adam: Adam<NNMiddle, f64, Device>,
    element_fe_adam: Adam<NNMiddle, f64, Device>,
    element_co_adam: Adam<NNMiddle, f64, Device>,
    element_ni_adam: Adam<NNMiddle, f64, Device>,
    element_cu_adam: Adam<NNMiddle, f64, Device>,
    element_zn_adam: Adam<NNMiddle, f64, Device>,
    element_as_adam: Adam<NNMiddle, f64, Device>,
    element_se_adam: Adam<NNMiddle, f64, Device>,
    element_br_adam: Adam<NNMiddle, f64, Device>,
    element_mo_adam: Adam<NNMiddle, f64, Device>,
    element_cd_adam: Adam<NNMiddle, f64, Device>,
    element_sn_adam: Adam<NNMiddle, f64, Device>,
    element_i_adam: Adam<NNMiddle, f64, Device>,

    amino_acid_gly_adam: Adam<NNLarge, f64, Device>,
    amino_acid_ala_adam: Adam<NNLarge, f64, Device>,
    amino_acid_val_adam: Adam<NNLarge, f64, Device>,
    amino_acid_leu_adam: Adam<NNLarge, f64, Device>,
    amino_acid_ile_adam: Adam<NNLarge, f64, Device>,
    amino_acid_ser_adam: Adam<NNLarge, f64, Device>,
    amino_acid_thr_adam: Adam<NNLarge, f64, Device>,
    amino_acid_asp_adam: Adam<NNLarge, f64, Device>,
    amino_acid_ash_adam: Adam<NNLarge, f64, Device>,
    amino_acid_asn_adam: Adam<NNLarge, f64, Device>,
    amino_acid_glu_adam: Adam<NNLarge, f64, Device>,
    amino_acid_glh_adam: Adam<NNLarge, f64, Device>,
    amino_acid_gln_adam: Adam<NNLarge, f64, Device>,
    amino_acid_lys_adam: Adam<NNLarge, f64, Device>,
    amino_acid_lyn_adam: Adam<NNLarge, f64, Device>,
    amino_acid_arg_adam: Adam<NNLarge, f64, Device>,
//    amino_acid_arn_adam: Adam<NNLarge, f64, Device>,
    amino_acid_cys_adam: Adam<NNLarge, f64, Device>,
    amino_acid_cyx_adam: Adam<NNLarge, f64, Device>,
//    amino_acid_cym_adam: Adam<NNLarge, f64, Device>,
    amino_acid_met_adam: Adam<NNLarge, f64, Device>,
    amino_acid_hid_adam: Adam<NNLarge, f64, Device>,
    amino_acid_hie_adam: Adam<NNLarge, f64, Device>,
    amino_acid_hip_adam: Adam<NNLarge, f64, Device>,
    amino_acid_phe_adam: Adam<NNLarge, f64, Device>,
    amino_acid_tyr_adam: Adam<NNLarge, f64, Device>,
    amino_acid_trp_adam: Adam<NNLarge, f64, Device>,
    amino_acid_pro_adam: Adam<NNLarge, f64, Device>,

    head_h_adam: Adam<NNMiddle, f64, Device>,
    tail_o_adam: Adam<NNMiddle, f64, Device>,
    tail_h_adam: Adam<NNMiddle, f64, Device>,
    head_ace_adam: Adam<NNLarge, f64, Device>,
    tail_nme_adam: Adam<NNLarge, f64, Device>,
    tail_nhe_adam: Adam<NNLarge, f64, Device>,

    molecule_wat_adam: Adam<NNSmall, f64, Device>,
}





/// Controling the update of the global neural network
///
/// # Fields
/// ```
/// ```
#[derive(Clone)]
pub struct NnUpdate
{
    pub update_element_h_nn: bool,
    pub update_element_b_nn: bool,
    pub update_element_c_nn: bool,
    pub update_element_n_nn: bool,
    pub update_element_o_nn: bool,
    pub update_element_f_nn: bool,
    pub update_element_na_nn: bool,
    pub update_element_mg_nn: bool,
    pub update_element_si_nn: bool,
    pub update_element_p_nn: bool,
    pub update_element_s_nn: bool,
    pub update_element_cl_nn: bool,
    pub update_element_k_nn: bool,
    pub update_element_ca_nn: bool,
    pub update_element_v_nn: bool,
    pub update_element_cr_nn: bool,
    pub update_element_mn_nn: bool,
    pub update_element_fe_nn: bool,
    pub update_element_co_nn: bool,
    pub update_element_ni_nn: bool,
    pub update_element_cu_nn: bool,
    pub update_element_zn_nn: bool,
    pub update_element_as_nn: bool,
    pub update_element_se_nn: bool,
    pub update_element_br_nn: bool,
    pub update_element_mo_nn: bool,
    pub update_element_cd_nn: bool,
    pub update_element_sn_nn: bool,
    pub update_element_i_nn: bool,

    pub update_amino_acid_gly_nn: bool,
    pub update_amino_acid_ala_nn: bool,
    pub update_amino_acid_val_nn: bool,
    pub update_amino_acid_leu_nn: bool,
    pub update_amino_acid_ile_nn: bool,
    pub update_amino_acid_ser_nn: bool,
    pub update_amino_acid_thr_nn: bool,
    pub update_amino_acid_asp_nn: bool,
    pub update_amino_acid_ash_nn: bool,
    pub update_amino_acid_asn_nn: bool,
    pub update_amino_acid_glu_nn: bool,
    pub update_amino_acid_glh_nn: bool,
    pub update_amino_acid_gln_nn: bool,
    pub update_amino_acid_lys_nn: bool,
    pub update_amino_acid_lyn_nn: bool,
    pub update_amino_acid_arg_nn: bool,
//    pub update_amino_acid_arn_nn: bool,
    pub update_amino_acid_cys_nn: bool,
    pub update_amino_acid_cyx_nn: bool,
//    pub update_amino_acid_cym_nn: bool,
    pub update_amino_acid_met_nn: bool,
    pub update_amino_acid_hid_nn: bool,
    pub update_amino_acid_hie_nn: bool,
    pub update_amino_acid_hip_nn: bool,
    pub update_amino_acid_phe_nn: bool,
    pub update_amino_acid_tyr_nn: bool,
    pub update_amino_acid_trp_nn: bool,
    pub update_amino_acid_pro_nn: bool,

    pub update_head_h_nn: bool,
    pub update_tail_o_nn: bool,
    pub update_tail_h_nn: bool,
    pub update_head_ace_nn: bool,
    pub update_tail_nme_nn: bool,
    pub update_tail_nhe_nn: bool,

    pub update_molecule_wat_nn: bool,
}










impl GlobalNN
{
    /// Construct a new global NN for the protein system
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn new() -> Self
    {
        // Define a Device (CPU or Cuda) to build NNs
        let dev: Device = Device::seed_from_u64(1314);

        // Build NN for each FragmentType
        let element_h_nn: NNLarge = BuildModule::build(&dev);
        let element_b_nn: NNMiddle = BuildModule::build(&dev);
        let element_c_nn: NNLarge = BuildModule::build(&dev);
        let element_n_nn: NNLarge = BuildModule::build(&dev);
        let element_o_nn: NNLarge = BuildModule::build(&dev);
        let element_f_nn: NNMiddle = BuildModule::build(&dev);
        let element_na_nn: NNMiddle = BuildModule::build(&dev);
        let element_mg_nn: NNMiddle = BuildModule::build(&dev);
        let element_si_nn: NNMiddle = BuildModule::build(&dev);
        let element_p_nn: NNMiddle = BuildModule::build(&dev);
        let element_s_nn: NNLarge = BuildModule::build(&dev);
        let element_cl_nn: NNMiddle = BuildModule::build(&dev);
        let element_k_nn: NNMiddle = BuildModule::build(&dev);
        let element_ca_nn: NNMiddle = BuildModule::build(&dev);
        let element_v_nn: NNMiddle = BuildModule::build(&dev);
        let element_cr_nn: NNMiddle = BuildModule::build(&dev);
        let element_mn_nn: NNMiddle = BuildModule::build(&dev);
        let element_fe_nn: NNMiddle = BuildModule::build(&dev);
        let element_co_nn: NNMiddle = BuildModule::build(&dev);
        let element_ni_nn: NNMiddle = BuildModule::build(&dev);
        let element_cu_nn: NNMiddle = BuildModule::build(&dev);
        let element_zn_nn: NNMiddle = BuildModule::build(&dev);
        let element_as_nn: NNMiddle = BuildModule::build(&dev);
        let element_se_nn: NNMiddle = BuildModule::build(&dev);
        let element_br_nn: NNMiddle = BuildModule::build(&dev);
        let element_mo_nn: NNMiddle = BuildModule::build(&dev);
        let element_cd_nn: NNMiddle = BuildModule::build(&dev);
        let element_sn_nn: NNMiddle = BuildModule::build(&dev);
        let element_i_nn: NNMiddle = BuildModule::build(&dev);

        let amino_acid_gly_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_ala_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_val_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_leu_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_ile_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_ser_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_thr_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_asp_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_ash_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_asn_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_glu_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_glh_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_gln_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_lys_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_lyn_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_arg_nn: NNLarge = BuildModule::build(&dev);
//        let amino_acid_arn_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_cys_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_cyx_nn: NNLarge = BuildModule::build(&dev);
//        let amino_acid_cym_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_met_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_hid_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_hie_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_hip_nn: NNLarge= BuildModule::build(&dev);
        let amino_acid_phe_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_tyr_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_trp_nn: NNLarge = BuildModule::build(&dev);
        let amino_acid_pro_nn: NNLarge = BuildModule::build(&dev);

        let head_h_nn: NNMiddle = BuildModule::build(&dev);
        let tail_o_nn: NNMiddle = BuildModule::build(&dev);
        let tail_h_nn: NNMiddle = BuildModule::build(&dev);
        let head_ace_nn: NNLarge = BuildModule::build(&dev);
        let tail_nme_nn: NNLarge = BuildModule::build(&dev);
        let tail_nhe_nn: NNLarge = BuildModule::build(&dev);

        let molecule_wat_nn: NNSmall = BuildModule::build(&dev);

        // Return the global NN
        GlobalNN
        {
            element_h_nn,
            element_b_nn,
            element_c_nn,
            element_n_nn,
            element_o_nn,
            element_f_nn,
            element_na_nn,
            element_mg_nn,
            element_si_nn,
            element_p_nn,
            element_s_nn,
            element_cl_nn,
            element_k_nn,
            element_ca_nn,
            element_v_nn,
            element_cr_nn,
            element_mn_nn,
            element_fe_nn,
            element_co_nn,
            element_ni_nn,
            element_cu_nn,
            element_zn_nn,
            element_as_nn,
            element_se_nn,
            element_br_nn,
            element_mo_nn,
            element_cd_nn,
            element_sn_nn,
            element_i_nn,

            amino_acid_gly_nn,
            amino_acid_ala_nn,
            amino_acid_val_nn,
            amino_acid_leu_nn,
            amino_acid_ile_nn,
            amino_acid_ser_nn,
            amino_acid_thr_nn,
            amino_acid_asp_nn,
            amino_acid_ash_nn,
            amino_acid_asn_nn,
            amino_acid_glu_nn,
            amino_acid_glh_nn,
            amino_acid_gln_nn,
            amino_acid_lys_nn,
            amino_acid_lyn_nn,
            amino_acid_arg_nn,
//            amino_acid_arn_nn,
            amino_acid_cys_nn,
            amino_acid_cyx_nn,
//            amino_acid_cym_nn,
            amino_acid_met_nn,
            amino_acid_hid_nn,
            amino_acid_hie_nn,
            amino_acid_hip_nn,
            amino_acid_phe_nn,
            amino_acid_tyr_nn,
            amino_acid_trp_nn,
            amino_acid_pro_nn,

            head_h_nn,
            tail_o_nn,
            tail_h_nn,
            head_ace_nn,
            tail_nme_nn,
            tail_nhe_nn,

            molecule_wat_nn,
        }
    }





    /// Save the global NN
    ///
    /// # Parameters
    /// ```
    /// dir: the directory for global NN saving
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

        // Save the NNs
        self.element_h_nn.save_safetensors(dir.as_ref().join("Element_H_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_H_NN.safetensors")));
        self.element_b_nn.save_safetensors(dir.as_ref().join("Element_B_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_B_NN.safetensors")));
        self.element_c_nn.save_safetensors(dir.as_ref().join("Element_C_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_C_NN.safetensors")));
        self.element_n_nn.save_safetensors(dir.as_ref().join("Element_N_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_N_NN.safetensors")));
        self.element_o_nn.save_safetensors(dir.as_ref().join("Element_O_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_O_NN.safetensors")));
        self.element_f_nn.save_safetensors(dir.as_ref().join("Element_F_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_F_NN.safetensors")));
        self.element_na_nn.save_safetensors(dir.as_ref().join("Element_Na_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Na_NN.safetensors")));
        self.element_mg_nn.save_safetensors(dir.as_ref().join("Element_Mg_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Mg_NN.safetensors")));
        self.element_si_nn.save_safetensors(dir.as_ref().join("Element_Si_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Si_NN.safetensors")));
        self.element_p_nn.save_safetensors(dir.as_ref().join("Element_P_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_P_NN.safetensors")));
        self.element_s_nn.save_safetensors(dir.as_ref().join("Element_S_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_S_NN.safetensors")));
        self.element_cl_nn.save_safetensors(dir.as_ref().join("Element_Cl_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Cl_NN.safetensors")));
        self.element_k_nn.save_safetensors(dir.as_ref().join("Element_K_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_K_NN.safetensors")));
        self.element_ca_nn.save_safetensors(dir.as_ref().join("Element_Ca_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Ca_NN.safetensors")));
        self.element_v_nn.save_safetensors(dir.as_ref().join("Element_V_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_V_NN.safetensors")));
        self.element_cr_nn.save_safetensors(dir.as_ref().join("Element_Cr_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Cr_NN.safetensors")));
        self.element_mn_nn.save_safetensors(dir.as_ref().join("Element_Mn_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Mn_NN.safetensors")));
        self.element_fe_nn.save_safetensors(dir.as_ref().join("Element_Fe_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Fe_NN.safetensors")));
        self.element_co_nn.save_safetensors(dir.as_ref().join("Element_Co_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Co_NN.safetensors")));
        self.element_ni_nn.save_safetensors(dir.as_ref().join("Element_Ni_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Ni_NN.safetensors")));
        self.element_cu_nn.save_safetensors(dir.as_ref().join("Element_Cu_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Cu_NN.safetensors")));
        self.element_zn_nn.save_safetensors(dir.as_ref().join("Element_Zn_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Zn_NN.safetensors")));
        self.element_as_nn.save_safetensors(dir.as_ref().join("Element_As_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_As_NN.safetensors")));
        self.element_se_nn.save_safetensors(dir.as_ref().join("Element_Se_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Se_NN.safetensors")));
        self.element_br_nn.save_safetensors(dir.as_ref().join("Element_Br_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Br_NN.safetensors")));
        self.element_mo_nn.save_safetensors(dir.as_ref().join("Element_Mo_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Mo_NN.safetensors")));
        self.element_cd_nn.save_safetensors(dir.as_ref().join("Element_Cd_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Cd_NN.safetensors")));
        self.element_sn_nn.save_safetensors(dir.as_ref().join("Element_Sn_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_Sn_NN.safetensors")));
        self.element_i_nn.save_safetensors(dir.as_ref().join("Element_I_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Element_I_NN.safetensors")));

        self.amino_acid_gly_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Gly_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Gly_NN.safetensors")));
        self.amino_acid_ala_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Ala_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Ala_NN.safetensors")));
        self.amino_acid_val_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Val_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Val_NN.safetensors")));
        self.amino_acid_leu_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Leu_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Leu_NN.safetensors")));
        self.amino_acid_ile_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Ile_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Ile_NN.safetensors")));
        self.amino_acid_ser_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Ser_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Ser_NN.safetensors")));
        self.amino_acid_thr_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Thr_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Thr_NN.safetensors")));
        self.amino_acid_asp_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Asp_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Asp_NN.safetensors")));
        self.amino_acid_ash_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Ash_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Ash_NN.safetensors")));
        self.amino_acid_asn_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Asn_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Asn_NN.safetensors")));
        self.amino_acid_glu_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Glu_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Glu_NN.safetensors")));
        self.amino_acid_glh_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Glh_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Glh_NN.safetensors")));
        self.amino_acid_gln_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Gln_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Gln_NN.safetensors")));
        self.amino_acid_lys_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Lys_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Lys_NN.safetensors")));
        self.amino_acid_lyn_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Lyn_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Lyn_NN.safetensors")));
        self.amino_acid_arg_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Arg_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Arg_NN.safetensors")));
//        self.amino_acid_arn_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Arn_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Arn_NN.safetensors")));
        self.amino_acid_cys_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Cys_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Cys_NN.safetensors")));
        self.amino_acid_cyx_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Cyx_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Cyx_NN.safetensors")));
//        self.amino_acid_cym_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Cym_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Cym_NN.safetensors")));
        self.amino_acid_met_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Met_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Met_NN.safetensors")));
        self.amino_acid_hid_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Hid_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Hid_NN.safetensors")));
        self.amino_acid_hie_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Hie_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Hie_NN.safetensors")));
        self.amino_acid_hip_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Hip_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Hip_NN.safetensors")));
        self.amino_acid_phe_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Phe_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Phe_NN.safetensors")));
        self.amino_acid_tyr_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Tyr_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Tyr_NN.safetensors")));
        self.amino_acid_trp_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Trp_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Trp_NN.safetensors")));
        self.amino_acid_pro_nn.save_safetensors(dir.as_ref().join("Amino_Acid_Pro_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Amino_Acid_Pro_NN.safetensors")));

        self.head_h_nn.save_safetensors(dir.as_ref().join("Head_H_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Head_H_NN.safetensors")));
        self.tail_o_nn.save_safetensors(dir.as_ref().join("Tail_O_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Tail_O_NN.safetensors")));
        self.tail_h_nn.save_safetensors(dir.as_ref().join("Tail_H_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Tail_H_NN.safetensors")));
        self.head_ace_nn.save_safetensors(dir.as_ref().join("Head_Ace_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Head_Ace_NN.safetensors")));
        self.tail_nme_nn.save_safetensors(dir.as_ref().join("Tail_Nme_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Tail_Nme_NN.safetensors")));
        self.tail_nhe_nn.save_safetensors(dir.as_ref().join("Tail_Nhe_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Tail_Nhe_NN.safetensors")));

        self.molecule_wat_nn.save_safetensors(dir.as_ref().join("Molecule_Wat_NN.safetensors")).expect(&error_file("creating", dir.as_ref().join("Molecule_Wat_NN.safetensors")));
    }





    /// Load the global NN
    ///
    /// # Parameters
    /// ```
    /// dir: the input directory to the global NN
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn load<P: AsRef<Path> + Debug>(dir: P) -> Self
    {
        let mut global_nn: GlobalNN = GlobalNN::new();

        // Read the NNs
        global_nn.element_h_nn.load_safetensors(dir.as_ref().join("Element_H_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_H_NN.safetensors")));
        global_nn.element_b_nn.load_safetensors(dir.as_ref().join("Element_B_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_B_NN.safetensors")));
        global_nn.element_c_nn.load_safetensors(dir.as_ref().join("Element_C_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_C_NN.safetensors")));
        global_nn.element_n_nn.load_safetensors(dir.as_ref().join("Element_N_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_N_NN.safetensors")));
        global_nn.element_o_nn.load_safetensors(dir.as_ref().join("Element_O_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_O_NN.safetensors")));
        global_nn.element_f_nn.load_safetensors(dir.as_ref().join("Element_F_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_F_NN.safetensors")));
        global_nn.element_na_nn.load_safetensors(dir.as_ref().join("Element_Na_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Na_NN.safetensors")));
        global_nn.element_mg_nn.load_safetensors(dir.as_ref().join("Element_Mg_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Mg_NN.safetensors")));
        global_nn.element_si_nn.load_safetensors(dir.as_ref().join("Element_Si_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Si_NN.safetensors")));
        global_nn.element_p_nn.load_safetensors(dir.as_ref().join("Element_P_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_P_NN.safetensors")));
        global_nn.element_s_nn.load_safetensors(dir.as_ref().join("Element_S_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_S_NN.safetensors")));
        global_nn.element_cl_nn.load_safetensors(dir.as_ref().join("Element_Cl_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Cl_NN.safetensors")));
        global_nn.element_k_nn.load_safetensors(dir.as_ref().join("Element_K_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_K_NN.safetensors")));
        global_nn.element_ca_nn.load_safetensors(dir.as_ref().join("Element_Ca_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Ca_NN.safetensors")));
        global_nn.element_v_nn.load_safetensors(dir.as_ref().join("Element_V_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_V_NN.safetensors")));
        global_nn.element_cr_nn.load_safetensors(dir.as_ref().join("Element_Cr_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Cr_NN.safetensors")));
        global_nn.element_mn_nn.load_safetensors(dir.as_ref().join("Element_Mn_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Mn_NN.safetensors")));
        global_nn.element_fe_nn.load_safetensors(dir.as_ref().join("Element_Fe_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Fe_NN.safetensors")));
        global_nn.element_co_nn.load_safetensors(dir.as_ref().join("Element_Co_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Co_NN.safetensors")));
        global_nn.element_ni_nn.load_safetensors(dir.as_ref().join("Element_Ni_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Ni_NN.safetensors")));
        global_nn.element_cu_nn.load_safetensors(dir.as_ref().join("Element_Cu_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Cu_NN.safetensors")));
        global_nn.element_zn_nn.load_safetensors(dir.as_ref().join("Element_Zn_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Zn_NN.safetensors")));
        global_nn.element_as_nn.load_safetensors(dir.as_ref().join("Element_As_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_As_NN.safetensors")));
        global_nn.element_se_nn.load_safetensors(dir.as_ref().join("Element_Se_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Se_NN.safetensors")));
        global_nn.element_br_nn.load_safetensors(dir.as_ref().join("Element_Br_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Br_NN.safetensors")));
        global_nn.element_mo_nn.load_safetensors(dir.as_ref().join("Element_Mo_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Mo_NN.safetensors")));
        global_nn.element_cd_nn.load_safetensors(dir.as_ref().join("Element_Cd_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Cd_NN.safetensors")));
        global_nn.element_sn_nn.load_safetensors(dir.as_ref().join("Element_Sn_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_Sn_NN.safetensors")));
        global_nn.element_i_nn.load_safetensors(dir.as_ref().join("Element_I_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Element_I_NN.safetensors")));

        global_nn.amino_acid_gly_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Gly_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Gly_NN.safetensors")));
        global_nn.amino_acid_ala_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Ala_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Ala_NN.safetensors")));
        global_nn.amino_acid_val_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Val_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Val_NN.safetensors")));
        global_nn.amino_acid_leu_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Leu_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Leu_NN.safetensors")));
        global_nn.amino_acid_ile_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Ile_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Ile_NN.safetensors")));
        global_nn.amino_acid_ser_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Ser_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Ser_NN.safetensors")));
        global_nn.amino_acid_thr_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Thr_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Thr_NN.safetensors")));
        global_nn.amino_acid_asp_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Asp_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Asp_NN.safetensors")));
        global_nn.amino_acid_ash_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Ash_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Ash_NN.safetensors")));
        global_nn.amino_acid_asn_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Asn_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Asn_NN.safetensors")));
        global_nn.amino_acid_glu_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Glu_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Glu_NN.safetensors")));
        global_nn.amino_acid_glh_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Glh_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Glh_NN.safetensors")));
        global_nn.amino_acid_gln_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Gln_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Gln_NN.safetensors")));
        global_nn.amino_acid_lys_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Lys_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Lys_NN.safetensors")));
        global_nn.amino_acid_lyn_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Lyn_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Lyn_NN.safetensors")));
        global_nn.amino_acid_arg_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Arg_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Arg_NN.safetensors")));
//        global_nn.amino_acid_arn_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Arn_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Arn_NN.safetensors")));
        global_nn.amino_acid_cys_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Cys_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Cys_NN.safetensors")));
        global_nn.amino_acid_cyx_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Cyx_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Cyx_NN.safetensors")));
//        global_nn.amino_acid_cym_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Cym_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Cym_NN.safetensors")));
        global_nn.amino_acid_met_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Met_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Met_NN.safetensors")));
        global_nn.amino_acid_hid_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Hid_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Hid_NN.safetensors")));
        global_nn.amino_acid_hie_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Hie_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Hie_NN.safetensors")));
        global_nn.amino_acid_hip_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Hip_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Hip_NN.safetensors")));
        global_nn.amino_acid_phe_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Phe_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Phe_NN.safetensors")));
        global_nn.amino_acid_tyr_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Tyr_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Tyr_NN.safetensors")));
        global_nn.amino_acid_trp_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Trp_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Trp_NN.safetensors")));
        global_nn.amino_acid_pro_nn.load_safetensors(dir.as_ref().join("Amino_Acid_Pro_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Amino_Acid_Pro_NN.safetensors")));

        global_nn.head_h_nn.load_safetensors(dir.as_ref().join("Head_H_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Head_H_NN.safetensors")));
        global_nn.tail_o_nn.load_safetensors(dir.as_ref().join("Tail_O_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Tail_O_NN.safetensors")));
        global_nn.tail_h_nn.load_safetensors(dir.as_ref().join("Tail_H_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Tail_H_NN.safetensors")));
        global_nn.head_ace_nn.load_safetensors(dir.as_ref().join("Head_Ace_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Head_Ace_NN.safetensors")));
        global_nn.tail_nme_nn.load_safetensors(dir.as_ref().join("Tail_Nme_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Tail_Nme_NN.safetensors")));
        global_nn.tail_nhe_nn.load_safetensors(dir.as_ref().join("Tail_Nhe_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Tail_Nhe_NN.safetensors")));

        global_nn.molecule_wat_nn.load_safetensors(dir.as_ref().join("Molecule_Wat_NN.safetensors")).expect(&error_file("reading", dir.as_ref().join("Molecule_Wat_NN.safetensors")));

        global_nn
    }





    /// Allocate gradients for all the neural networks in global NN
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn alloc_grads(&self, nn_update: &NnUpdate) -> Gradients<f64, Device>
    {
        // Define a Device (CPU or Cuda) to build NNs
        let dev: Device = Device::seed_from_u64(1314);

        // Define and initialize a global_temporary_tensor
        let temporary_tensor: Tensor<Rank0, f64, Device, NoneTape> = dev.zeros();
        let temporary_grads: Gradients<f64, Device> = temporary_tensor.alloc_grads();
        let mut global_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = temporary_tensor.traced(temporary_grads);
        

        if nn_update.update_element_h_nn
        {
            let element_h_nn_grads: Gradients<f64, Device> = self.element_h_nn.alloc_grads();
            let element_h_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_h_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_h_temporary_tensor;
        }
        if nn_update.update_element_b_nn
        {
            let element_b_nn_grads: Gradients<f64, Device> = self.element_b_nn.alloc_grads();
            let element_b_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_b_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_b_temporary_tensor;
        }
        if nn_update.update_element_c_nn
        {
            let element_c_nn_grads: Gradients<f64, Device> = self.element_c_nn.alloc_grads();
            let element_c_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_c_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_c_temporary_tensor;
        }
        if nn_update.update_element_n_nn
        {
            let element_n_nn_grads: Gradients<f64, Device> = self.element_n_nn.alloc_grads();
            let element_n_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_n_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_n_temporary_tensor;
        }
        if nn_update.update_element_o_nn
        {
            let element_o_nn_grads: Gradients<f64, Device> = self.element_o_nn.alloc_grads();
            let element_o_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_o_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_o_temporary_tensor;
        }
        if nn_update.update_element_f_nn
        {
            let element_f_nn_grads: Gradients<f64, Device> = self.element_f_nn.alloc_grads();
            let element_f_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_f_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_f_temporary_tensor;
        }
        if nn_update.update_element_na_nn
        {
            let element_na_nn_grads: Gradients<f64, Device> = self.element_na_nn.alloc_grads();
            let element_na_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_na_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_na_temporary_tensor;
        }
        if nn_update.update_element_mg_nn
        {
            let element_mg_nn_grads: Gradients<f64, Device> = self.element_mg_nn.alloc_grads();
            let element_mg_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_mg_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_mg_temporary_tensor;
        }
        if nn_update.update_element_si_nn
        {
            let element_si_nn_grads: Gradients<f64, Device> = self.element_si_nn.alloc_grads();
            let element_si_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_si_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_si_temporary_tensor;
        }
        if nn_update.update_element_p_nn
        {
            let element_p_nn_grads: Gradients<f64, Device> = self.element_p_nn.alloc_grads();
            let element_p_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_p_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_p_temporary_tensor;
        }
        if nn_update.update_element_s_nn
        {
            let element_s_nn_grads: Gradients<f64, Device> = self.element_s_nn.alloc_grads();
            let element_s_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_s_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_s_temporary_tensor;
        }
        if nn_update.update_element_cl_nn
        {
            let element_cl_nn_grads: Gradients<f64, Device> = self.element_cl_nn.alloc_grads();
            let element_cl_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_cl_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_cl_temporary_tensor;
        }
        if nn_update.update_element_k_nn
        {
            let element_k_nn_grads: Gradients<f64, Device> = self.element_k_nn.alloc_grads();
            let element_k_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_k_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_k_temporary_tensor;
        }
        if nn_update.update_element_ca_nn
        {
            let element_ca_nn_grads: Gradients<f64, Device> = self.element_ca_nn.alloc_grads();
            let element_ca_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_ca_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_ca_temporary_tensor;
        }
        if nn_update.update_element_v_nn
        {
            let element_v_nn_grads: Gradients<f64, Device> = self.element_v_nn.alloc_grads();
            let element_v_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_v_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_v_temporary_tensor;
        }
        if nn_update.update_element_cr_nn
        {
            let element_cr_nn_grads: Gradients<f64, Device> = self.element_cr_nn.alloc_grads();
            let element_cr_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_cr_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_cr_temporary_tensor;
        }
        if nn_update.update_element_mn_nn
        {
            let element_mn_nn_grads: Gradients<f64, Device> = self.element_mn_nn.alloc_grads();
            let element_mn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_mn_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_mn_temporary_tensor;
        }
        if nn_update.update_element_fe_nn
        {
            let element_fe_nn_grads: Gradients<f64, Device> = self.element_fe_nn.alloc_grads();
            let element_fe_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_fe_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_fe_temporary_tensor;
        }
        if nn_update.update_element_co_nn
        {
            let element_co_nn_grads: Gradients<f64, Device> = self.element_co_nn.alloc_grads();
            let element_co_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_co_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_co_temporary_tensor;
        }
        if nn_update.update_element_ni_nn
        {
            let element_ni_nn_grads: Gradients<f64, Device> = self.element_ni_nn.alloc_grads();
            let element_ni_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_ni_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_ni_temporary_tensor;
        }
        if nn_update.update_element_cu_nn
        {
            let element_cu_nn_grads: Gradients<f64, Device> = self.element_cu_nn.alloc_grads();
            let element_cu_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_cu_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_cu_temporary_tensor;
        }
        if nn_update.update_element_zn_nn
        {
            let element_zn_nn_grads: Gradients<f64, Device> = self.element_zn_nn.alloc_grads();
            let element_zn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_zn_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_zn_temporary_tensor;
        }
        if nn_update.update_element_as_nn
        {
            let element_as_nn_grads: Gradients<f64, Device> = self.element_as_nn.alloc_grads();
            let element_as_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_as_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_as_temporary_tensor;
        }
        if nn_update.update_element_se_nn
        {
            let element_se_nn_grads: Gradients<f64, Device> = self.element_se_nn.alloc_grads();
            let element_se_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_se_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_se_temporary_tensor;
        }
        if nn_update.update_element_br_nn
        {
            let element_br_nn_grads: Gradients<f64, Device> = self.element_br_nn.alloc_grads();
            let element_br_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_br_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_br_temporary_tensor;
        }
        if nn_update.update_element_mo_nn
        {
            let element_mo_nn_grads: Gradients<f64, Device> = self.element_mo_nn.alloc_grads();
            let element_mo_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_mo_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_mo_temporary_tensor;
        }
        if nn_update.update_element_cd_nn
        {
            let element_cd_nn_grads: Gradients<f64, Device> = self.element_cd_nn.alloc_grads();
            let element_cd_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_cd_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_cd_temporary_tensor;
        }
        if nn_update.update_element_sn_nn
        {
            let element_sn_nn_grads: Gradients<f64, Device> = self.element_sn_nn.alloc_grads();
            let element_sn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_sn_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_sn_temporary_tensor;
        }
        if nn_update.update_element_i_nn
        {
            let element_i_nn_grads: Gradients<f64, Device> = self.element_i_nn.alloc_grads();
            let element_i_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_i_nn_grads);
            global_temporary_tensor = global_temporary_tensor + element_i_temporary_tensor;
        }


        if nn_update.update_amino_acid_gly_nn
        {
            let amino_acid_gly_nn_grads: Gradients<f64, Device> = self.amino_acid_gly_nn.alloc_grads();
            let amino_acid_gly_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_gly_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_gly_temporary_tensor;
        }
        if nn_update.update_amino_acid_ala_nn
        {
            let amino_acid_ala_nn_grads: Gradients<f64, Device> = self.amino_acid_ala_nn.alloc_grads();
            let amino_acid_ala_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_ala_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_ala_temporary_tensor;
        }
        if nn_update.update_amino_acid_val_nn
        {
            let amino_acid_val_nn_grads: Gradients<f64, Device> = self.amino_acid_val_nn.alloc_grads();
            let amino_acid_val_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_val_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_val_temporary_tensor;
        }
        if nn_update.update_amino_acid_leu_nn
        {
            let amino_acid_leu_nn_grads: Gradients<f64, Device> = self.amino_acid_leu_nn.alloc_grads();
            let amino_acid_leu_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_leu_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_leu_temporary_tensor;
        }
        if nn_update.update_amino_acid_ile_nn
        {
            let amino_acid_ile_nn_grads: Gradients<f64, Device> = self.amino_acid_ile_nn.alloc_grads();
            let amino_acid_ile_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_ile_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_ile_temporary_tensor;
        }
        if nn_update.update_amino_acid_ser_nn
        {
            let amino_acid_ser_nn_grads: Gradients<f64, Device> = self.amino_acid_ser_nn.alloc_grads();
            let amino_acid_ser_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_ser_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_ser_temporary_tensor;
        }
        if nn_update.update_amino_acid_thr_nn
        {
            let amino_acid_thr_nn_grads: Gradients<f64, Device> = self.amino_acid_thr_nn.alloc_grads();
            let amino_acid_thr_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_thr_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_thr_temporary_tensor;
        }
        if nn_update.update_amino_acid_asp_nn
        {
            let amino_acid_asp_nn_grads: Gradients<f64, Device> = self.amino_acid_asp_nn.alloc_grads();
            let amino_acid_asp_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_asp_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_asp_temporary_tensor;
        }
        if nn_update.update_amino_acid_ash_nn
        {
            let amino_acid_ash_nn_grads: Gradients<f64, Device> = self.amino_acid_ash_nn.alloc_grads();
            let amino_acid_ash_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_ash_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_ash_temporary_tensor;
        }
        if nn_update.update_amino_acid_asn_nn
        {
            let amino_acid_asn_nn_grads: Gradients<f64, Device> = self.amino_acid_asn_nn.alloc_grads();
            let amino_acid_asn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_asn_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_asn_temporary_tensor;
        }
        if nn_update.update_amino_acid_glu_nn
        {
            let amino_acid_glu_nn_grads: Gradients<f64, Device> = self.amino_acid_glu_nn.alloc_grads();
            let amino_acid_glu_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_glu_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_glu_temporary_tensor;
        }
        if nn_update.update_amino_acid_glh_nn
        {
            let amino_acid_glh_nn_grads: Gradients<f64, Device> = self.amino_acid_glh_nn.alloc_grads();
            let amino_acid_glh_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_glh_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_glh_temporary_tensor;
        }
        if nn_update.update_amino_acid_gln_nn
        {
            let amino_acid_gln_nn_grads: Gradients<f64, Device> = self.amino_acid_gln_nn.alloc_grads();
            let amino_acid_gln_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_gln_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_gln_temporary_tensor;
        }
        if nn_update.update_amino_acid_lys_nn
        {
            let amino_acid_lys_nn_grads: Gradients<f64, Device> = self.amino_acid_lys_nn.alloc_grads();
            let amino_acid_lys_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_lys_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_lys_temporary_tensor;
        }
        if nn_update.update_amino_acid_lyn_nn
        {
            let amino_acid_lyn_nn_grads: Gradients<f64, Device> = self.amino_acid_lyn_nn.alloc_grads();
            let amino_acid_lyn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_lyn_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_lyn_temporary_tensor;
        }
        if nn_update.update_amino_acid_arg_nn
        {
            let amino_acid_arg_nn_grads: Gradients<f64, Device> = self.amino_acid_arg_nn.alloc_grads();
            let amino_acid_arg_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_arg_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_arg_temporary_tensor;
        }
/*
        if nn_update.update_amino_acid_arn_nn
        {
            let amino_acid_arn_nn_grads: Gradients<f64, Device> = self.amino_acid_arn_nn.alloc_grads();
            let amino_acid_arn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_arn_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_arn_temporary_tensor;
        }
*/
        if nn_update.update_amino_acid_cys_nn
        {
            let amino_acid_cys_nn_grads: Gradients<f64, Device> = self.amino_acid_cys_nn.alloc_grads();
            let amino_acid_cys_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_cys_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_cys_temporary_tensor;
        }
        if nn_update.update_amino_acid_cyx_nn
        {
            let amino_acid_cyx_nn_grads: Gradients<f64, Device> = self.amino_acid_cyx_nn.alloc_grads();
            let amino_acid_cyx_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_cyx_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_cyx_temporary_tensor;
        }
/*
        if nn_update.update_amino_acid_cym_nn
        {
            let amino_acid_cym_nn_grads: Gradients<f64, Device> = self.amino_acid_cym_nn.alloc_grads();
            let amino_acid_cym_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_cym_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_cym_temporary_tensor;
        }
*/
        if nn_update.update_amino_acid_met_nn
        {
            let amino_acid_met_nn_grads: Gradients<f64, Device> = self.amino_acid_met_nn.alloc_grads();
            let amino_acid_met_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_met_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_met_temporary_tensor;
        }
        if nn_update.update_amino_acid_hid_nn
        {
            let amino_acid_hid_nn_grads: Gradients<f64, Device> = self.amino_acid_hid_nn.alloc_grads();
            let amino_acid_hid_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_hid_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_hid_temporary_tensor;
        }
        if nn_update.update_amino_acid_hie_nn
        {
            let amino_acid_hie_nn_grads: Gradients<f64, Device> = self.amino_acid_hie_nn.alloc_grads();
            let amino_acid_hie_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_hie_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_hie_temporary_tensor;
        }
        if nn_update.update_amino_acid_hip_nn
        {
            let amino_acid_hip_nn_grads: Gradients<f64, Device> = self.amino_acid_hip_nn.alloc_grads();
            let amino_acid_hip_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_hip_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_hip_temporary_tensor;
        }
        if nn_update.update_amino_acid_phe_nn
        {
            let amino_acid_phe_nn_grads: Gradients<f64, Device> = self.amino_acid_phe_nn.alloc_grads();
            let amino_acid_phe_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_phe_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_phe_temporary_tensor;
        }
        if nn_update.update_amino_acid_tyr_nn
        {
            let amino_acid_tyr_nn_grads: Gradients<f64, Device> = self.amino_acid_tyr_nn.alloc_grads();
            let amino_acid_tyr_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_tyr_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_tyr_temporary_tensor;
        }
        if nn_update.update_amino_acid_trp_nn
        {
            let amino_acid_trp_nn_grads: Gradients<f64, Device> = self.amino_acid_trp_nn.alloc_grads();
            let amino_acid_trp_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_trp_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_trp_temporary_tensor;
        }
        if nn_update.update_amino_acid_pro_nn
        {
            let amino_acid_pro_nn_grads: Gradients<f64, Device> = self.amino_acid_pro_nn.alloc_grads();
            let amino_acid_pro_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_pro_nn_grads);
            global_temporary_tensor = global_temporary_tensor + amino_acid_pro_temporary_tensor;
        }


        if nn_update.update_head_h_nn
        {
            let head_h_nn_grads: Gradients<f64, Device> = self.head_h_nn.alloc_grads();
            let head_h_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(head_h_nn_grads);
            global_temporary_tensor = global_temporary_tensor + head_h_temporary_tensor;
        }
        if nn_update.update_tail_o_nn
        {
            let tail_o_nn_grads: Gradients<f64, Device> = self.tail_o_nn.alloc_grads();
            let tail_o_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(tail_o_nn_grads);
            global_temporary_tensor = global_temporary_tensor + tail_o_temporary_tensor;
        }
        if nn_update.update_tail_h_nn
        {
            let tail_h_nn_grads: Gradients<f64, Device> = self.tail_h_nn.alloc_grads();
            let tail_h_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(tail_h_nn_grads);
            global_temporary_tensor = global_temporary_tensor + tail_h_temporary_tensor;
        }
        if nn_update.update_head_ace_nn
        {
            let head_ace_nn_grads: Gradients<f64, Device> = self.head_ace_nn.alloc_grads();
            let head_ace_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(head_ace_nn_grads);
            global_temporary_tensor = global_temporary_tensor + head_ace_temporary_tensor;
        }
        if nn_update.update_tail_nme_nn
        {
            let tail_nme_nn_grads: Gradients<f64, Device> = self.tail_nme_nn.alloc_grads();
            let tail_nme_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(tail_nme_nn_grads);
            global_temporary_tensor = global_temporary_tensor + tail_nme_temporary_tensor;
        }
        if nn_update.update_tail_nhe_nn
        {
            let tail_nhe_nn_grads: Gradients<f64, Device> = self.tail_nhe_nn.alloc_grads();
            let tail_nhe_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(tail_nhe_nn_grads);
            global_temporary_tensor = global_temporary_tensor + tail_nhe_temporary_tensor;
        }


        if nn_update.update_molecule_wat_nn
        {
            let molecule_wat_nn_grads: Gradients<f64, Device> = self.molecule_wat_nn.alloc_grads();
            let molecule_wat_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(molecule_wat_nn_grads);
            global_temporary_tensor = global_temporary_tensor + molecule_wat_temporary_tensor;
        }


        global_temporary_tensor.backward()
    }





    /// Zero all the gradients associated with global NN
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn zero_grads(&self, global_grads: &mut Gradients<f64, Device>, nn_update: &NnUpdate)
    {
        if nn_update.update_element_h_nn
        {
            self.element_h_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_b_nn
        {
            self.element_b_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_c_nn
        {
            self.element_c_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_n_nn
        {
            self.element_n_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_o_nn
        {
            self.element_o_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_f_nn
        {
            self.element_f_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_na_nn
        {
            self.element_na_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_mg_nn
        {
            self.element_mg_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_si_nn
        {
            self.element_si_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_p_nn
        {
            self.element_p_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_s_nn
        {
            self.element_s_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_cl_nn
        {
            self.element_cl_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_k_nn
        {
            self.element_k_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_ca_nn
        {
            self.element_ca_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_v_nn
        {
            self.element_v_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_cr_nn
        {
            self.element_cr_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_mn_nn
        {
            self.element_mn_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_fe_nn
        {
            self.element_fe_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_co_nn
        {
            self.element_co_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_ni_nn
        {
            self.element_ni_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_cu_nn
        {
            self.element_cu_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_zn_nn
        {
            self.element_zn_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_as_nn
        {
            self.element_as_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_se_nn
        {
            self.element_se_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_br_nn
        {
            self.element_br_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_mo_nn
        {
            self.element_mo_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_cd_nn
        {
            self.element_cd_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_sn_nn
        {
            self.element_sn_nn.zero_grads(global_grads);
        }
        if nn_update.update_element_i_nn
        {
            self.element_i_nn.zero_grads(global_grads);
        }


        if nn_update.update_amino_acid_gly_nn
        {
            self.amino_acid_gly_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_ala_nn
        {
            self.amino_acid_ala_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_val_nn
        {
            self.amino_acid_val_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_leu_nn
        {
            self.amino_acid_leu_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_ile_nn
        {
            self.amino_acid_ile_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_ser_nn
        {
            self.amino_acid_ser_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_thr_nn
        {
            self.amino_acid_thr_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_asp_nn
        {
            self.amino_acid_asp_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_ash_nn
        {
            self.amino_acid_ash_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_asn_nn
        {
            self.amino_acid_asn_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_glu_nn
        {
            self.amino_acid_glu_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_glh_nn
        {
            self.amino_acid_glh_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_gln_nn
        {
            self.amino_acid_gln_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_lys_nn
        {
            self.amino_acid_lys_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_lyn_nn
        {
            self.amino_acid_lyn_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_arg_nn
        {
            self.amino_acid_arg_nn.zero_grads(global_grads);
        }
/*
        if nn_update.update_amino_acid_arn_nn
        {
            self.amino_acid_arn_nn.zero_grads(global_grads);
        }
*/
        if nn_update.update_amino_acid_cys_nn
        {
            self.amino_acid_cys_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_cyx_nn
        {
            self.amino_acid_cyx_nn.zero_grads(global_grads);
        }
/*
        if nn_update.update_amino_acid_cym_nn
        {
            self.amino_acid_cym_nn.zero_grads(global_grads);
        }
*/
        if nn_update.update_amino_acid_met_nn
        {
            self.amino_acid_met_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_hid_nn
        {
            self.amino_acid_hid_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_hie_nn
        {
            self.amino_acid_hie_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_hip_nn
        {
            self.amino_acid_hip_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_phe_nn
        {
            self.amino_acid_phe_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_tyr_nn
        {
            self.amino_acid_tyr_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_trp_nn
        {
            self.amino_acid_trp_nn.zero_grads(global_grads);
        }
        if nn_update.update_amino_acid_pro_nn
        {
            self.amino_acid_pro_nn.zero_grads(global_grads);
        }


        if nn_update.update_head_h_nn
        {
            self.head_h_nn.zero_grads(global_grads);
        }
        if nn_update.update_tail_o_nn
        {
            self.tail_o_nn.zero_grads(global_grads);
        }
        if nn_update.update_tail_h_nn
        {
            self.tail_h_nn.zero_grads(global_grads);
        }
        if nn_update.update_head_ace_nn
        {
            self.head_ace_nn.zero_grads(global_grads);
        }
        if nn_update.update_tail_nme_nn
        {
            self.tail_nme_nn.zero_grads(global_grads);
        }
        if nn_update.update_tail_nhe_nn
        {
            self.tail_nhe_nn.zero_grads(global_grads);
        }


        if nn_update.update_molecule_wat_nn
        {
            self.molecule_wat_nn.zero_grads(global_grads)
        }
    }





    /// Get the neural network for a given fragment
    ///
    /// # Parameters
    /// ```
    /// fragment_type: the input fragment type
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn get_fragment_nn(&self, fragment_type: &FragmentType) -> NN<'_>
    {
        match fragment_type
        {
            FragmentType::Atom(element) =>
            {
                match element
                {
                    Element::H => NN::Large(&self.element_h_nn),
                    Element::C => NN::Large(&self.element_c_nn),
                    Element::O => NN::Large(&self.element_o_nn),
                    Element::N => NN::Large(&self.element_n_nn),
                    Element::S => NN::Large(&self.element_s_nn),
                    Element::P => NN::Middle(&self.element_p_nn),
                    Element::Na => NN::Middle(&self.element_na_nn),
                    Element::Cl => NN::Middle(&self.element_cl_nn),
                    Element::K => NN::Middle(&self.element_k_nn),
                    Element::Ca => NN::Middle(&self.element_ca_nn),
                    Element::Mg => NN::Middle(&self.element_mg_nn),
                    Element::F => NN::Middle(&self.element_f_nn),
                    Element::Fe => NN::Middle(&self.element_fe_nn),
                    Element::Cu => NN::Middle(&self.element_cu_nn),
                    Element::Zn => NN::Middle(&self.element_zn_nn),
                    Element::Mn => NN::Middle(&self.element_mn_nn),
                    Element::Mo => NN::Middle(&self.element_mo_nn),
                    Element::Co => NN::Middle(&self.element_co_nn),
                    Element::Cr => NN::Middle(&self.element_cr_nn),
                    Element::V => NN::Middle(&self.element_v_nn),
                    Element::Sn => NN::Middle(&self.element_sn_nn),
                    Element::Ni => NN::Middle(&self.element_ni_nn),
                    Element::Si => NN::Middle(&self.element_si_nn),
                    Element::Se => NN::Middle(&self.element_se_nn),
                    Element::I => NN::Middle(&self.element_i_nn),
                    Element::Br => NN::Middle(&self.element_br_nn),
                    Element::As => NN::Middle(&self.element_as_nn),
                    Element::B => NN::Middle(&self.element_b_nn),
                    Element::Cd => NN::Middle(&self.element_cd_nn),
                    _ => panic!("{}", error_non_bioelement_nn(element)),
                }
            },

            FragmentType::Residue(amino_acid) =>
            {
                match amino_acid
                {
                    AminoAcid::GLY => NN::Large(&self.amino_acid_gly_nn),
                    AminoAcid::ALA => NN::Large(&self.amino_acid_ala_nn),
                    AminoAcid::VAL => NN::Large(&self.amino_acid_val_nn),
                    AminoAcid::LEU => NN::Large(&self.amino_acid_leu_nn),
                    AminoAcid::ILE => NN::Large(&self.amino_acid_ile_nn),
                    AminoAcid::SER => NN::Large(&self.amino_acid_ser_nn),
                    AminoAcid::THR => NN::Large(&self.amino_acid_thr_nn),
                    AminoAcid::ASP => NN::Large(&self.amino_acid_asp_nn),
                    AminoAcid::ASH => NN::Large(&self.amino_acid_ash_nn),
                    AminoAcid::ASN => NN::Large(&self.amino_acid_asn_nn),
                    AminoAcid::GLU => NN::Large(&self.amino_acid_glu_nn),
                    AminoAcid::GLH => NN::Large(&self.amino_acid_glh_nn),
                    AminoAcid::GLN => NN::Large(&self.amino_acid_gln_nn),
                    AminoAcid::LYS => NN::Large(&self.amino_acid_lys_nn),
                    AminoAcid::LYN => NN::Large(&self.amino_acid_lyn_nn),
                    AminoAcid::ARG => NN::Large(&self.amino_acid_arg_nn),
//                    AminoAcid::ARN => NN::Large(&self.amino_acid_arn_nn),
                    AminoAcid::CYS => NN::Large(&self.amino_acid_cys_nn),
                    AminoAcid::CYX => NN::Large(&self.amino_acid_cyx_nn),
//                    AminoAcid::CYM => NN::Large(&self.amino_acid_cym_nn),
                    AminoAcid::MET => NN::Large(&self.amino_acid_met_nn),
                    AminoAcid::HID => NN::Large(&self.amino_acid_hid_nn),
                    AminoAcid::HIE => NN::Large(&self.amino_acid_hie_nn),
                    AminoAcid::HIP => NN::Large(&self.amino_acid_hip_nn),
                    AminoAcid::PHE => NN::Large(&self.amino_acid_phe_nn),
                    AminoAcid::TYR => NN::Large(&self.amino_acid_tyr_nn),
                    AminoAcid::TRP => NN::Large(&self.amino_acid_trp_nn),
                    AminoAcid::PRO => NN::Large(&self.amino_acid_pro_nn),
                }
            },

            FragmentType::Head(cap) =>
            {
                match cap
                {
                    Cap::H => NN::Middle(&self.head_h_nn),
                    Cap::ACE => NN::Large(&self.head_ace_nn),
                    _ => panic!("{}", error_type("Head", &format!("{:?}", cap))),
                }
            },

            FragmentType::Tail(cap) =>
            {
                match cap
                {
                    Cap::O => NN::Middle(&self.tail_o_nn),
                    Cap::H => NN::Middle(&self.tail_h_nn),
                    Cap::NME => NN::Large(&self.tail_nme_nn),
                    Cap::NHE => NN::Large(&self.tail_nhe_nn),
                    _ => panic!("{}", error_type("Tail", &format!("{:?}", cap))),
                }
            },

            FragmentType::Molecule(molecule) =>
            {
                match molecule
                {
                    Molecule::WAT => NN::Small(&self.molecule_wat_nn),
                }
            },
        }
    }





    /// Get the mutable neural network for a given fragment
    ///
    /// # Parameters
    /// ```
    /// fragment_type: the input fragment type
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn get_fragment_nn_mut(&mut self, fragment_type: &FragmentType) -> NNMut<'_>
    {
        match fragment_type
        {
            FragmentType::Atom(element) =>
            {
                match element
                {
                    Element::H => NNMut::Large(&mut self.element_h_nn),
                    Element::C => NNMut::Large(&mut self.element_c_nn),
                    Element::O => NNMut::Large(&mut self.element_o_nn),
                    Element::N => NNMut::Large(&mut self.element_n_nn),
                    Element::S => NNMut::Large(&mut self.element_s_nn),
                    Element::P => NNMut::Middle(&mut self.element_p_nn),
                    Element::Na => NNMut::Middle(&mut self.element_na_nn),
                    Element::Cl => NNMut::Middle(&mut self.element_cl_nn),
                    Element::K => NNMut::Middle(&mut self.element_k_nn),
                    Element::Ca => NNMut::Middle(&mut self.element_ca_nn),
                    Element::Mg => NNMut::Middle(&mut self.element_mg_nn),
                    Element::F => NNMut::Middle(&mut self.element_f_nn),
                    Element::Fe => NNMut::Middle(&mut self.element_fe_nn),
                    Element::Cu => NNMut::Middle(&mut self.element_cu_nn),
                    Element::Zn => NNMut::Middle(&mut self.element_zn_nn),
                    Element::Mn => NNMut::Middle(&mut self.element_mn_nn),
                    Element::Mo => NNMut::Middle(&mut self.element_mo_nn),
                    Element::Co => NNMut::Middle(&mut self.element_co_nn),
                    Element::Cr => NNMut::Middle(&mut self.element_cr_nn),
                    Element::V => NNMut::Middle(&mut self.element_v_nn),
                    Element::Sn => NNMut::Middle(&mut self.element_sn_nn),
                    Element::Ni => NNMut::Middle(&mut self.element_ni_nn),
                    Element::Si => NNMut::Middle(&mut self.element_si_nn),
                    Element::Se => NNMut::Middle(&mut self.element_se_nn),
                    Element::I => NNMut::Middle(&mut self.element_i_nn),
                    Element::Br => NNMut::Middle(&mut self.element_br_nn),
                    Element::As => NNMut::Middle(&mut self.element_as_nn),
                    Element::B => NNMut::Middle(&mut self.element_b_nn),
                    Element::Cd => NNMut::Middle(&mut self.element_cd_nn),
                    _ => panic!("{}", error_non_bioelement_nn(element)),
                }
            },

            FragmentType::Residue(amino_acid) =>
            {
                match amino_acid
                {
                    AminoAcid::GLY => NNMut::Large(&mut self.amino_acid_gly_nn),
                    AminoAcid::ALA => NNMut::Large(&mut self.amino_acid_ala_nn),
                    AminoAcid::VAL => NNMut::Large(&mut self.amino_acid_val_nn),
                    AminoAcid::LEU => NNMut::Large(&mut self.amino_acid_leu_nn),
                    AminoAcid::ILE => NNMut::Large(&mut self.amino_acid_ile_nn),
                    AminoAcid::SER => NNMut::Large(&mut self.amino_acid_ser_nn),
                    AminoAcid::THR => NNMut::Large(&mut self.amino_acid_thr_nn),
                    AminoAcid::ASP => NNMut::Large(&mut self.amino_acid_asp_nn),
                    AminoAcid::ASH => NNMut::Large(&mut self.amino_acid_ash_nn),
                    AminoAcid::ASN => NNMut::Large(&mut self.amino_acid_asn_nn),
                    AminoAcid::GLU => NNMut::Large(&mut self.amino_acid_glu_nn),
                    AminoAcid::GLH => NNMut::Large(&mut self.amino_acid_glh_nn),
                    AminoAcid::GLN => NNMut::Large(&mut self.amino_acid_gln_nn),
                    AminoAcid::LYS => NNMut::Large(&mut self.amino_acid_lys_nn),
                    AminoAcid::LYN => NNMut::Large(&mut self.amino_acid_lyn_nn),
                    AminoAcid::ARG => NNMut::Large(&mut self.amino_acid_arg_nn),
//                    AminoAcid::ARN => NNMut::Large(&mut self.amino_acid_arn_nn),
                    AminoAcid::CYS => NNMut::Large(&mut self.amino_acid_cys_nn),
                    AminoAcid::CYX => NNMut::Large(&mut self.amino_acid_cyx_nn),
//                    AminoAcid::CYM => NNMut::Large(&mut self.amino_acid_cym_nn),
                    AminoAcid::MET => NNMut::Large(&mut self.amino_acid_met_nn),
                    AminoAcid::HID => NNMut::Large(&mut self.amino_acid_hid_nn),
                    AminoAcid::HIE => NNMut::Large(&mut self.amino_acid_hie_nn),
                    AminoAcid::HIP => NNMut::Large(&mut self.amino_acid_hip_nn),
                    AminoAcid::PHE => NNMut::Large(&mut self.amino_acid_phe_nn),
                    AminoAcid::TYR => NNMut::Large(&mut self.amino_acid_tyr_nn),
                    AminoAcid::TRP => NNMut::Large(&mut self.amino_acid_trp_nn),
                    AminoAcid::PRO => NNMut::Large(&mut self.amino_acid_pro_nn),
                }
            },

            FragmentType::Head(cap) =>
            {
                match cap
                {
                    Cap::H => NNMut::Middle(&mut self.head_h_nn),
                    Cap::ACE => NNMut::Large(&mut self.head_ace_nn),
                    _ => panic!("{}", error_type("Head", &format!("{:?}", cap))),
                }
            },

            FragmentType::Tail(cap) =>
            {
                match cap
                {
                    Cap::O => NNMut::Middle(&mut self.tail_o_nn),
                    Cap::H => NNMut::Middle(&mut self.tail_h_nn),
                    Cap::NME => NNMut::Large(&mut self.tail_nme_nn),
                    Cap::NHE => NNMut::Large(&mut self.tail_nhe_nn),
                    _ => panic!("{}", error_type("Tail", &format!("{:?}", cap))),
                }
            },

            FragmentType::Molecule(molecule) =>
            {
                match molecule
                {
                    Molecule::WAT => NNMut::Small(&mut self.molecule_wat_nn),
                }
            },
        }
    }
}










impl GlobalAdam
{
    /// Construct a new global Adam optimizer for the global NN
    ///
    /// # Parameters
    /// ```
    /// global_nn: the input global NN of the protein system
    /// adam_config: the input configuration for the global Adam optimizer
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn new(global_nn: &GlobalNN, adam_config: AdamConfig) -> Self
    {
        GlobalAdam
        {
            element_h_adam: Adam::new(&global_nn.element_h_nn, adam_config),
            element_b_adam: Adam::new(&global_nn.element_b_nn, adam_config),
            element_c_adam: Adam::new(&global_nn.element_c_nn, adam_config),
            element_n_adam: Adam::new(&global_nn.element_n_nn, adam_config),
            element_o_adam: Adam::new(&global_nn.element_o_nn, adam_config),
            element_f_adam: Adam::new(&global_nn.element_f_nn, adam_config),
            element_na_adam: Adam::new(&global_nn.element_na_nn, adam_config),
            element_mg_adam: Adam::new(&global_nn.element_mg_nn, adam_config),
            element_si_adam: Adam::new(&global_nn.element_si_nn, adam_config),
            element_p_adam: Adam::new(&global_nn.element_p_nn, adam_config),
            element_s_adam: Adam::new(&global_nn.element_s_nn, adam_config),
            element_cl_adam: Adam::new(&global_nn.element_cl_nn, adam_config),
            element_k_adam: Adam::new(&global_nn.element_k_nn, adam_config),
            element_ca_adam: Adam::new(&global_nn.element_ca_nn, adam_config),
            element_v_adam: Adam::new(&global_nn.element_v_nn, adam_config),
            element_cr_adam: Adam::new(&global_nn.element_cr_nn, adam_config),
            element_mn_adam: Adam::new(&global_nn.element_mn_nn, adam_config),
            element_fe_adam: Adam::new(&global_nn.element_fe_nn, adam_config),
            element_co_adam: Adam::new(&global_nn.element_co_nn, adam_config),
            element_ni_adam: Adam::new(&global_nn.element_ni_nn, adam_config),
            element_cu_adam: Adam::new(&global_nn.element_cu_nn, adam_config),
            element_zn_adam: Adam::new(&global_nn.element_zn_nn, adam_config),
            element_as_adam: Adam::new(&global_nn.element_as_nn, adam_config),
            element_se_adam: Adam::new(&global_nn.element_se_nn, adam_config),
            element_br_adam: Adam::new(&global_nn.element_br_nn, adam_config),
            element_mo_adam: Adam::new(&global_nn.element_mo_nn, adam_config),
            element_cd_adam: Adam::new(&global_nn.element_cd_nn, adam_config),
            element_sn_adam: Adam::new(&global_nn.element_sn_nn, adam_config),
            element_i_adam: Adam::new(&global_nn.element_i_nn, adam_config),

            amino_acid_gly_adam: Adam::new(&global_nn.amino_acid_gly_nn, adam_config),
            amino_acid_ala_adam: Adam::new(&global_nn.amino_acid_ala_nn, adam_config),
            amino_acid_val_adam: Adam::new(&global_nn.amino_acid_val_nn, adam_config),
            amino_acid_leu_adam: Adam::new(&global_nn.amino_acid_leu_nn, adam_config),
            amino_acid_ile_adam: Adam::new(&global_nn.amino_acid_ile_nn, adam_config),
            amino_acid_ser_adam: Adam::new(&global_nn.amino_acid_ser_nn, adam_config),
            amino_acid_thr_adam: Adam::new(&global_nn.amino_acid_thr_nn, adam_config),
            amino_acid_asp_adam: Adam::new(&global_nn.amino_acid_asp_nn, adam_config),
            amino_acid_ash_adam: Adam::new(&global_nn.amino_acid_ash_nn, adam_config),
            amino_acid_asn_adam: Adam::new(&global_nn.amino_acid_asn_nn, adam_config),
            amino_acid_glu_adam: Adam::new(&global_nn.amino_acid_glu_nn, adam_config),
            amino_acid_glh_adam: Adam::new(&global_nn.amino_acid_glh_nn, adam_config),
            amino_acid_gln_adam: Adam::new(&global_nn.amino_acid_gln_nn, adam_config),
            amino_acid_lys_adam: Adam::new(&global_nn.amino_acid_lys_nn, adam_config),
            amino_acid_lyn_adam: Adam::new(&global_nn.amino_acid_lyn_nn, adam_config),
            amino_acid_arg_adam: Adam::new(&global_nn.amino_acid_arg_nn, adam_config),
//            amino_acid_arn_adam: Adam::new(&global_nn.amino_acid_arn_nn, adam_config),
            amino_acid_cys_adam: Adam::new(&global_nn.amino_acid_cys_nn, adam_config),
            amino_acid_cyx_adam: Adam::new(&global_nn.amino_acid_cyx_nn, adam_config),
//            amino_acid_cym_adam: Adam::new(&global_nn.amino_acid_cym_nn, adam_config),
            amino_acid_met_adam: Adam::new(&global_nn.amino_acid_met_nn, adam_config),
            amino_acid_hid_adam: Adam::new(&global_nn.amino_acid_hid_nn, adam_config),
            amino_acid_hie_adam: Adam::new(&global_nn.amino_acid_hie_nn, adam_config),
            amino_acid_hip_adam: Adam::new(&global_nn.amino_acid_hip_nn, adam_config),
            amino_acid_phe_adam: Adam::new(&global_nn.amino_acid_phe_nn, adam_config),
            amino_acid_tyr_adam: Adam::new(&global_nn.amino_acid_tyr_nn, adam_config),
            amino_acid_trp_adam: Adam::new(&global_nn.amino_acid_trp_nn, adam_config),
            amino_acid_pro_adam: Adam::new(&global_nn.amino_acid_pro_nn, adam_config),

            head_h_adam: Adam::new(&global_nn.head_h_nn, adam_config),
            tail_o_adam: Adam::new(&global_nn.tail_o_nn, adam_config),
            tail_h_adam: Adam::new(&global_nn.tail_h_nn, adam_config),
            head_ace_adam: Adam::new(&global_nn.head_ace_nn, adam_config),
            tail_nme_adam: Adam::new(&global_nn.tail_nme_nn, adam_config),
            tail_nhe_adam: Adam::new(&global_nn.tail_nhe_nn, adam_config),

            molecule_wat_adam: Adam::new(&global_nn.molecule_wat_nn, adam_config),
        }
    }





    /// Update all the parameters for the input global NN using the input global gradients
    ///
    /// # Parameters
    /// ```
    /// global_nn: the input global NN of the protein system
    /// global_grads: the input global gradients of the global NN
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn update(&mut self, global_nn: &mut GlobalNN, global_grads: &Gradients<f64, Device>, nn_update: &NnUpdate)
    {
        if nn_update.update_element_h_nn
        {
            self.element_h_adam.update(&mut global_nn.element_h_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element H"));
        }
        if nn_update.update_element_b_nn
        {
            self.element_b_adam.update(&mut global_nn.element_b_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element B"));
        }
        if nn_update.update_element_c_nn
        {
            self.element_c_adam.update(&mut global_nn.element_c_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element C"));
        }
        if nn_update.update_element_n_nn
        {
            self.element_n_adam.update(&mut global_nn.element_n_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element N"));
        }
        if nn_update.update_element_o_nn
        {
            self.element_o_adam.update(&mut global_nn.element_o_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element O"));
        }
        if nn_update.update_element_f_nn
        {
            self.element_f_adam.update(&mut global_nn.element_f_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element F"));
        }
        if nn_update.update_element_na_nn
        {
            self.element_na_adam.update(&mut global_nn.element_na_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Na"));
        }
        if nn_update.update_element_mg_nn
        {
            self.element_mg_adam.update(&mut global_nn.element_mg_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Mg"));
        }
        if nn_update.update_element_si_nn
        {
            self.element_si_adam.update(&mut global_nn.element_si_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Si"));
        }
        if nn_update.update_element_p_nn
        {
            self.element_p_adam.update(&mut global_nn.element_p_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element P"));
        }
        if nn_update.update_element_s_nn
        {
            self.element_s_adam.update(&mut global_nn.element_s_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element S"));
        }
        if nn_update.update_element_cl_nn
        {
            self.element_cl_adam.update(&mut global_nn.element_cl_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Cl"));
        }
        if nn_update.update_element_k_nn
        {
            self.element_k_adam.update(&mut global_nn.element_k_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element K"));
        }
        if nn_update.update_element_ca_nn
        {
            self.element_ca_adam.update(&mut global_nn.element_ca_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Ca"));
        }
        if nn_update.update_element_v_nn
        {
            self.element_v_adam.update(&mut global_nn.element_v_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element V"));
        }
        if nn_update.update_element_cr_nn
        {
            self.element_cr_adam.update(&mut global_nn.element_cr_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Cr"));
        }
        if nn_update.update_element_mn_nn
        {
            self.element_mn_adam.update(&mut global_nn.element_mn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Mn"));
        }
        if nn_update.update_element_fe_nn
        {
            self.element_fe_adam.update(&mut global_nn.element_fe_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Fe"));
        }
        if nn_update.update_element_co_nn
        {
            self.element_co_adam.update(&mut global_nn.element_co_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Co"));
        }
        if nn_update.update_element_ni_nn
        {
            self.element_ni_adam.update(&mut global_nn.element_ni_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Ni"));
        }
        if nn_update.update_element_cu_nn
        {
            self.element_cu_adam.update(&mut global_nn.element_cu_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Cu"));
        }
        if nn_update.update_element_zn_nn
        {
            self.element_zn_adam.update(&mut global_nn.element_zn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Zn"));
        }
        if nn_update.update_element_as_nn
        {
            self.element_as_adam.update(&mut global_nn.element_as_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element As"));
        }
        if nn_update.update_element_se_nn
        {
            self.element_se_adam.update(&mut global_nn.element_se_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Se"));
        }
        if nn_update.update_element_br_nn
        {
            self.element_br_adam.update(&mut global_nn.element_br_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Br"));
        }
        if nn_update.update_element_mo_nn
        {
            self.element_mo_adam.update(&mut global_nn.element_mo_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Mo"));
        }
        if nn_update.update_element_cd_nn
        {
            self.element_cd_adam.update(&mut global_nn.element_cd_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Cd"));
        }
        if nn_update.update_element_sn_nn
        {
            self.element_sn_adam.update(&mut global_nn.element_sn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Sn"));
        }
        if nn_update.update_element_i_nn
        {
            self.element_i_adam.update(&mut global_nn.element_i_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element I"));
        }


        if nn_update.update_amino_acid_gly_nn
        {
            self.amino_acid_gly_adam.update(&mut global_nn.amino_acid_gly_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Gly"));
        }
        if nn_update.update_amino_acid_ala_nn
        {
            self.amino_acid_ala_adam.update(&mut global_nn.amino_acid_ala_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Ala"));
        }
        if nn_update.update_amino_acid_val_nn
        {
            self.amino_acid_val_adam.update(&mut global_nn.amino_acid_val_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Val"));
        }
        if nn_update.update_amino_acid_leu_nn
        {
            self.amino_acid_leu_adam.update(&mut global_nn.amino_acid_leu_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Leu"));
        }
        if nn_update.update_amino_acid_ile_nn
        {
            self.amino_acid_ile_adam.update(&mut global_nn.amino_acid_ile_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Ile"));
        }
        if nn_update.update_amino_acid_ser_nn
        {
            self.amino_acid_ser_adam.update(&mut global_nn.amino_acid_ser_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Ser"));
        }
        if nn_update.update_amino_acid_thr_nn
        {
            self.amino_acid_thr_adam.update(&mut global_nn.amino_acid_thr_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Thr"));
        }
        if nn_update.update_amino_acid_asp_nn
        {
            self.amino_acid_asp_adam.update(&mut global_nn.amino_acid_asp_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Asp"));
        }
        if nn_update.update_amino_acid_ash_nn
        {
            self.amino_acid_ash_adam.update(&mut global_nn.amino_acid_ash_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Ash"));
        }
        if nn_update.update_amino_acid_asn_nn
        {
            self.amino_acid_asn_adam.update(&mut global_nn.amino_acid_asn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Asn"));
        }
        if nn_update.update_amino_acid_glu_nn
        {
            self.amino_acid_glu_adam.update(&mut global_nn.amino_acid_glu_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Glu"));
        }
        if nn_update.update_amino_acid_glh_nn
        {
            self.amino_acid_glh_adam.update(&mut global_nn.amino_acid_glh_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Glh"));
        }
        if nn_update.update_amino_acid_gln_nn
        {
            self.amino_acid_gln_adam.update(&mut global_nn.amino_acid_gln_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Gln"));
        }
        if nn_update.update_amino_acid_lys_nn
        {
            self.amino_acid_lys_adam.update(&mut global_nn.amino_acid_lys_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Lys"));
        }
        if nn_update.update_amino_acid_lyn_nn
        {
            self.amino_acid_lyn_adam.update(&mut global_nn.amino_acid_lyn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Lyn"));
        }
        if nn_update.update_amino_acid_arg_nn
        {
            self.amino_acid_arg_adam.update(&mut global_nn.amino_acid_arg_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Arg"));
        }
/*
        if nn_update.update_amino_acid_arn_nn
        {
            self.amino_acid_arn_adam.update(&mut global_nn.amino_acid_arn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Arn"));
        }
*/
        if nn_update.update_amino_acid_cys_nn
        {
            self.amino_acid_cys_adam.update(&mut global_nn.amino_acid_cys_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Cys"));
        }
        if nn_update.update_amino_acid_cyx_nn
        {
            self.amino_acid_cyx_adam.update(&mut global_nn.amino_acid_cyx_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Cyx"));
        }
/*
        if nn_update.update_amino_acid_cym_nn
        {
            self.amino_acid_cym_adam.update(&mut global_nn.amino_acid_cym_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Cym"));
        }
*/
        if nn_update.update_amino_acid_met_nn
        {
            self.amino_acid_met_adam.update(&mut global_nn.amino_acid_met_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Met"));
        }
        if nn_update.update_amino_acid_hid_nn
        {
            self.amino_acid_hid_adam.update(&mut global_nn.amino_acid_hid_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Hid"));
        }
        if nn_update.update_amino_acid_hie_nn
        {
            self.amino_acid_hie_adam.update(&mut global_nn.amino_acid_hie_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Hie"));
        }
        if nn_update.update_amino_acid_hip_nn
        {
            self.amino_acid_hip_adam.update(&mut global_nn.amino_acid_hip_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Hip"));
        }
        if nn_update.update_amino_acid_phe_nn
        {
            self.amino_acid_phe_adam.update(&mut global_nn.amino_acid_phe_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Phe"));
        }
        if nn_update.update_amino_acid_tyr_nn
        {
            self.amino_acid_tyr_adam.update(&mut global_nn.amino_acid_tyr_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Tyr"));
        }
        if nn_update.update_amino_acid_trp_nn
        {
            self.amino_acid_trp_adam.update(&mut global_nn.amino_acid_trp_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Trp"));
        }
        if nn_update.update_amino_acid_pro_nn
        {
            self.amino_acid_pro_adam.update(&mut global_nn.amino_acid_pro_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Pro"));
        }


        if nn_update.update_head_h_nn
        {
            self.head_h_adam.update(&mut global_nn.head_h_nn, &global_grads).expect(&error_nn_para_update("Adam", "Head H"));
        }
        if nn_update.update_tail_o_nn
        {
            self.tail_o_adam.update(&mut global_nn.tail_o_nn, &global_grads).expect(&error_nn_para_update("Adam", "Tail O"));
        }
        if nn_update.update_tail_h_nn
        {
            self.tail_h_adam.update(&mut global_nn.tail_h_nn, &global_grads).expect(&error_nn_para_update("Adam", "Tail H"));
        }
        if nn_update.update_head_ace_nn
        {
            self.head_ace_adam.update(&mut global_nn.head_ace_nn, &global_grads).expect(&error_nn_para_update("Adam", "Head ACE"));
        }
        if nn_update.update_tail_nme_nn
        {
            self.tail_nme_adam.update(&mut global_nn.tail_nme_nn, &global_grads).expect(&error_nn_para_update("Adam", "Tail NME"));
        }
        if nn_update.update_tail_nhe_nn
        {
            self.tail_nhe_adam.update(&mut global_nn.tail_nhe_nn, &global_grads).expect(&error_nn_para_update("Adam", "Tail NHE"));
        }


        if nn_update.update_molecule_wat_nn
        {
            self.molecule_wat_adam.update(&mut global_nn.molecule_wat_nn, &global_grads).expect(&error_nn_para_update("Adam", "Molecule Wat"));
        }
    }
}










