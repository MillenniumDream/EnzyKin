//! Initialization, saving, and loading of the global long-ranged interaction parameters for protein system
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::fmt::Debug;
use crate::common::constants::{Element, ANGSTROM_TO_BOHR, KCALMOL_TO_HARTREE};
use crate::common::error::*;
use phf::phf_map;










/// The basic structure containing the long-ranged interaction parameters for an atom
///
/// # Fields
/// ```
/// atom_type: the element type of the atom
/// vdw_r: the van der Waals radius of the atom (Unit: Å)
/// epsilon: the depth of the potential well (Unit: kcal/mol)
/// charge: partial charge of the atom (Unit: e)
/// ```
#[derive(Debug, Clone)]
pub struct AtomPara
{
    pub atom_type: Element,
    pub vdw_r: f64,
    pub epsilon: f64,
    pub charge: f64,
}





/// The basic structure containing the long-ranged interaction parameters (in a.u.) for an atom
///
/// # Fields
/// ```
/// atom_type: the element type of the atom
/// vdw_r: the van der Waals radius of the atom (Unit: bohr)
/// epsilon: the depth of the potential well (Unit: Hartree)
/// charge: partial charge of the atom (Unit: e)
/// ```
#[derive(Debug, Clone)]
pub struct AtomParaAu
{
    pub atom_type: Element,
    pub vdw_r: f64,
    pub epsilon: f64,
    pub charge: f64,
}





/// Of note, the atomic long-ranged interaction parameters are derived from NN, relying on its chemical environment.
/// The parameters here are just for initialization.
pub const H_PARA: AtomPara = AtomPara {atom_type: Element::H,  vdw_r: 1.4593,  epsilon: 0.0208,  charge: 0.0};
pub const H_P_PARA: AtomPara = AtomPara {atom_type: Element::H,  vdw_r: 0.3019,  epsilon: 0.0047,  charge: 1.0};
pub const B_PARA: AtomPara = AtomPara {atom_type: Element::B,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const C_PARA: AtomPara = AtomPara {atom_type: Element::C,  vdw_r: 1.9069,  epsilon: 0.1078,  charge: 0.0};
pub const N_PARA: AtomPara = AtomPara {atom_type: Element::N,  vdw_r: 1.8886,  epsilon: 0.0858,  charge: 0.0};
pub const O_PARA: AtomPara = AtomPara {atom_type: Element::O,  vdw_r: 1.8200,  epsilon: 0.0930,  charge: 0.0};
pub const F_PARA: AtomPara = AtomPara {atom_type: Element::F,  vdw_r: 1.7029,  epsilon: 0.0832,  charge: 0.0};
pub const F_M_PARA: AtomPara = AtomPara {atom_type: Element::F,  vdw_r: 1.7029,  epsilon: 0.0832,  charge: -1.0};
pub const NA_PARA: AtomPara = AtomPara {atom_type: Element::Na,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const NA_P_PARA: AtomPara = AtomPara {atom_type: Element::Na,  vdw_r: 0.0,  epsilon: 0.0,  charge: 1.0};
pub const MG_PARA: AtomPara = AtomPara {atom_type: Element::Mg,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const MG_2P_PARA: AtomPara = AtomPara {atom_type: Element::Mg,  vdw_r: 0.0,  epsilon: 0.0,  charge: 2.0};
pub const SI_PARA: AtomPara = AtomPara {atom_type: Element::Si,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const P_PARA: AtomPara = AtomPara {atom_type: Element::P,  vdw_r: 2.0732,  epsilon: 0.2295,  charge: 0.0};
pub const S_PARA: AtomPara = AtomPara {atom_type: Element::S,  vdw_r: 1.9825,  epsilon: 0.2824,  charge: 0.0};
pub const CL_PARA: AtomPara = AtomPara {atom_type: Element::Cl,  vdw_r: 1.9452,  epsilon: 0.2638,  charge: 0.0};
pub const CL_M_PARA: AtomPara = AtomPara {atom_type: Element::Cl,  vdw_r: 1.9452,  epsilon: 0.2638,  charge: -1.0};
pub const K_PARA: AtomPara = AtomPara {atom_type: Element::K,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const K_P_PARA: AtomPara = AtomPara {atom_type: Element::K,  vdw_r: 0.0,  epsilon: 0.0,  charge: 1.0};
pub const CA_PARA: AtomPara = AtomPara {atom_type: Element::Ca,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const CA_2P_PARA: AtomPara = AtomPara {atom_type: Element::Ca,  vdw_r: 0.0,  epsilon: 0.0,  charge: 2.0};
pub const V_PARA: AtomPara = AtomPara {atom_type: Element::V,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const CR_PARA: AtomPara = AtomPara {atom_type: Element::Cr,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const CR_3P_PARA: AtomPara = AtomPara {atom_type: Element::Cr,  vdw_r: 0.0,  epsilon: 0.0,  charge: 3.0};
pub const MN_PARA: AtomPara = AtomPara {atom_type: Element::Mn,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const MN_2P_PARA: AtomPara = AtomPara {atom_type: Element::Mn,  vdw_r: 0.0,  epsilon: 0.0,  charge: 2.0};
pub const FE_PARA: AtomPara = AtomPara {atom_type: Element::Fe,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const FE_2P_PARA: AtomPara = AtomPara {atom_type: Element::Fe,  vdw_r: 0.0,  epsilon: 0.0,  charge: 2.0};
pub const FE_3P_PARA: AtomPara = AtomPara {atom_type: Element::Fe,  vdw_r: 0.0,  epsilon: 0.0,  charge: 3.0};
pub const CO_PARA: AtomPara = AtomPara {atom_type: Element::Co,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const CO_2P_PARA: AtomPara = AtomPara {atom_type: Element::Co,  vdw_r: 0.0,  epsilon: 0.0,  charge: 2.0};
pub const NI_PARA: AtomPara = AtomPara {atom_type: Element::Ni,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const NI_2P_PARA: AtomPara = AtomPara {atom_type: Element::Ni,  vdw_r: 0.0,  epsilon: 0.0,  charge: 2.0};
pub const CU_PARA: AtomPara = AtomPara {atom_type: Element::Cu,  vdw_r: 2.2100,  epsilon: 0.1729,  charge: 0.0};
pub const CU_2P_PARA: AtomPara = AtomPara {atom_type: Element::Cu,  vdw_r: 2.2100,  epsilon: 0.1729,  charge: 2.0};
pub const ZN_PARA: AtomPara = AtomPara {atom_type: Element::Zn,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const ZN_2P_PARA: AtomPara = AtomPara {atom_type: Element::Zn,  vdw_r: 0.0,  epsilon: 0.0,  charge: 2.0};
pub const AS_PARA: AtomPara = AtomPara {atom_type: Element::As,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const SE_PARA: AtomPara = AtomPara {atom_type: Element::Se,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const SE_2M_PARA: AtomPara = AtomPara {atom_type: Element::Se,  vdw_r: 0.0,  epsilon: 0.0,  charge: -2.0};
pub const SE_4P_PARA: AtomPara = AtomPara {atom_type: Element::Se,  vdw_r: 0.0,  epsilon: 0.0,  charge: 4.0};
pub const BR_PARA: AtomPara = AtomPara {atom_type: Element::Br,  vdw_r: 2.0275,  epsilon: 0.3932,  charge: 0.0};
pub const BR_M_PARA: AtomPara = AtomPara {atom_type: Element::Br,  vdw_r: 2.0275,  epsilon: 0.3932,  charge: -1.0};
pub const MO_PARA: AtomPara = AtomPara {atom_type: Element::Mo,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const CD_PARA: AtomPara = AtomPara {atom_type: Element::Cd,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const SN_PARA: AtomPara = AtomPara {atom_type: Element::Sn,  vdw_r: 0.0,  epsilon: 0.0,  charge: 0.0};
pub const SN_2P_PARA: AtomPara = AtomPara {atom_type: Element::Sn,  vdw_r: 0.0,  epsilon: 0.0,  charge: 2.0};
pub const I_PARA: AtomPara = AtomPara {atom_type: Element::I,  vdw_r: 2.1558,  epsilon: 0.4955,  charge: 0.0};
pub const I_M_PARA: AtomPara = AtomPara {atom_type: Element::I,  vdw_r: 2.1558,  epsilon: 0.4955,  charge: -1.0};



// 'STR_TO_ATOM_PARA' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
static STR_TO_ATOM_PARA: phf::Map<&'static str, AtomPara> = phf_map!
{
    "H" => H_PARA,
    "H+" => H_P_PARA,
    "B" => B_PARA,
    "C" => C_PARA,
    "N" => N_PARA,
    "O" => O_PARA,
    "F" => F_PARA,
    "F-" => F_M_PARA,
    "Na" => NA_PARA,
    "Na+" => NA_P_PARA,
    "Mg" => MG_PARA,
    "Mg2+" => MG_2P_PARA,
    "Si" => SI_PARA,
    "P" => P_PARA,
    "S" => S_PARA,
    "Cl" => CL_PARA,
    "Cl-" => CL_M_PARA,
    "K" => K_PARA,
    "K+" => K_P_PARA,
    "Ca" => CA_PARA,
    "Ca2+" => CA_2P_PARA,
    "V" => V_PARA,
    "Cr" => CR_PARA,
    "Cr3+" => CR_3P_PARA,
    "Mn" => MN_PARA,
    "Mn2+" => MN_2P_PARA,
    "Fe" => FE_PARA,
    "Fe2+" => FE_2P_PARA,
    "Fe3+" => FE_3P_PARA,
    "Co" => CO_PARA,
    "Co2+" => CO_2P_PARA,
    "Ni" => NI_PARA,
    "Ni2+" => NI_2P_PARA,
    "Cu" => CU_PARA,
    "Cu2+" => CU_2P_PARA,
    "Zn" => ZN_PARA,
    "Zn2+" => ZN_2P_PARA,
    "As" => AS_PARA,
    "Se" => SE_PARA,
    "Se2-" => SE_2M_PARA,
    "Se4+" => SE_4P_PARA,
    "Br" => BR_PARA,
    "Br-" => BR_M_PARA,
    "Mo" => MO_PARA,
    "Cd" => CD_PARA,
    "Sn" => SN_PARA,
    "Sn2+" => SN_2P_PARA,
    "I" => I_PARA,
    "I-" => I_M_PARA,
};



impl AtomPara
{
    pub fn from_str(biological_element: &str) -> Self
    {
        STR_TO_ATOM_PARA.get(biological_element).cloned().expect(&error_biological_element(biological_element))
    }
}





impl AtomParaAu
{
    pub fn from_atom_para(atom_para: &AtomPara) -> Self
    {
        AtomParaAu
        {
            atom_type: atom_para.atom_type.clone(),
            vdw_r: atom_para.vdw_r * ANGSTROM_TO_BOHR,
            epsilon: atom_para.epsilon * KCALMOL_TO_HARTREE,
            charge: atom_para.charge,
        }
    }
}










/// Long-ranged interaction parameters for a fragment with specific number of atoms
type FragmentPara<const N: usize> = [AtomPara; N];

/// The global long-ranged interaction parameters for the protein system, containing all the sub-parameters for the framents except atoms (i.e., amino acids, heads, tails, and molecules)
///
/// # Fields
/// ```
/// ```
#[derive(Debug)]
pub struct GlobalPara
{
    pub amino_acid_gly_para: FragmentPara<7>,
    pub amino_acid_ala_para: FragmentPara<10>,
    pub amino_acid_val_para: FragmentPara<16>,
    pub amino_acid_leu_para: FragmentPara<19>,
    pub amino_acid_ile_para: FragmentPara<19>,
    pub amino_acid_ser_para: FragmentPara<11>,
    pub amino_acid_thr_para: FragmentPara<14>,
    pub amino_acid_asp_para: FragmentPara<12>,
    pub amino_acid_ash_para: FragmentPara<13>,
    pub amino_acid_asn_para: FragmentPara<14>,
    pub amino_acid_glu_para: FragmentPara<15>,
    pub amino_acid_glh_para: FragmentPara<16>,
    pub amino_acid_gln_para: FragmentPara<17>,
    pub amino_acid_lys_para: FragmentPara<22>,
    pub amino_acid_lyn_para: FragmentPara<21>,
    pub amino_acid_arg_para: FragmentPara<24>,
//    pub amino_acid_arn_para: FragmentPara<23>,
    pub amino_acid_cys_para: FragmentPara<11>,
    pub amino_acid_cyx_para: FragmentPara<10>,
//    pub amino_acid_cym_para: FragmentPara<10>,
    pub amino_acid_met_para: FragmentPara<17>,
    pub amino_acid_hid_para: FragmentPara<17>,
    pub amino_acid_hie_para: FragmentPara<17>,
    pub amino_acid_hip_para: FragmentPara<18>,
    pub amino_acid_phe_para: FragmentPara<20>,
    pub amino_acid_tyr_para: FragmentPara<21>,
    pub amino_acid_trp_para: FragmentPara<24>,
    pub amino_acid_pro_para: FragmentPara<14>,

    pub amino_acid_ngly_para: FragmentPara<9>,
    pub amino_acid_nala_para: FragmentPara<12>,
    pub amino_acid_nval_para: FragmentPara<18>,
    pub amino_acid_nleu_para: FragmentPara<21>,
    pub amino_acid_nile_para: FragmentPara<21>,
    pub amino_acid_nser_para: FragmentPara<13>,
    pub amino_acid_nthr_para: FragmentPara<16>,
    pub amino_acid_nasp_para: FragmentPara<14>,
    pub amino_acid_nasn_para: FragmentPara<16>,
    pub amino_acid_nglu_para: FragmentPara<17>,
    pub amino_acid_ngln_para: FragmentPara<19>,
    pub amino_acid_nlys_para: FragmentPara<24>,
    pub amino_acid_narg_para: FragmentPara<26>,
    pub amino_acid_ncys_para: FragmentPara<13>,
    pub amino_acid_ncyx_para: FragmentPara<12>,
    pub amino_acid_nmet_para: FragmentPara<19>,
    pub amino_acid_nhid_para: FragmentPara<19>,
    pub amino_acid_nhie_para: FragmentPara<19>,
    pub amino_acid_nhip_para: FragmentPara<20>,
    pub amino_acid_nphe_para: FragmentPara<22>,
    pub amino_acid_ntyr_para: FragmentPara<23>,
    pub amino_acid_ntrp_para: FragmentPara<26>,
    pub amino_acid_npro_para: FragmentPara<16>,

    pub amino_acid_cgly_para: FragmentPara<8>,
    pub amino_acid_cala_para: FragmentPara<11>,
    pub amino_acid_cval_para: FragmentPara<17>,
    pub amino_acid_cleu_para: FragmentPara<20>,
    pub amino_acid_cile_para: FragmentPara<20>,
    pub amino_acid_cser_para: FragmentPara<12>,
    pub amino_acid_cthr_para: FragmentPara<15>,
    pub amino_acid_casp_para: FragmentPara<13>,
    pub amino_acid_casn_para: FragmentPara<15>,
    pub amino_acid_cglu_para: FragmentPara<16>,
    pub amino_acid_cgln_para: FragmentPara<18>,
    pub amino_acid_clys_para: FragmentPara<23>,
    pub amino_acid_carg_para: FragmentPara<25>,
    pub amino_acid_ccys_para: FragmentPara<12>,
    pub amino_acid_ccyx_para: FragmentPara<11>,
    pub amino_acid_cmet_para: FragmentPara<18>,
    pub amino_acid_chid_para: FragmentPara<18>,
    pub amino_acid_chie_para: FragmentPara<18>,
    pub amino_acid_chip_para: FragmentPara<19>,
    pub amino_acid_cphe_para: FragmentPara<21>,
    pub amino_acid_ctyr_para: FragmentPara<22>,
    pub amino_acid_ctrp_para: FragmentPara<25>,
    pub amino_acid_cpro_para: FragmentPara<15>,

    pub head_ace_para: FragmentPara<6>,
    pub tail_nme_para: FragmentPara<6>,
    pub tail_nhe_para: FragmentPara<3>,

    pub molecule_wat_para: FragmentPara<3>,
}










impl GlobalPara
{
    /// Initialize the global long-ranged interaction parameters based on Amber force-field
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn initialize() -> Self
    {
        let amino_acid_gly_para: FragmentPara<7> = [
                                                       AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                       AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                       AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.025200},
                                                       AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.069800},
                                                       AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.069800},
                                                       AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                       AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                   ];
        let amino_acid_ala_para: FragmentPara<10> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.033700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.082300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.182500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.060300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.060300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.060300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_val_para: FragmentPara<16> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.087500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.096900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.298500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.029700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.319200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.079100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.079100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.079100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.319200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.079100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.079100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.079100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_leu_para: FragmentPara<19> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.051800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.092200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.110200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.045700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.045700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.353100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.036100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.412100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.412100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_ile_para: FragmentPara<19> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.059700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.086900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.130300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.018700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.320400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.088200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.088200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.088200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.043000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.023600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.023600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.066000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.018600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.018600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.018600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_ser_para: FragmentPara<11> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.024900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.084300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.211700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.035200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.035200},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.654600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.427500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_thr_para: FragmentPara<14> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.038900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.100700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.365400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.004300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.243800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.064200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.064200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.064200},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.676100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.410200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_asp_para: FragmentPara<12> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.516300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.293600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.038100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.088000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.030300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.012200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.012200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.799400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.801400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.801400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.536600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.581900},
                                                    ];
        let amino_acid_ash_para: FragmentPara<13> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.034100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.086400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.031600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.048800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.048800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.646200},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.555400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.637600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.474700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_asn_para: FragmentPara<14> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.014300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.104800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.204100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.079700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.079700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.713000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.593100},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.919100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.419600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.419600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_glu_para: FragmentPara<15> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.516300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.293600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.039700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.110500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.056000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.017300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.017300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.013600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.042500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.042500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.805400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.818800},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.818800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.536600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.581900},
                                                    ];
        let amino_acid_glh_para: FragmentPara<16> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.014500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.077900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.007100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.025600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.025600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.017400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.043000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.043000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.680100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.583800},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.651100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.464100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_gln_para: FragmentPara<17> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.003100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.085000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.003600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.017100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.017100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.064500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.035200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.035200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.695100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.608600},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.940700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.425100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.425100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_lys_para: FragmentPara<22> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.347900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.274700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.240000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.142600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.009400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.036200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.036200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.018700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.010300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.010300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.047900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.062100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.062100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.014300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.113500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.113500},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.385400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.340000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.340000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.340000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.734100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.589400},
                                                    ];
        let amino_acid_lyn_para: FragmentPara<21> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.072060},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.099400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.048450},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.034000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.034000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.066120},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.010410},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.010410},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.037680},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.011550},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.011550},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.326040},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge: -0.033580},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge: -0.033580},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -1.035810},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.386040},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.386040},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_arg_para: FragmentPara<24> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.347900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.274700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.263700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.156000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.000700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.032700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.032700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.039000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.028500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.028500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.048600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.068700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.068700},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.529500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.345600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.807600},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.862700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.447800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.447800},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.862700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.447800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.447800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.734100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.589400},
                                                    ];
/*
        let amino_acid_arn_para: FragmentPara<23> = [
                                                        AtomPara {atom_type: Element::N, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::C, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::C, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::C, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::C, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::N, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::C, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::N, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::N, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::H, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::C, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                        AtomPara {atom_type: Element::O, vdw_r: 0.0, epsilon: 0.0, charge: 0.0},
                                                    ];
*/
        let amino_acid_cys_para: FragmentPara<11> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.021300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.112400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.123100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.111200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.111200},
                                                        AtomPara {atom_type: Element::S, vdw_r:  2.000000, epsilon:  0.250000, charge: -0.311900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.193300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_cyx_para: FragmentPara<10> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.042900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.076600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.079000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.091000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.091000},
                                                        AtomPara {atom_type: Element::S, vdw_r:  2.000000, epsilon:  0.250000, charge: -0.108100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
/*
        let amino_acid_cym_para: FragmentPara<10> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.035100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.050800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.241300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.112200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.112200},
                                                        AtomPara {atom_type: Element::S, vdw_r:  2.000000, epsilon:  0.250000, charge: -0.884400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
*/
        let amino_acid_met_para: FragmentPara<17> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.023700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.088000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.034200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.024100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.024100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.001800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.044000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.044000},
                                                        AtomPara {atom_type: Element::S, vdw_r:  2.000000, epsilon:  0.250000, charge: -0.273700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.053600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.068400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.068400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.068400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_hid_para: FragmentPara<17> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.018800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.088100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.046200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.040200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.040200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.026600},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.381100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.364900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.205700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.359000, epsilon:  0.015000, charge:  0.139200},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.572700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.129200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.114700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_hie_para: FragmentPara<17> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.058100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.136000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.007400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.036700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.036700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.186800},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.543200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.163500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.359000, epsilon:  0.015000, charge:  0.143500},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.279500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.333900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.220700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.186200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_hip_para: FragmentPara<18> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.347900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.274700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.135400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.121200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.041400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.081000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.081000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.001200},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.151300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.386600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.017000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.359000, epsilon:  0.015000, charge:  0.268100},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.171800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.391100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.114100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.231700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.734100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.589400},
                                                    ];
        let amino_acid_phe_para: FragmentPara<20> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.002400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.097800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.034300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.029500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.029500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.011800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.125600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.133000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.170400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.143000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.107200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.129700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.170400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.143000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.125600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.133000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_tyr_para: FragmentPara<21> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.001400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.087600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.015200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.029500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.029500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.001100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.190600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.169900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.234100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.165600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.322600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.557900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.399200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.234100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.165600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.190600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.169900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_trp_para: FragmentPara<24> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.027500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.112300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.005000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.033900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.033900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.141500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.163800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.206200},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.341800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.341200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.138000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.260100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.157200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.113400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.141700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.197200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.144700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.238700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.170000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.124300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                    ];
        let amino_acid_pro_para: FragmentPara<14> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.254800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.019200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.039100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.039100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.018900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.021300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.021300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.007000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.025300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.025300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.026600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.064100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.589600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.574800},
                                                    ];


        let amino_acid_ngly_para: FragmentPara<9> = [
                                                       AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.294300},
                                                       AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.164200},
                                                       AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.164200},
                                                       AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.164200},
                                                       AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.010000},
                                                       AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.089500},
                                                       AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.089500},
                                                       AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.616300},
                                                       AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.572200},
                                                   ];
        let amino_acid_nala_para: FragmentPara<12> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.141400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.199700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.199700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.199700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.096200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.088900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.059700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.030000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.030000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.030000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.616300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.572200},
                                                    ];
        let amino_acid_nval_para: FragmentPara<18> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.057700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.227200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.227200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.227200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.005400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.109300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.319600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.022100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.312900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.073500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.073500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.073500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.312900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.073500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.073500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.073500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.616300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.572200},
                                                    ];
        let amino_acid_nleu_para: FragmentPara<21> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.101000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.214800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.214800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.214800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.010400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.105300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.024400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.025600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.025600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.342100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.038000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.410500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.098000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.098000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.098000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.410500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.098000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.098000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.098000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_nile_para: FragmentPara<21> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.031100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.232900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.232900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.232900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.025700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.103100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.188500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.021300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.372000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.094700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.094700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.094700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.038700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.020100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.020100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.090800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.022600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.022600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.022600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_nser_para: FragmentPara<13> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.184900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.189800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.189800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.189800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.056700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.078200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.259600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.027300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.027300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.671400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.423900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.616300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.572200},
                                                    ];
        let amino_acid_nthr_para: FragmentPara<16> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.181200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.193400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.193400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.193400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.003400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.108700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.451400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge: -0.032300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.255400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.062700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.062700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.062700},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.676400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.407000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.616300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.572200},
                                                    ];
        let amino_acid_nasp_para: FragmentPara<14> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.078200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.220000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.220000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.220000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.029200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.114100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.023500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.016900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.016900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.819400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.808400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.808400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.562100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.588900},
                                                    ];
        let amino_acid_nasn_para: FragmentPara<16> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.180100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.192100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.192100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.192100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.036800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.123100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.028300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.051500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.051500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.583300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.574400},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.863400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.409700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.409700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.616300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.572200},
                                                    ];
        let amino_acid_nglu_para: FragmentPara<17> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.001700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.239100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.239100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.239100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.058800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.120200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.090900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.023200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.023200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.023600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.031500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.031500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.808700},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.818900},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.818900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.562100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.588900},
                                                    ];
        let amino_acid_ngln_para: FragmentPara<19> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.149300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.199600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.199600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.199600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.053600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.101500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.065100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.005000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.005000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.090300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.033100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.033100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.735400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.613300},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -1.003100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.442900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.442900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_nlys_para: FragmentPara<24> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.096600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.216500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.216500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.216500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.001500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.118000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.021200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.028300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.028300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.004800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.012100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.012100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.060800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.063300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.063300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.018100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.117100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.117100},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.376400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.338200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.338200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.338200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.721400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.601300},
                                                    ];
        let amino_acid_narg_para: FragmentPara<26> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.130500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.208300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.208300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.208300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.022300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.124200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.011800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.022600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.022600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.023600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.030900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.030900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.093500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.052700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.052700},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.565000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.359200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.828100},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.869300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.449400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.449400},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.869300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.449400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.449400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.721400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.601300},
                                                    ];
        let amino_acid_ncys_para: FragmentPara<13> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.132500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.202300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.202300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.202300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.092700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.141100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.119500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.118800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.118800},
                                                        AtomPara {atom_type: Element::S, vdw_r:  2.000000, epsilon:  0.250000, charge: -0.329800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.197500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_ncyx_para: FragmentPara<12> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.206900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.181500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.181500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.181500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.105500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.092200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.027700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.068000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.068000},
                                                        AtomPara {atom_type: Element::S, vdw_r:  2.000000, epsilon:  0.250000, charge: -0.098400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_nmet_para: FragmentPara<19> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.159200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.198400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.198400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.198400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.022100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.111600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.086500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.012500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.012500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.033400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.029200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.029200},
                                                        AtomPara {atom_type: Element::S, vdw_r:  2.000000, epsilon:  0.250000, charge: -0.277400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.034100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.059700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.059700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.059700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_nhid_para: FragmentPara<19> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.154200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.196300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.196300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.196300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.096400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.095800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.025900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.020900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.020900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.039900},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.381900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.363200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.212700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.359000, epsilon:  0.015000, charge:  0.138500},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.571100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.104600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.129900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_nhie_para: FragmentPara<19> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.147200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.201600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.201600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.201600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.023600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.138000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.048900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.022300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.022300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.174000},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.557900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.180400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.359000, epsilon:  0.015000, charge:  0.139700},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.278100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.332400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.234900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.196300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_nhip_para: FragmentPara<20> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.256000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.170400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.170400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.170400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.058100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.104700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.048400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.053100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.053100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.023600},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.151000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.382100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.001100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.359000, epsilon:  0.015000, charge:  0.264500},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.173900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.392100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.143300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.249500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.721400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.601300},
                                                    ];
        let amino_acid_nphe_para: FragmentPara<22> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.173700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.192100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.192100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.192100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.073300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.104100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.033000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.010400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.010400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.003100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.139150},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.137400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.160250},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.143300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.120800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.132900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.160250},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.143300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.139150},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.137400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_ntyr_para: FragmentPara<23> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.194000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.187300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.187300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.187300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.057000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.098300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.065900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.010200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.010200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.020500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.200200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.172000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.223900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.165000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.313900},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.557800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.400100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.223900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.165000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.200200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.172000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_ntrp_para: FragmentPara<26> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge:  0.191300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.188800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.188800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.188800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.042100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.116200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.054300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.022200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.022200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.165400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.178800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.219500},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.344400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.341200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.157500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.271000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.158900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.108000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.141100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.203400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.145800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.226500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.164600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.113200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.612300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.571300},
                                                    ];
        let amino_acid_npro_para: FragmentPara<16> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.202000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.312000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.312000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.012000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.121000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.115000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.100000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.100000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.526000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.500000},
                                                    ];


        let amino_acid_cgly_para: FragmentPara<8> = [
                                                       AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                       AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                       AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.249300},
                                                       AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.105600},
                                                       AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.105600},
                                                       AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.723100},
                                                       AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.785500},
                                                       AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.785500},
                                                   ];
        let amino_acid_cala_para: FragmentPara<11> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.174700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.106700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.209300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.076400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.076400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.076400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.773100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.805500},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.805500},
                                                    ];
        let amino_acid_cval_para: FragmentPara<17> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.343800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.143800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.194000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.030800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.306400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.083600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.083600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.083600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.306400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.083600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.083600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.083600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.835000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.817300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.817300},
                                                    ];
        let amino_acid_cleu_para: FragmentPara<20> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.284700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.134600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.246900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.097400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.097400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.370600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.037400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.416300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.103800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.103800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.103800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.416300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.103800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.103800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.103800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.832600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.819900},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.819900},
                                                    ];
        let amino_acid_cile_para: FragmentPara<20> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.310000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.137500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.036300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.076600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.349800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.102100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.102100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.102100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.032300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.032100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.032100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.069900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.019600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.019600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.019600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.834300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.819000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.819000},
                                                    ];
        let amino_acid_cser_para: FragmentPara<12> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.272200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.130400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.112300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.081300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.081300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.651400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.447400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.811300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.813200},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.813200},
                                                    ];
        let amino_acid_cthr_para: FragmentPara<15> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.242000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.120700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.302500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.007800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.185300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.058600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.058600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.058600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.649600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.411900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.781000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.804400},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.804400},
                                                    ];
        let amino_acid_casp_para: FragmentPara<13> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.519200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.305500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.181700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.104600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.067700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.021200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.021200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.885100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.816200},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.816200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.725600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.788700},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.788700},
                                                    ];
        let amino_acid_casn_para: FragmentPara<15> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.208000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.135800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.229900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.102300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.102300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.715300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.601000},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.908400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.415000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.415000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.805000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.814700},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.814700},
                                                    ];
        let amino_acid_cglu_para: FragmentPara<16> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.519200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.305500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.205900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.139900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.007100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.007800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.007800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.067500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.054800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge: -0.054800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.818300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.822000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.822000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.742000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.793000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.793000},
                                                    ];
        let amino_acid_cgln_para: FragmentPara<18> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.224800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.123200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.066400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.045200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.045200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.021000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.020300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.020300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.709300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.609800},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.957400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.430400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.430400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.777500},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.804200},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.804200},
                                                    ];
        let amino_acid_clys_para: FragmentPara<23> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.348100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.276400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.290300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.143800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.053800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.048200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.048200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.022700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.013400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.013400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.039200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.061100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.061100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.017600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.112100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.100000, epsilon:  0.015700, charge:  0.112100},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.374100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.337400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.337400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.337400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.848800},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.825200},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.825200},
                                                    ];
        let amino_acid_carg_para: FragmentPara<25> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.348100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.276400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.306800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.144700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.037400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.037100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.037100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.074400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.018500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.018500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.111400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.046800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.046800},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.556400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.347900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.836800},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.873700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.449300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.449300},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.873700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.449300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.449300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.855700},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.826600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.826600},
                                                    ];
        let amino_acid_ccys_para: FragmentPara<12> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.163500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.139600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.199600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.143700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.143700},
                                                        AtomPara {atom_type: Element::S, vdw_r:  2.000000, epsilon:  0.250000, charge: -0.310200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.206800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.749700},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.798100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.798100},
                                                    ];
        let amino_acid_ccyx_para: FragmentPara<11> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.131800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.093800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.194300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.122800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.122800},
                                                        AtomPara {atom_type: Element::S, vdw_r:  2.000000, epsilon:  0.250000, charge: -0.052900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.761800},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.804100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.804100},
                                                    ];
        let amino_acid_cmet_para: FragmentPara<18> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.259700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.127700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.023600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.048000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.048000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.049200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.031700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.031700},
                                                        AtomPara {atom_type: Element::S, vdw_r:  2.000000, epsilon:  0.250000, charge: -0.269200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.037600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.062500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.062500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.062500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.801300},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.810500},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.810500},
                                                    ];
        let amino_acid_chid_para: FragmentPara<18> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.173900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.110000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.104600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.056500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.056500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.029300},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.389200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.375500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.192500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.359000, epsilon:  0.015000, charge:  0.141800},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.562900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.100100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.124100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.761500},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.801600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.801600},
                                                    ];
        let amino_acid_chie_para: FragmentPara<18> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.269900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.165000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.106800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.062000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.062000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.272400},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.551700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.155800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.359000, epsilon:  0.015000, charge:  0.144800},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.267000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.331900},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.258800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.195700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.791600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.806500},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.806500},
                                                    ];
        let amino_acid_chip_para: FragmentPara<19> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.348100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.276400},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.144500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.111500},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.080000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.086800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.086800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.029800},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.150100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.388300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.025100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.359000, epsilon:  0.015000, charge:  0.269400},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.168300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.391300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.125600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.233600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.803200},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.817700},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.817700},
                                                    ];
        let amino_acid_cphe_para: FragmentPara<21> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.182500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.109800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.095900},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.044300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.044300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.055200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.130000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.140800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.184700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.146100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.094400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.128000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.184700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.146100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.130000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.140800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.766000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.802600},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.802600},
                                                    ];
        let amino_acid_ctyr_para: FragmentPara<22> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.201500},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.109200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.075200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.049000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.049000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.024300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.192200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.178000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.245800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.167300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.339500},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.721000, epsilon:  0.210400, charge: -0.564300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.000000, epsilon:  0.000000, charge:  0.401700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.245800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.167300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.192200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.178000},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.781700},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.807000},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.807000},
                                                    ];
        let amino_acid_ctrp_para: FragmentPara<25> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.382100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.268100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.208400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.127200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.074200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.049700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.049700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.079600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.180800},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.409000, epsilon:  0.015000, charge:  0.204300},
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.331600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.341300},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.122200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.259400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.156700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.102000},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.140100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.228700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.150700},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge: -0.183700},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.459000, epsilon:  0.015000, charge:  0.149100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.107800},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.765800},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.801100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.801100},
                                                    ];
        let amino_acid_cpro_para: FragmentPara<15> = [
                                                        AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.280200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.043400},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.033100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.033100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge:  0.046600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.017200},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.017200},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.054300},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.038100},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.038100},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.133600},
                                                        AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.077600},
                                                        AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.663100},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.769700},
                                                        AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.769700},
                                                    ];


        let head_ace_para: FragmentPara<6> = [
                                                     AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.112300},
                                                     AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.366200},
                                                     AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.112300},
                                                     AtomPara {atom_type: Element::H, vdw_r:  1.487000, epsilon:  0.015700, charge:  0.112300},
                                                     AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.086000, charge:  0.597200},
                                                     AtomPara {atom_type: Element::O, vdw_r:  1.661200, epsilon:  0.210000, charge: -0.567900},
                                                 ];
        let tail_nme_para: FragmentPara<6> = [
                                                     AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.415700},
                                                     AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.271900},
                                                     AtomPara {atom_type: Element::C, vdw_r:  1.908000, epsilon:  0.109400, charge: -0.149000},
                                                     AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.097600},
                                                     AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.097600},
                                                     AtomPara {atom_type: Element::H, vdw_r:  1.387000, epsilon:  0.015700, charge:  0.097600},
                                                 ];
        let tail_nhe_para: FragmentPara<3> = [
                                                     AtomPara {atom_type: Element::N, vdw_r:  1.824000, epsilon:  0.170000, charge: -0.463000},
                                                     AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.231500},
                                                     AtomPara {atom_type: Element::H, vdw_r:  0.600000, epsilon:  0.015700, charge:  0.231500},
                                                 ];


        let molecule_wat_para: FragmentPara<3> = [
                                                     AtomPara {atom_type: Element::O, vdw_r:  1.768300, epsilon:  0.152000, charge: -0.834000},
                                                     AtomPara {atom_type: Element::H, vdw_r:  0.301900, epsilon:  0.047000, charge:  0.417000},
                                                     AtomPara {atom_type: Element::H, vdw_r:  0.301900, epsilon:  0.047000, charge:  0.417000},
                                                 ];


        GlobalPara
        {
            amino_acid_gly_para,
            amino_acid_ala_para,
            amino_acid_val_para,
            amino_acid_leu_para,
            amino_acid_ile_para,
            amino_acid_ser_para,
            amino_acid_thr_para,
            amino_acid_asp_para,
            amino_acid_ash_para,
            amino_acid_asn_para,
            amino_acid_glu_para,
            amino_acid_glh_para,
            amino_acid_gln_para,
            amino_acid_lys_para,
            amino_acid_lyn_para,
            amino_acid_arg_para,
//            amino_acid_arn_para,
            amino_acid_cys_para,
            amino_acid_cyx_para,
//            amino_acid_cym_para,
            amino_acid_met_para,
            amino_acid_hid_para,
            amino_acid_hie_para,
            amino_acid_hip_para,
            amino_acid_phe_para,
            amino_acid_tyr_para,
            amino_acid_trp_para,
            amino_acid_pro_para,

            amino_acid_ngly_para,
            amino_acid_nala_para,
            amino_acid_nval_para,
            amino_acid_nleu_para,
            amino_acid_nile_para,
            amino_acid_nser_para,
            amino_acid_nthr_para,
            amino_acid_nasp_para,
            amino_acid_nasn_para,
            amino_acid_nglu_para,
            amino_acid_ngln_para,
            amino_acid_nlys_para,
            amino_acid_narg_para,
            amino_acid_ncys_para,
            amino_acid_ncyx_para,
            amino_acid_nmet_para,
            amino_acid_nhid_para,
            amino_acid_nhie_para,
            amino_acid_nhip_para,
            amino_acid_nphe_para,
            amino_acid_ntyr_para,
            amino_acid_ntrp_para,
            amino_acid_npro_para,

            amino_acid_cgly_para,
            amino_acid_cala_para,
            amino_acid_cval_para,
            amino_acid_cleu_para,
            amino_acid_cile_para,
            amino_acid_cser_para,
            amino_acid_cthr_para,
            amino_acid_casp_para,
            amino_acid_casn_para,
            amino_acid_cglu_para,
            amino_acid_cgln_para,
            amino_acid_clys_para,
            amino_acid_carg_para,
            amino_acid_ccys_para,
            amino_acid_ccyx_para,
            amino_acid_cmet_para,
            amino_acid_chid_para,
            amino_acid_chie_para,
            amino_acid_chip_para,
            amino_acid_cphe_para,
            amino_acid_ctyr_para,
            amino_acid_ctrp_para,
            amino_acid_cpro_para,

            head_ace_para,
            tail_nme_para,
            tail_nhe_para,

            molecule_wat_para,
        }
    }





    /// Get the long-ranged interaction parameters for a given fragment
    ///
    /// # Parameters
    /// ```
    /// fragment_name: the input fragment name for parameters extract
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn get_fragment_para(&self, fragment_name: &str) -> &[AtomPara]
    {
        match fragment_name
        {
            "GLY" => &self.amino_acid_gly_para,
            "ALA" => &self.amino_acid_ala_para,
            "VAL" => &self.amino_acid_val_para,
            "LEU" => &self.amino_acid_leu_para,
            "ILE" => &self.amino_acid_ile_para,
            "SER" => &self.amino_acid_ser_para,
            "THR" => &self.amino_acid_thr_para,
            "ASP" => &self.amino_acid_asp_para,
            "ASH" => &self.amino_acid_ash_para,
            "ASN" => &self.amino_acid_asn_para,
            "GLU" => &self.amino_acid_glu_para,
            "GLH" => &self.amino_acid_glh_para,
            "GLN" => &self.amino_acid_gln_para,
            "LYS" => &self.amino_acid_lys_para,
            "LYN" => &self.amino_acid_lyn_para,
            "ARG" => &self.amino_acid_arg_para,
//            "ARN" => &self.amino_acid_arn_para,
            "CYS" => &self.amino_acid_cys_para,
            "CYX" => &self.amino_acid_cyx_para,
//            "CYM" => &self.amino_acid_cym_para,
            "MET" => &self.amino_acid_met_para,
            "HID" => &self.amino_acid_hid_para,
            "HIE" => &self.amino_acid_hie_para,
            "HIP" => &self.amino_acid_hip_para,
            "PHE" => &self.amino_acid_phe_para,
            "TYR" => &self.amino_acid_tyr_para,
            "TRP" => &self.amino_acid_trp_para,
            "PRO" => &self.amino_acid_pro_para,

            "NGLY" => &self.amino_acid_ngly_para,
            "NALA" => &self.amino_acid_nala_para,
            "NVAL" => &self.amino_acid_nval_para,
            "NLEU" => &self.amino_acid_nleu_para,
            "NILE" => &self.amino_acid_nile_para,
            "NSER" => &self.amino_acid_nser_para,
            "NTHR" => &self.amino_acid_nthr_para,
            "NASP" => &self.amino_acid_nasp_para,
            "NASN" => &self.amino_acid_nasn_para,
            "NGLU" => &self.amino_acid_nglu_para,
            "NGLN" => &self.amino_acid_ngln_para,
            "NLYS" => &self.amino_acid_nlys_para,
            "NARG" => &self.amino_acid_narg_para,
            "NCYS" => &self.amino_acid_ncys_para,
            "NCYX" => &self.amino_acid_ncyx_para,
            "NMET" => &self.amino_acid_nmet_para,
            "NHID" => &self.amino_acid_nhid_para,
            "NHIE" => &self.amino_acid_nhie_para,
            "NHIP" => &self.amino_acid_nhip_para,
            "NPHE" => &self.amino_acid_nphe_para,
            "NTYR" => &self.amino_acid_ntyr_para,
            "NTRP" => &self.amino_acid_ntrp_para,
            "NPRO" => &self.amino_acid_npro_para,

            "CGLY" => &self.amino_acid_cgly_para,
            "CALA" => &self.amino_acid_cala_para,
            "CVAL" => &self.amino_acid_cval_para,
            "CLEU" => &self.amino_acid_cleu_para,
            "CILE" => &self.amino_acid_cile_para,
            "CSER" => &self.amino_acid_cser_para,
            "CTHR" => &self.amino_acid_cthr_para,
            "CASP" => &self.amino_acid_casp_para,
            "CASN" => &self.amino_acid_casn_para,
            "CGLU" => &self.amino_acid_cglu_para,
            "CGLN" => &self.amino_acid_cgln_para,
            "CLYS" => &self.amino_acid_clys_para,
            "CARG" => &self.amino_acid_carg_para,
            "CCYS" => &self.amino_acid_ccys_para,
            "CCYX" => &self.amino_acid_ccyx_para,
            "CMET" => &self.amino_acid_cmet_para,
            "CHID" => &self.amino_acid_chid_para,
            "CHIE" => &self.amino_acid_chie_para,
            "CHIP" => &self.amino_acid_chip_para,
            "CPHE" => &self.amino_acid_cphe_para,
            "CTYR" => &self.amino_acid_ctyr_para,
            "CTRP" => &self.amino_acid_ctrp_para,
            "CPRO" => &self.amino_acid_cpro_para,

            "ACE" => &self.head_ace_para,
            "NME" => &self.tail_nme_para,
            "NHE" => &self.tail_nhe_para,

            "WAT" => &self.molecule_wat_para,

            _ => panic!("{}", error_unparameterized_fragment(fragment_name)),
        }
    }





    /// Get the mutable long-ranged interaction parameters for a given fragment
    ///
    /// # Parameters
    /// ```
    /// fragment_name: the input fragment name for parameters extract
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn get_fragment_para_mut(&mut self, fragment_name: &str) -> &mut [AtomPara]
    {
        match fragment_name
        {
            "GLY" => &mut self.amino_acid_gly_para,
            "ALA" => &mut self.amino_acid_ala_para,
            "VAL" => &mut self.amino_acid_val_para,
            "LEU" => &mut self.amino_acid_leu_para,
            "ILE" => &mut self.amino_acid_ile_para,
            "SER" => &mut self.amino_acid_ser_para,
            "THR" => &mut self.amino_acid_thr_para,
            "ASP" => &mut self.amino_acid_asp_para,
            "ASH" => &mut self.amino_acid_ash_para,
            "ASN" => &mut self.amino_acid_asn_para,
            "GLU" => &mut self.amino_acid_glu_para,
            "GLH" => &mut self.amino_acid_glh_para,
            "GLN" => &mut self.amino_acid_gln_para,
            "LYS" => &mut self.amino_acid_lys_para,
            "LYN" => &mut self.amino_acid_lyn_para,
            "ARG" => &mut self.amino_acid_arg_para,
//            "ARN" => &mut self.amino_acid_arn_para,
            "CYS" => &mut self.amino_acid_cys_para,
            "CYX" => &mut self.amino_acid_cyx_para,
//            "CYM" => &mut self.amino_acid_cym_para,
            "MET" => &mut self.amino_acid_met_para,
            "HID" => &mut self.amino_acid_hid_para,
            "HIE" => &mut self.amino_acid_hie_para,
            "HIP" => &mut self.amino_acid_hip_para,
            "PHE" => &mut self.amino_acid_phe_para,
            "TYR" => &mut self.amino_acid_tyr_para,
            "TRP" => &mut self.amino_acid_trp_para,
            "PRO" => &mut self.amino_acid_pro_para,

            "NGLY" => &mut self.amino_acid_ngly_para,
            "NALA" => &mut self.amino_acid_nala_para,
            "NVAL" => &mut self.amino_acid_nval_para,
            "NLEU" => &mut self.amino_acid_nleu_para,
            "NILE" => &mut self.amino_acid_nile_para,
            "NSER" => &mut self.amino_acid_nser_para,
            "NTHR" => &mut self.amino_acid_nthr_para,
            "NASP" => &mut self.amino_acid_nasp_para,
            "NASN" => &mut self.amino_acid_nasn_para,
            "NGLU" => &mut self.amino_acid_nglu_para,
            "NGLN" => &mut self.amino_acid_ngln_para,
            "NLYS" => &mut self.amino_acid_nlys_para,
            "NARG" => &mut self.amino_acid_narg_para,
            "NCYS" => &mut self.amino_acid_ncys_para,
            "NCYX" => &mut self.amino_acid_ncyx_para,
            "NMET" => &mut self.amino_acid_nmet_para,
            "NHID" => &mut self.amino_acid_nhid_para,
            "NHIE" => &mut self.amino_acid_nhie_para,
            "NHIP" => &mut self.amino_acid_nhip_para,
            "NPHE" => &mut self.amino_acid_nphe_para,
            "NTYR" => &mut self.amino_acid_ntyr_para,
            "NTRP" => &mut self.amino_acid_ntrp_para,
            "NPRO" => &mut self.amino_acid_npro_para,

            "CGLY" => &mut self.amino_acid_cgly_para,
            "CALA" => &mut self.amino_acid_cala_para,
            "CVAL" => &mut self.amino_acid_cval_para,
            "CLEU" => &mut self.amino_acid_cleu_para,
            "CILE" => &mut self.amino_acid_cile_para,
            "CSER" => &mut self.amino_acid_cser_para,
            "CTHR" => &mut self.amino_acid_cthr_para,
            "CASP" => &mut self.amino_acid_casp_para,
            "CASN" => &mut self.amino_acid_casn_para,
            "CGLU" => &mut self.amino_acid_cglu_para,
            "CGLN" => &mut self.amino_acid_cgln_para,
            "CLYS" => &mut self.amino_acid_clys_para,
            "CARG" => &mut self.amino_acid_carg_para,
            "CCYS" => &mut self.amino_acid_ccys_para,
            "CCYX" => &mut self.amino_acid_ccyx_para,
            "CMET" => &mut self.amino_acid_cmet_para,
            "CHID" => &mut self.amino_acid_chid_para,
            "CHIE" => &mut self.amino_acid_chie_para,
            "CHIP" => &mut self.amino_acid_chip_para,
            "CPHE" => &mut self.amino_acid_cphe_para,
            "CTYR" => &mut self.amino_acid_ctyr_para,
            "CTRP" => &mut self.amino_acid_ctrp_para,
            "CPRO" => &mut self.amino_acid_cpro_para,

            "ACE" => &mut self.head_ace_para,
            "NME" => &mut self.tail_nme_para,
            "NHE" => &mut self.tail_nhe_para,

            "WAT" => &mut self.molecule_wat_para,

            _ => panic!("{}", error_unparameterized_fragment(fragment_name)),
        }
    }





    /// Save the global long-ranged interaction parameters
    ///
    /// # Parameters
    /// ```
    /// dir: the directory for global long-ranged interaction parameters saving
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

        // Create the handle of the output file
        let para_output_file: PathBuf = dir.as_ref().join("GlobalPara");
        let mut para_output = File::create(&para_output_file).expect(&error_file("creating", &para_output_file));
        para_output.write_all(b"Long-ranged interaction parameters of all the fragments\n").expect(&error_file("writing", &para_output_file));

        // Write the parameters of all the fragments in sequence
        save_fragment(&self, &para_output_file, "GLY");
        save_fragment(&self, &para_output_file, "ALA");
        save_fragment(&self, &para_output_file, "VAL");
        save_fragment(&self, &para_output_file, "LEU");
        save_fragment(&self, &para_output_file, "ILE");
        save_fragment(&self, &para_output_file, "SER");
        save_fragment(&self, &para_output_file, "THR");
        save_fragment(&self, &para_output_file, "ASP");
        save_fragment(&self, &para_output_file, "ASH");
        save_fragment(&self, &para_output_file, "ASN");
        save_fragment(&self, &para_output_file, "GLU");
        save_fragment(&self, &para_output_file, "GLH");
        save_fragment(&self, &para_output_file, "GLN");
        save_fragment(&self, &para_output_file, "LYS");
        save_fragment(&self, &para_output_file, "LYN");
        save_fragment(&self, &para_output_file, "ARG");
//        save_fragment(&self, &para_output_file, "ARN");
        save_fragment(&self, &para_output_file, "CYS");
        save_fragment(&self, &para_output_file, "CYX");
//        save_fragment(&self, &para_output_file, "CYM");
        save_fragment(&self, &para_output_file, "MET");
        save_fragment(&self, &para_output_file, "HID");
        save_fragment(&self, &para_output_file, "HIE");
        save_fragment(&self, &para_output_file, "HIP");
        save_fragment(&self, &para_output_file, "PHE");
        save_fragment(&self, &para_output_file, "TYR");
        save_fragment(&self, &para_output_file, "TRP");
        save_fragment(&self, &para_output_file, "PRO");

        save_fragment(&self, &para_output_file, "NGLY");
        save_fragment(&self, &para_output_file, "NALA");
        save_fragment(&self, &para_output_file, "NVAL");
        save_fragment(&self, &para_output_file, "NLEU");
        save_fragment(&self, &para_output_file, "NILE");
        save_fragment(&self, &para_output_file, "NSER");
        save_fragment(&self, &para_output_file, "NTHR");
        save_fragment(&self, &para_output_file, "NASP");
        save_fragment(&self, &para_output_file, "NASN");
        save_fragment(&self, &para_output_file, "NGLU");
        save_fragment(&self, &para_output_file, "NGLN");
        save_fragment(&self, &para_output_file, "NLYS");
        save_fragment(&self, &para_output_file, "NARG");
        save_fragment(&self, &para_output_file, "NCYS");
        save_fragment(&self, &para_output_file, "NCYX");
        save_fragment(&self, &para_output_file, "NMET");
        save_fragment(&self, &para_output_file, "NHID");
        save_fragment(&self, &para_output_file, "NHIE");
        save_fragment(&self, &para_output_file, "NHIP");
        save_fragment(&self, &para_output_file, "NPHE");
        save_fragment(&self, &para_output_file, "NTYR");
        save_fragment(&self, &para_output_file, "NTRP");
        save_fragment(&self, &para_output_file, "NPRO");

        save_fragment(&self, &para_output_file, "CGLY");
        save_fragment(&self, &para_output_file, "CALA");
        save_fragment(&self, &para_output_file, "CVAL");
        save_fragment(&self, &para_output_file, "CLEU");
        save_fragment(&self, &para_output_file, "CILE");
        save_fragment(&self, &para_output_file, "CSER");
        save_fragment(&self, &para_output_file, "CTHR");
        save_fragment(&self, &para_output_file, "CASP");
        save_fragment(&self, &para_output_file, "CASN");
        save_fragment(&self, &para_output_file, "CGLU");
        save_fragment(&self, &para_output_file, "CGLN");
        save_fragment(&self, &para_output_file, "CLYS");
        save_fragment(&self, &para_output_file, "CARG");
        save_fragment(&self, &para_output_file, "CCYS");
        save_fragment(&self, &para_output_file, "CCYX");
        save_fragment(&self, &para_output_file, "CMET");
        save_fragment(&self, &para_output_file, "CHID");
        save_fragment(&self, &para_output_file, "CHIE");
        save_fragment(&self, &para_output_file, "CHIP");
        save_fragment(&self, &para_output_file, "CPHE");
        save_fragment(&self, &para_output_file, "CTYR");
        save_fragment(&self, &para_output_file, "CTRP");
        save_fragment(&self, &para_output_file, "CPRO");

        save_fragment(&self, &para_output_file, "ACE");
        save_fragment(&self, &para_output_file, "NME");
        save_fragment(&self, &para_output_file, "NHE");

        save_fragment(&self, &para_output_file, "WAT");



        // An inner function to save a specific fragment parameters
        fn save_fragment<P: AsRef<Path> + Debug>(global_para: &GlobalPara, para_output_file: P, fragment_name: &str)
        {
            // Save the header
            let mut para_output = File::options().append(true).open(&para_output_file).expect(&error_file("opening", &para_output_file));
            para_output.write_all(format!("{fragment_name}\n").as_bytes()).expect(&error_file("writing", &para_output_file));
            para_output.write_all(b"atom_type           vdw_r         epsilon          charge\n").expect(&error_file("writing", &para_output_file));

            // Save the parameters for the specific fragment
            let fragment_para: &[AtomPara] = global_para.get_fragment_para(fragment_name);
            for i in 0..fragment_para.len()
            {
                para_output.write_all(format!("    {:?}     {:15.8} {:15.8} {:15.8}\n", fragment_para[i].atom_type, fragment_para[i].vdw_r, fragment_para[i].epsilon, fragment_para[i].charge).as_bytes()).expect(&error_file("writing", &para_output_file));
            }
            para_output.write_all(b"\n").expect(&error_file("writing", &para_output_file));
        }
    }





    /// Load the global long-ranged interaction parameters
    ///
    /// # Parameters
    /// ```
    /// dir: the directory for global long-ranged interaction parameters loading
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn load<P: AsRef<Path> + Debug>(dir: P) -> Self
    {
        let filename: PathBuf = dir.as_ref().join("GlobalPara");
        let mut global_para: GlobalPara = GlobalPara::initialize();

        // Read in the parameters for all the fragments in sequence
        load_fragment(&mut global_para, &filename, "GLY");
        load_fragment(&mut global_para, &filename, "ALA");
        load_fragment(&mut global_para, &filename, "VAL");
        load_fragment(&mut global_para, &filename, "LEU");
        load_fragment(&mut global_para, &filename, "ILE");
        load_fragment(&mut global_para, &filename, "SER");
        load_fragment(&mut global_para, &filename, "THR");
        load_fragment(&mut global_para, &filename, "ASP");
        load_fragment(&mut global_para, &filename, "ASH");
        load_fragment(&mut global_para, &filename, "ASN");
        load_fragment(&mut global_para, &filename, "GLU");
        load_fragment(&mut global_para, &filename, "GLH");
        load_fragment(&mut global_para, &filename, "GLN");
        load_fragment(&mut global_para, &filename, "LYS");
        load_fragment(&mut global_para, &filename, "LYN");
        load_fragment(&mut global_para, &filename, "ARG");
//        load_fragment(&mut global_para, &filename, "ARN");
        load_fragment(&mut global_para, &filename, "CYS");
        load_fragment(&mut global_para, &filename, "CYX");
//        load_fragment(&mut global_para, &filename, "CYM");
        load_fragment(&mut global_para, &filename, "MET");
        load_fragment(&mut global_para, &filename, "HID");
        load_fragment(&mut global_para, &filename, "HIE");
        load_fragment(&mut global_para, &filename, "HIP");
        load_fragment(&mut global_para, &filename, "PHE");
        load_fragment(&mut global_para, &filename, "TYR");
        load_fragment(&mut global_para, &filename, "TRP");
        load_fragment(&mut global_para, &filename, "PRO");

        load_fragment(&mut global_para, &filename, "NGLY");
        load_fragment(&mut global_para, &filename, "NALA");
        load_fragment(&mut global_para, &filename, "NVAL");
        load_fragment(&mut global_para, &filename, "NLEU");
        load_fragment(&mut global_para, &filename, "NILE");
        load_fragment(&mut global_para, &filename, "NSER");
        load_fragment(&mut global_para, &filename, "NTHR");
        load_fragment(&mut global_para, &filename, "NASP");
        load_fragment(&mut global_para, &filename, "NASN");
        load_fragment(&mut global_para, &filename, "NGLU");
        load_fragment(&mut global_para, &filename, "NGLN");
        load_fragment(&mut global_para, &filename, "NLYS");
        load_fragment(&mut global_para, &filename, "NARG");
        load_fragment(&mut global_para, &filename, "NCYS");
        load_fragment(&mut global_para, &filename, "NCYX");
        load_fragment(&mut global_para, &filename, "NMET");
        load_fragment(&mut global_para, &filename, "NHID");
        load_fragment(&mut global_para, &filename, "NHIE");
        load_fragment(&mut global_para, &filename, "NHIP");
        load_fragment(&mut global_para, &filename, "NPHE");
        load_fragment(&mut global_para, &filename, "NTYR");
        load_fragment(&mut global_para, &filename, "NTRP");
        load_fragment(&mut global_para, &filename, "NPRO");

        load_fragment(&mut global_para, &filename, "CGLY");
        load_fragment(&mut global_para, &filename, "CALA");
        load_fragment(&mut global_para, &filename, "CVAL");
        load_fragment(&mut global_para, &filename, "CLEU");
        load_fragment(&mut global_para, &filename, "CILE");
        load_fragment(&mut global_para, &filename, "CSER");
        load_fragment(&mut global_para, &filename, "CTHR");
        load_fragment(&mut global_para, &filename, "CASP");
        load_fragment(&mut global_para, &filename, "CASN");
        load_fragment(&mut global_para, &filename, "CGLU");
        load_fragment(&mut global_para, &filename, "CGLN");
        load_fragment(&mut global_para, &filename, "CLYS");
        load_fragment(&mut global_para, &filename, "CARG");
        load_fragment(&mut global_para, &filename, "CCYS");
        load_fragment(&mut global_para, &filename, "CCYX");
        load_fragment(&mut global_para, &filename, "CMET");
        load_fragment(&mut global_para, &filename, "CHID");
        load_fragment(&mut global_para, &filename, "CHIE");
        load_fragment(&mut global_para, &filename, "CHIP");
        load_fragment(&mut global_para, &filename, "CPHE");
        load_fragment(&mut global_para, &filename, "CTYR");
        load_fragment(&mut global_para, &filename, "CTRP");
        load_fragment(&mut global_para, &filename, "CPRO");

        load_fragment(&mut global_para, &filename, "ACE");
        load_fragment(&mut global_para, &filename, "NME");
        load_fragment(&mut global_para, &filename, "NHE");

        load_fragment(&mut global_para, &filename, "WAT");

        return global_para;



        // An inner function to load a specific fragment parameters
        fn load_fragment<P: AsRef<Path> + Debug>(global_para: &mut GlobalPara, filename: P, fragment_name: &str)
        {
            let content = fs::read_to_string(&filename).expect(&error_file("reading", &filename));                // Read the whole file
            let mut line = content.lines();                 // Take the iteractor for the lines of the file

            // Find the parameters for the specific fragment
            loop
            {
                let fragment: &str = match line.next()
                {
                    Some(value) => value.trim(),
                    None => panic!("{}", error_read(&filename)),
                };
                if fragment == fragment_name
                {
                    break
                }
            }

            // Ignore the header
            line.next();

            // Read in the parameters
            let fragment_para: &mut [AtomPara] = global_para.get_fragment_para_mut(fragment_name);
            for i in 0..fragment_para.len()
            {
                let atom_para: Vec<&str> = match line.next()
                {
                    Some(value) => value.split_whitespace().collect(),
                    None => panic!("{}", error_read(&filename)),
                };
                if atom_para.len() != 4
                {
                    panic!("{}", error_read(&filename));
                }
                if Element::from_str(atom_para[0]) != fragment_para[i].atom_type
                {
                    panic!("{}", error_read(&filename));
                }
                fragment_para[i].vdw_r = atom_para[1].parse().expect(&error_read(&filename));
                fragment_para[i].epsilon = atom_para[2].parse().expect(&error_read(&filename));
                fragment_para[i].charge = atom_para[3].parse().expect(&error_read(&filename));
            }
        }
    }
}










