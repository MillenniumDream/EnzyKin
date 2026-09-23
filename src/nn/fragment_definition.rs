//! Define all the fragments in the protein-structure-based neural network
use crate::common::constants::Element;
use crate::common::error::*;
use phf::phf_map;
use savefile_derive::Savefile;





pub const H_ATOM_TYPE: [Element; 1] = [Element::H];
pub const HE_ATOM_TYPE: [Element; 1] = [Element::He];

pub const LI_ATOM_TYPE: [Element; 1] = [Element::Li];
pub const BE_ATOM_TYPE: [Element; 1] = [Element::Be];
pub const B_ATOM_TYPE: [Element; 1] = [Element::B];
pub const C_ATOM_TYPE: [Element; 1] = [Element::C];
pub const N_ATOM_TYPE: [Element; 1] = [Element::N];
pub const O_ATOM_TYPE: [Element; 1] = [Element::O];
pub const F_ATOM_TYPE: [Element; 1] = [Element::F];
pub const NE_ATOM_TYPE: [Element; 1] = [Element::Ne];

pub const NA_ATOM_TYPE: [Element; 1] = [Element::Na];
pub const MG_ATOM_TYPE: [Element; 1] = [Element::Mg];
pub const AL_ATOM_TYPE: [Element; 1] = [Element::Al];
pub const SI_ATOM_TYPE: [Element; 1] = [Element::Si];
pub const P_ATOM_TYPE: [Element; 1] = [Element::P];
pub const S_ATOM_TYPE: [Element; 1] = [Element::S];
pub const CL_ATOM_TYPE: [Element; 1] = [Element::Cl];
pub const AR_ATOM_TYPE: [Element; 1] = [Element::Ar];

pub const K_ATOM_TYPE: [Element; 1] = [Element::K];
pub const CA_ATOM_TYPE: [Element; 1] = [Element::Ca];
pub const SC_ATOM_TYPE: [Element; 1] = [Element::Sc];
pub const TI_ATOM_TYPE: [Element; 1] = [Element::Ti];
pub const V_ATOM_TYPE: [Element; 1] = [Element::V];
pub const CR_ATOM_TYPE: [Element; 1] = [Element::Cr];
pub const MN_ATOM_TYPE: [Element; 1] = [Element::Mn];
pub const FE_ATOM_TYPE: [Element; 1] = [Element::Fe];
pub const CO_ATOM_TYPE: [Element; 1] = [Element::Co];
pub const NI_ATOM_TYPE: [Element; 1] = [Element::Ni];
pub const CU_ATOM_TYPE: [Element; 1] = [Element::Cu];
pub const ZN_ATOM_TYPE: [Element; 1] = [Element::Zn];
pub const GA_ATOM_TYPE: [Element; 1] = [Element::Ga];
pub const GE_ATOM_TYPE: [Element; 1] = [Element::Ge];
pub const AS_ATOM_TYPE: [Element; 1] = [Element::As];
pub const SE_ATOM_TYPE: [Element; 1] = [Element::Se];
pub const BR_ATOM_TYPE: [Element; 1] = [Element::Br];
pub const KR_ATOM_TYPE: [Element; 1] = [Element::Kr];

pub const RB_ATOM_TYPE: [Element; 1] = [Element::Rb];
pub const SR_ATOM_TYPE: [Element; 1] = [Element::Sr];
pub const Y_ATOM_TYPE: [Element; 1] = [Element::Y];
pub const ZR_ATOM_TYPE: [Element; 1] = [Element::Zr];
pub const NB_ATOM_TYPE: [Element; 1] = [Element::Nb];
pub const MO_ATOM_TYPE: [Element; 1] = [Element::Mo];
pub const TC_ATOM_TYPE: [Element; 1] = [Element::Tc];
pub const RU_ATOM_TYPE: [Element; 1] = [Element::Ru];
pub const RH_ATOM_TYPE: [Element; 1] = [Element::Rh];
pub const PD_ATOM_TYPE: [Element; 1] = [Element::Pd];
pub const AG_ATOM_TYPE: [Element; 1] = [Element::Ag];
pub const CD_ATOM_TYPE: [Element; 1] = [Element::Cd];
pub const IN_ATOM_TYPE: [Element; 1] = [Element::In];
pub const SN_ATOM_TYPE: [Element; 1] = [Element::Sn];
pub const SB_ATOM_TYPE: [Element; 1] = [Element::Sb];
pub const TE_ATOM_TYPE: [Element; 1] = [Element::Te];
pub const I_ATOM_TYPE: [Element; 1] = [Element::I];
pub const XE_ATOM_TYPE: [Element; 1] = [Element::Xe];

pub const CS_ATOM_TYPE: [Element; 1] = [Element::Cs];
pub const BA_ATOM_TYPE: [Element; 1] = [Element::Ba];
pub const HF_ATOM_TYPE: [Element; 1] = [Element::Hf];
pub const TA_ATOM_TYPE: [Element; 1] = [Element::Ta];
pub const W_ATOM_TYPE: [Element; 1] = [Element::W];
pub const RE_ATOM_TYPE: [Element; 1] = [Element::Re];
pub const OS_ATOM_TYPE: [Element; 1] = [Element::Os];
pub const IR_ATOM_TYPE: [Element; 1] = [Element::Ir];
pub const PT_ATOM_TYPE: [Element; 1] = [Element::Pt];
pub const AU_ATOM_TYPE: [Element; 1] = [Element::Au];
pub const HG_ATOM_TYPE: [Element; 1] = [Element::Hg];
pub const TL_ATOM_TYPE: [Element; 1] = [Element::Tl];
pub const PB_ATOM_TYPE: [Element; 1] = [Element::Pb];
pub const BI_ATOM_TYPE: [Element; 1] = [Element::Bi];
pub const PO_ATOM_TYPE: [Element; 1] = [Element::Po];
pub const AT_ATOM_TYPE: [Element; 1] = [Element::At];
pub const RN_ATOM_TYPE: [Element; 1] = [Element::Rn];

pub const FR_ATOM_TYPE: [Element; 1] = [Element::Fr];
pub const RA_ATOM_TYPE: [Element; 1] = [Element::Ra];
pub const RF_ATOM_TYPE: [Element; 1] = [Element::Rf];
pub const DB_ATOM_TYPE: [Element; 1] = [Element::Db];
pub const SG_ATOM_TYPE: [Element; 1] = [Element::Sg];
pub const BH_ATOM_TYPE: [Element; 1] = [Element::Bh];
pub const HS_ATOM_TYPE: [Element; 1] = [Element::Hs];
pub const MT_ATOM_TYPE: [Element; 1] = [Element::Mt];
pub const DS_ATOM_TYPE: [Element; 1] = [Element::Ds];
pub const RG_ATOM_TYPE: [Element; 1] = [Element::Rg];

pub const LA_ATOM_TYPE: [Element; 1] = [Element::La];
pub const CE_ATOM_TYPE: [Element; 1] = [Element::Ce];
pub const PR_ATOM_TYPE: [Element; 1] = [Element::Pr];
pub const ND_ATOM_TYPE: [Element; 1] = [Element::Nd];
pub const PM_ATOM_TYPE: [Element; 1] = [Element::Pm];
pub const SM_ATOM_TYPE: [Element; 1] = [Element::Sm];
pub const EU_ATOM_TYPE: [Element; 1] = [Element::Eu];
pub const GD_ATOM_TYPE: [Element; 1] = [Element::Gd];
pub const TB_ATOM_TYPE: [Element; 1] = [Element::Tb];
pub const DY_ATOM_TYPE: [Element; 1] = [Element::Dy];
pub const HO_ATOM_TYPE: [Element; 1] = [Element::Ho];
pub const ER_ATOM_TYPE: [Element; 1] = [Element::Er];
pub const TM_ATOM_TYPE: [Element; 1] = [Element::Tm];
pub const YB_ATOM_TYPE: [Element; 1] = [Element::Yb];
pub const LU_ATOM_TYPE: [Element; 1] = [Element::Lu];

pub const AC_ATOM_TYPE: [Element; 1] = [Element::Ac];
pub const TH_ATOM_TYPE: [Element; 1] = [Element::Th];
pub const PA_ATOM_TYPE: [Element; 1] = [Element::Pa];
pub const U_ATOM_TYPE: [Element; 1] = [Element::U];
pub const NP_ATOM_TYPE: [Element; 1] = [Element::Np];
pub const PU_ATOM_TYPE: [Element; 1] = [Element::Pu];
pub const AM_ATOM_TYPE: [Element; 1] = [Element::Am];
pub const CM_ATOM_TYPE: [Element; 1] = [Element::Cm];
pub const BK_ATOM_TYPE: [Element; 1] = [Element::Bk];
pub const CF_ATOM_TYPE: [Element; 1] = [Element::Cf];
pub const ES_ATOM_TYPE: [Element; 1] = [Element::Es];
pub const FM_ATOM_TYPE: [Element; 1] = [Element::Fm];
pub const MD_ATOM_TYPE: [Element; 1] = [Element::Md];
pub const NO_ATOM_TYPE: [Element; 1] = [Element::No];
pub const LR_ATOM_TYPE: [Element; 1] = [Element::Lr];





#[derive(Clone, Debug, PartialEq, Eq, Hash, Savefile)]
pub enum AminoAcid
{
    GLY, ALA, VAL, LEU, ILE,
    SER, THR, ASP, ASH, ASN, GLU, GLH, GLN, LYS, LYN, ARG,
    CYS, CYX, MET,
    HID, HIE, HIP, PHE, TYR, TRP,
    PRO,
}

pub const GLY_ATOM_TYPE: [Element; 7] = [Element::N, Element::H, Element::C, Element::H, Element::H, Element::C, Element::O,];
pub const ALA_ATOM_TYPE: [Element; 10] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::H, Element::C, Element::O,];
pub const VAL_ATOM_TYPE: [Element; 16] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::H, Element::C, Element::H, Element::H, Element::H, Element::C, Element::O,];
pub const LEU_ATOM_TYPE: [Element; 19] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::H, Element::C, Element::H, Element::H, Element::H, Element::C, Element::O,];
pub const ILE_ATOM_TYPE: [Element; 19] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::H, Element::C, Element::O,];
pub const SER_ATOM_TYPE: [Element; 11] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::O, Element::H, Element::C, Element::O,];
pub const THR_ATOM_TYPE: [Element; 14] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::H, Element::O, Element::H, Element::C, Element::O,];
pub const ASP_ATOM_TYPE: [Element; 12] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::O, Element::O, Element::C, Element::O,];
pub const ASH_ATOM_TYPE: [Element; 13] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::O, Element::O, Element::H, Element::C, Element::O,];
pub const ASN_ATOM_TYPE: [Element; 14] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::O, Element::N, Element::H, Element::H, Element::C, Element::O,];
pub const GLU_ATOM_TYPE: [Element; 15] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::O, Element::O, Element::C, Element::O,];
pub const GLH_ATOM_TYPE: [Element; 16] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::O, Element::O, Element::H, Element::C, Element::O,];
pub const GLN_ATOM_TYPE: [Element; 17] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::O, Element::N, Element::H, Element::H, Element::C, Element::O,];
pub const LYS_ATOM_TYPE: [Element; 22] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::N, Element::H, Element::H, Element::H, Element::C, Element::O,];
pub const LYN_ATOM_TYPE: [Element; 21] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::N, Element::H, Element::H, Element::C, Element::O,];
pub const ARG_ATOM_TYPE: [Element; 24] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::N, Element::H, Element::C, Element::N, Element::H, Element::H, Element::N, Element::H, Element::H, Element::C, Element::O,];
//pub const ARN_ATOM_TYPE: [Element; 23] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::N, Element::H, Element::C, Element::N, Element::H, Element::H, Element::N, Element::H, Element::C, Element::O,];
pub const CYS_ATOM_TYPE: [Element; 11] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::S, Element::H, Element::C, Element::O,];
pub const CYX_ATOM_TYPE: [Element; 10] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::S, Element::C, Element::O,];
//pub const CYM_ATOM_TYPE: [Element; 10] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::S, Element::C, Element::O,];
pub const MET_ATOM_TYPE: [Element; 17] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::S, Element::C, Element::H, Element::H, Element::H, Element::C, Element::O,];
pub const HID_ATOM_TYPE: [Element; 17] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::N, Element::H, Element::C, Element::H, Element::N, Element::C, Element::H, Element::C, Element::O,];
pub const HIE_ATOM_TYPE: [Element; 17] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::N, Element::C, Element::H, Element::N, Element::H, Element::C, Element::H, Element::C, Element::O,];
pub const HIP_ATOM_TYPE: [Element; 18] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::N, Element::H, Element::C, Element::H, Element::N, Element::H, Element::C, Element::H, Element::C, Element::O,];
pub const PHE_ATOM_TYPE: [Element; 20] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::C, Element::H, Element::C, Element::H, Element::C, Element::H, Element::C, Element::H, Element::C, Element::H, Element::C, Element::O,];
pub const TYR_ATOM_TYPE: [Element; 21] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::C, Element::H, Element::C, Element::H, Element::C, Element::O, Element::H, Element::C, Element::H, Element::C, Element::H, Element::C, Element::O,];
pub const TRP_ATOM_TYPE: [Element; 24] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::C, Element::H, Element::N, Element::H, Element::C, Element::C, Element::H, Element::C, Element::H, Element::C, Element::H, Element::C, Element::H, Element::C, Element::C, Element::O,];
pub const PRO_ATOM_TYPE: [Element; 14] = [Element::N, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::C, Element::O,];





#[derive(Clone, Debug, PartialEq, Savefile)]
pub enum Molecule
{
    WAT,
}

pub const WAT_ATOM_TYPE: [Element; 3] = [Element::O, Element::H, Element::H,];





#[derive(Clone, Debug, PartialEq, Savefile)]
pub enum Cap
{
    H, O,
    ACE, NME, NHE,
}

pub const ACE_ATOM_TYPE: [Element; 6] = [Element::H, Element::C, Element::H, Element::H, Element::C, Element::O,];
pub const NME_ATOM_TYPE: [Element; 6] = [Element::N, Element::H, Element::C, Element::H, Element::H, Element::H,];
pub const NHE_ATOM_TYPE: [Element; 3] = [Element::N, Element::H, Element::H,];










/// The entire protein system is divided into fragments (atoms, amino acid residues, and molecules).
///
/// # Fields
/// ```
/// Atom: atom
/// Residue: amino acid residue
/// Molecule: solvent or substrate
/// Head: head of a peptide
/// Tail: tail of a peptide
/// ```
#[derive(Clone, Debug, PartialEq, Savefile)]
pub enum FragmentType
{
    Atom (Element),
    Residue (AminoAcid),
    Molecule (Molecule),
    Head (Cap),
    Tail (Cap),
}

// 'STR_FRAGMENT_TYPE' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
static STR_TO_FRAGMENT_TYPE: phf::Map<&'static str, FragmentType> = phf_map!
{
    "H" => FragmentType::Atom(Element::H),
    "H+" => FragmentType::Atom(Element::H),
    "He" => FragmentType::Atom(Element::He),

    "Li" => FragmentType::Atom(Element::Li),
    "Be" => FragmentType::Atom(Element::Be),
    "B" => FragmentType::Atom(Element::B),
    "C" => FragmentType::Atom(Element::C),
    "N" => FragmentType::Atom(Element::N),
    "O" => FragmentType::Atom(Element::O),
    "F" => FragmentType::Atom(Element::F),
    "F-" => FragmentType::Atom(Element::F),
    "Ne" => FragmentType::Atom(Element::Ne),

    "Na" => FragmentType::Atom(Element::Na),
    "Na+" => FragmentType::Atom(Element::Na),
    "Mg" => FragmentType::Atom(Element::Mg),
    "Mg2+" => FragmentType::Atom(Element::Mg),
    "Al" => FragmentType::Atom(Element::Al),
    "Si" => FragmentType::Atom(Element::Si),
    "P" => FragmentType::Atom(Element::P),
    "S" => FragmentType::Atom(Element::S),
    "Cl" => FragmentType::Atom(Element::Cl),
    "Cl-" => FragmentType::Atom(Element::Cl),
    "Ar" => FragmentType::Atom(Element::Ar),

    "K" => FragmentType::Atom(Element::K),
    "K+" => FragmentType::Atom(Element::K),
    "Ca" => FragmentType::Atom(Element::Ca),
    "Ca2+" => FragmentType::Atom(Element::Ca),
    "Sc" => FragmentType::Atom(Element::Sc),
    "Ti" => FragmentType::Atom(Element::Ti),
    "V" => FragmentType::Atom(Element::V),
    "Cr" => FragmentType::Atom(Element::Cr),
    "Cr3+" => FragmentType::Atom(Element::Cr),
    "Mn" => FragmentType::Atom(Element::Mn),
    "Mn2+" => FragmentType::Atom(Element::Mn),
    "Fe" => FragmentType::Atom(Element::Fe),
    "Fe2+" => FragmentType::Atom(Element::Fe),
    "Fe3+" => FragmentType::Atom(Element::Fe),
    "Co" => FragmentType::Atom(Element::Co),
    "Co2+" => FragmentType::Atom(Element::Co),
    "Ni" => FragmentType::Atom(Element::Ni),
    "Ni2+" => FragmentType::Atom(Element::Ni),
    "Cu" => FragmentType::Atom(Element::Cu),
    "Cu2+" => FragmentType::Atom(Element::Cu),
    "Zn" => FragmentType::Atom(Element::Zn),
    "Zn2+" => FragmentType::Atom(Element::Zn),
    "Ga" => FragmentType::Atom(Element::Ga),
    "Ge" => FragmentType::Atom(Element::Ge),
    "As" => FragmentType::Atom(Element::As),
    "Se" => FragmentType::Atom(Element::Se),
    "Se2-" => FragmentType::Atom(Element::Se),
    "Se4+" => FragmentType::Atom(Element::Se),
    "Br" => FragmentType::Atom(Element::Br),
    "Br-" => FragmentType::Atom(Element::Br),
    "Kr" => FragmentType::Atom(Element::Kr),

    "Rb" => FragmentType::Atom(Element::Rb),
    "Sr" => FragmentType::Atom(Element::Sr),
    "Y" => FragmentType::Atom(Element::Y),
    "Zr" => FragmentType::Atom(Element::Zr),
    "Nb" => FragmentType::Atom(Element::Nb),
    "Mo" => FragmentType::Atom(Element::Mo),
    "Tc" => FragmentType::Atom(Element::Tc),
    "Ru" => FragmentType::Atom(Element::Ru),
    "Rh" => FragmentType::Atom(Element::Rh),
    "Pd" => FragmentType::Atom(Element::Pd),
    "Ag" => FragmentType::Atom(Element::Ag),
    "Cd" => FragmentType::Atom(Element::Cd),
    "In" => FragmentType::Atom(Element::In),
    "Sn" => FragmentType::Atom(Element::Sn),
    "Sn2+" => FragmentType::Atom(Element::Sn),
    "Sb" => FragmentType::Atom(Element::Sb),
    "Te" => FragmentType::Atom(Element::Te),
    "I" => FragmentType::Atom(Element::I),
    "I-" => FragmentType::Atom(Element::I),
    "Xe" => FragmentType::Atom(Element::Xe),

    "Cs" => FragmentType::Atom(Element::Cs),
    "Ba" => FragmentType::Atom(Element::Ba),
    "Hf" => FragmentType::Atom(Element::Hf),
    "Ta" => FragmentType::Atom(Element::Ta),
    "W" => FragmentType::Atom(Element::W),
    "Re" => FragmentType::Atom(Element::Re),
    "Os" => FragmentType::Atom(Element::Os),
    "Ir" => FragmentType::Atom(Element::Ir),
    "Pt" => FragmentType::Atom(Element::Pt),
    "Au" => FragmentType::Atom(Element::Au),
    "Hg" => FragmentType::Atom(Element::Hg),
    "Tl" => FragmentType::Atom(Element::Tl),
    "Pb" => FragmentType::Atom(Element::Pb),
    "Bi" => FragmentType::Atom(Element::Bi),
    "Po" => FragmentType::Atom(Element::Po),
    "At" => FragmentType::Atom(Element::At),
    "Rn" => FragmentType::Atom(Element::Rn),

    "Fr" => FragmentType::Atom(Element::Fr),
    "Ra" => FragmentType::Atom(Element::Ra),
    "Rf" => FragmentType::Atom(Element::Rf),
    "Db" => FragmentType::Atom(Element::Db),
    "Sg" => FragmentType::Atom(Element::Sg),
    "Bh" => FragmentType::Atom(Element::Bh),
    "Hs" => FragmentType::Atom(Element::Hs),
    "Mt" => FragmentType::Atom(Element::Mt),
    "Ds" => FragmentType::Atom(Element::Ds),
    "Rg" => FragmentType::Atom(Element::Rg),

    "La" => FragmentType::Atom(Element::La),
    "Ce" => FragmentType::Atom(Element::Ce),
    "Pr" => FragmentType::Atom(Element::Pr),
    "Nd" => FragmentType::Atom(Element::Nd),
    "Pm" => FragmentType::Atom(Element::Pm),
    "Sm" => FragmentType::Atom(Element::Sm),
    "Eu" => FragmentType::Atom(Element::Eu),
    "Gd" => FragmentType::Atom(Element::Gd),
    "Tb" => FragmentType::Atom(Element::Tb),
    "Dy" => FragmentType::Atom(Element::Dy),
    "Ho" => FragmentType::Atom(Element::Ho),
    "Er" => FragmentType::Atom(Element::Er),
    "Tm" => FragmentType::Atom(Element::Tm),
    "Yb" => FragmentType::Atom(Element::Yb),
    "Lu" => FragmentType::Atom(Element::Lu),

    "Ac" => FragmentType::Atom(Element::Ac),
    "Th" => FragmentType::Atom(Element::Th),
    "Pa" => FragmentType::Atom(Element::Pa),
    "U" => FragmentType::Atom(Element::U),
    "Np" => FragmentType::Atom(Element::Np),
    "Pu" => FragmentType::Atom(Element::Pu),
    "Am" => FragmentType::Atom(Element::Am),
    "Cm" => FragmentType::Atom(Element::Cm),
    "Bk" => FragmentType::Atom(Element::Bk),
    "Cf" => FragmentType::Atom(Element::Cf),
    "Es" => FragmentType::Atom(Element::Es),
    "Fm" => FragmentType::Atom(Element::Fm),
    "Md" => FragmentType::Atom(Element::Md),
    "No" => FragmentType::Atom(Element::No),
    "Lr" => FragmentType::Atom(Element::Lr),

    "GLY" => FragmentType::Residue(AminoAcid::GLY),
    "ALA" => FragmentType::Residue(AminoAcid::ALA),
    "VAL" => FragmentType::Residue(AminoAcid::VAL),
    "LEU" => FragmentType::Residue(AminoAcid::LEU),
    "ILE" => FragmentType::Residue(AminoAcid::ILE),
    "SER" => FragmentType::Residue(AminoAcid::SER),
    "THR" => FragmentType::Residue(AminoAcid::THR),
    "ASP" => FragmentType::Residue(AminoAcid::ASP),
    "ASH" => FragmentType::Residue(AminoAcid::ASH),
    "ASN" => FragmentType::Residue(AminoAcid::ASN),
    "GLU" => FragmentType::Residue(AminoAcid::GLU),
    "GLH" => FragmentType::Residue(AminoAcid::GLH),
    "GLN" => FragmentType::Residue(AminoAcid::GLN),
    "LYS" => FragmentType::Residue(AminoAcid::LYS),
    "LYN" => FragmentType::Residue(AminoAcid::LYN),
    "ARG" => FragmentType::Residue(AminoAcid::ARG),
//    "ARN" => FragmentType::Residue(AminoAcid::ARN),
    "CYS" => FragmentType::Residue(AminoAcid::CYS),
    "CYX" => FragmentType::Residue(AminoAcid::CYX),
//    "CYM" => FragmentType::Residue(AminoAcid::CYM),
    "MET" => FragmentType::Residue(AminoAcid::MET),
    "HID" => FragmentType::Residue(AminoAcid::HID),
    "HIE" => FragmentType::Residue(AminoAcid::HIE),
    "HIP" => FragmentType::Residue(AminoAcid::HIP),
    "PHE" => FragmentType::Residue(AminoAcid::PHE),
    "TYR" => FragmentType::Residue(AminoAcid::TYR),
    "TRP" => FragmentType::Residue(AminoAcid::TRP),
    "PRO" => FragmentType::Residue(AminoAcid::PRO),

    "WAT" => FragmentType::Molecule(Molecule::WAT),

    "ACE" => FragmentType::Head(Cap::ACE),
    "NME" => FragmentType::Tail(Cap::NME),
    "NHE" => FragmentType::Tail(Cap::NHE),
};

// 'STR_NATOM' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
static STR_TO_NATOM: phf::Map<&'static str, usize> = phf_map!
{
    "H" => 1,
    "H+" => 1,
    "He" => 1,

    "Li" => 1,
    "Be" => 1,
    "B" => 1,
    "C" => 1,
    "N" => 1,
    "O" => 1,
    "F" => 1,
    "F-" => 1,
    "Ne" => 1,

    "Na" => 1,
    "Na+" => 1,
    "Mg" => 1,
    "Mg2+" => 1,
    "Al" => 1,
    "Si" => 1,
    "P" => 1,
    "S" => 1,
    "Cl" => 1,
    "Cl-" => 1,
    "Ar" => 1,

    "K" => 1,
    "K+" => 1,
    "Ca" => 1,
    "Ca2+" => 1,
    "Sc" => 1,
    "Ti" => 1,
    "V" => 1,
    "Cr" => 1,
    "Cr3+" => 1,
    "Mn" => 1,
    "Mn2+" => 1,
    "Fe" => 1,
    "Fe2+" => 1,
    "Fe3+" => 1,
    "Co" => 1,
    "Co2+" => 1,
    "Ni" => 1,
    "Ni2+" => 1,
    "Cu" => 1,
    "Cu2+" => 1,
    "Zn" => 1,
    "Zn2+" => 1,
    "Ga" => 1,
    "Ge" => 1,
    "As" => 1,
    "Se" => 1,
    "Se2-" => 1,
    "Se4+" => 1,
    "Br" => 1,
    "Br-" => 1,
    "Kr" => 1,

    "Rb" => 1,
    "Sr" => 1,
    "Y" => 1,
    "Zr" => 1,
    "Nb" => 1,
    "Mo" => 1,
    "Tc" => 1,
    "Ru" => 1,
    "Rh" => 1,
    "Pd" => 1,
    "Ag" => 1,
    "Cd" => 1,
    "In" => 1,
    "Sn" => 1,
    "Sn2+" => 1,
    "Sb" => 1,
    "Te" => 1,
    "I" => 1,
    "I-" => 1,
    "Xe" => 1,

    "Cs" => 1,
    "Ba" => 1,
    "Hf" => 1,
    "Ta" => 1,
    "W" => 1,
    "Re" => 1,
    "Os" => 1,
    "Ir" => 1,
    "Pt" => 1,
    "Au" => 1,
    "Hg" => 1,
    "Tl" => 1,
    "Pb" => 1,
    "Bi" => 1,
    "Po" => 1,
    "At" => 1,
    "Rn" => 1,

    "Fr" => 1,
    "Ra" => 1,
    "Rf" => 1,
    "Db" => 1,
    "Sg" => 1,
    "Bh" => 1,
    "Hs" => 1,
    "Mt" => 1,
    "Ds" => 1,
    "Rg" => 1,

    "La" => 1,
    "Ce" => 1,
    "Pr" => 1,
    "Nd" => 1,
    "Pm" => 1,
    "Sm" => 1,
    "Eu" => 1,
    "Gd" => 1,
    "Tb" => 1,
    "Dy" => 1,
    "Ho" => 1,
    "Er" => 1,
    "Tm" => 1,
    "Yb" => 1,
    "Lu" => 1,

    "Ac" => 1,
    "Th" => 1,
    "Pa" => 1,
    "U" => 1,
    "Np" => 1,
    "Pu" => 1,
    "Am" => 1,
    "Cm" => 1,
    "Bk" => 1,
    "Cf" => 1,
    "Es" => 1,
    "Fm" => 1,
    "Md" => 1,
    "No" => 1,
    "Lr" => 1,

    "GLY" => 7,
    "ALA" => 10,
    "VAL" => 16,
    "LEU" => 19,
    "ILE" => 19,
    "SER" => 11,
    "THR" => 14,
    "ASP" => 12,
    "ASH" => 13,
    "ASN" => 14,
    "GLU" => 15,
    "GLH" => 16,
    "GLN" => 17,
    "LYS" => 22,
    "LYN" => 21,
    "ARG" => 24,
//    "ARN" => 23,
    "CYS" => 11,
    "CYX" => 10,
//    "CYM" => 10,
    "MET" => 17,
    "HID" => 17,
    "HIE" => 17,
    "HIP" => 18,
    "PHE" => 20,
    "TYR" => 21,
    "TRP" => 24,
    "PRO" => 14,

    "WAT" => 3,

    "ACE" => 6,
    "NME" => 6,
    "NHE" => 3,
};

// 'STR_ATOM_TYPE' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
static STR_TO_ATOM_TYPE: phf::Map<&'static str, &[Element]> = phf_map!
{
    "H" => &H_ATOM_TYPE,
    "H+" => &H_ATOM_TYPE,
    "He" => &HE_ATOM_TYPE,

    "Li" => &LI_ATOM_TYPE,
    "Be" => &BE_ATOM_TYPE,
    "B" => &B_ATOM_TYPE,
    "C" => &C_ATOM_TYPE,
    "N" => &N_ATOM_TYPE,
    "O" => &O_ATOM_TYPE,
    "F" => &F_ATOM_TYPE,
    "F-" => &F_ATOM_TYPE,
    "Ne" => &NE_ATOM_TYPE,

    "Na" => &NA_ATOM_TYPE,
    "Na+" => &NA_ATOM_TYPE,
    "Mg" => &MG_ATOM_TYPE,
    "Mg2+" => &MG_ATOM_TYPE,
    "Al" => &AL_ATOM_TYPE,
    "Si" => &SI_ATOM_TYPE,
    "P" => &P_ATOM_TYPE,
    "S" => &S_ATOM_TYPE,
    "Cl" => &CL_ATOM_TYPE,
    "Cl-" => &CL_ATOM_TYPE,
    "Ar" => &AR_ATOM_TYPE,

    "K" => &K_ATOM_TYPE,
    "K+" => &K_ATOM_TYPE,
    "Ca" => &CA_ATOM_TYPE,
    "Ca2+" => &CA_ATOM_TYPE,
    "Sc" => &SC_ATOM_TYPE,
    "Ti" => &TI_ATOM_TYPE,
    "V" => &V_ATOM_TYPE,
    "Cr" => &CR_ATOM_TYPE,
    "Cr3+" => &CR_ATOM_TYPE,
    "Mn" => &MN_ATOM_TYPE,
    "Mn2+" => &MN_ATOM_TYPE,
    "Fe" => &FE_ATOM_TYPE,
    "Fe2+" => &FE_ATOM_TYPE,
    "Fe3+" => &FE_ATOM_TYPE,
    "Co" => &CO_ATOM_TYPE,
    "Co2+" => &CO_ATOM_TYPE,
    "Ni" => &NI_ATOM_TYPE,
    "Ni2+" => &NI_ATOM_TYPE,
    "Cu" => &CU_ATOM_TYPE,
    "Cu2+" => &CU_ATOM_TYPE,
    "Zn" => &ZN_ATOM_TYPE,
    "Zn2+" => &ZN_ATOM_TYPE,
    "Ga" => &GA_ATOM_TYPE,
    "Ge" => &GE_ATOM_TYPE,
    "As" => &AS_ATOM_TYPE,
    "Se" => &SE_ATOM_TYPE,
    "Se2-" => &SE_ATOM_TYPE,
    "Se4+" => &SE_ATOM_TYPE,
    "Br" => &BR_ATOM_TYPE,
    "Br-" => &BR_ATOM_TYPE,
    "Kr" => &KR_ATOM_TYPE,

    "Rb" => &RB_ATOM_TYPE,
    "Sr" => &SR_ATOM_TYPE,
    "Y" => &Y_ATOM_TYPE,
    "Zr" => &ZR_ATOM_TYPE,
    "Nb" => &NB_ATOM_TYPE,
    "Mo" => &MO_ATOM_TYPE,
    "Tc" => &TC_ATOM_TYPE,
    "Ru" => &RU_ATOM_TYPE,
    "Rh" => &RH_ATOM_TYPE,
    "Pd" => &PD_ATOM_TYPE,
    "Ag" => &AG_ATOM_TYPE,
    "Cd" => &CD_ATOM_TYPE,
    "In" => &IN_ATOM_TYPE,
    "Sn" => &SN_ATOM_TYPE,
    "Sn2+" => &SN_ATOM_TYPE,
    "Sb" => &SB_ATOM_TYPE,
    "Te" => &TE_ATOM_TYPE,
    "I" => &I_ATOM_TYPE,
    "I-" => &I_ATOM_TYPE,
    "Xe" => &XE_ATOM_TYPE,

    "Cs" => &CS_ATOM_TYPE,
    "Ba" => &BA_ATOM_TYPE,
    "Hf" => &HF_ATOM_TYPE,
    "Ta" => &TA_ATOM_TYPE,
    "W" => &W_ATOM_TYPE,
    "Re" => &RE_ATOM_TYPE,
    "Os" => &OS_ATOM_TYPE,
    "Ir" => &IR_ATOM_TYPE,
    "Pt" => &PT_ATOM_TYPE,
    "Au" => &AU_ATOM_TYPE,
    "Hg" => &HG_ATOM_TYPE,
    "Tl" => &TL_ATOM_TYPE,
    "Pb" => &PB_ATOM_TYPE,
    "Bi" => &BI_ATOM_TYPE,
    "Po" => &PO_ATOM_TYPE,
    "At" => &AT_ATOM_TYPE,
    "Rn" => &RN_ATOM_TYPE,

    "Fr" => &FR_ATOM_TYPE,
    "Ra" => &RA_ATOM_TYPE,
    "Rf" => &RF_ATOM_TYPE,
    "Db" => &DB_ATOM_TYPE,
    "Sg" => &SG_ATOM_TYPE,
    "Bh" => &BH_ATOM_TYPE,
    "Hs" => &HS_ATOM_TYPE,
    "Mt" => &MT_ATOM_TYPE,
    "Ds" => &DS_ATOM_TYPE,
    "Rg" => &RG_ATOM_TYPE,

    "La" => &LA_ATOM_TYPE,
    "Ce" => &CE_ATOM_TYPE,
    "Pr" => &PR_ATOM_TYPE,
    "Nd" => &ND_ATOM_TYPE,
    "Pm" => &PM_ATOM_TYPE,
    "Sm" => &SM_ATOM_TYPE,
    "Eu" => &EU_ATOM_TYPE,
    "Gd" => &GD_ATOM_TYPE,
    "Tb" => &TB_ATOM_TYPE,
    "Dy" => &DY_ATOM_TYPE,
    "Ho" => &HO_ATOM_TYPE,
    "Er" => &ER_ATOM_TYPE,
    "Tm" => &TM_ATOM_TYPE,
    "Yb" => &YB_ATOM_TYPE,
    "Lu" => &LU_ATOM_TYPE,

    "Ac" => &AC_ATOM_TYPE,
    "Th" => &TH_ATOM_TYPE,
    "Pa" => &PA_ATOM_TYPE,
    "U" => &U_ATOM_TYPE,
    "Np" => &NP_ATOM_TYPE,
    "Pu" => &PU_ATOM_TYPE,
    "Am" => &AM_ATOM_TYPE,
    "Cm" => &CM_ATOM_TYPE,
    "Bk" => &BK_ATOM_TYPE,
    "Cf" => &CF_ATOM_TYPE,
    "Es" => &ES_ATOM_TYPE,
    "Fm" => &FM_ATOM_TYPE,
    "Md" => &MD_ATOM_TYPE,
    "No" => &NO_ATOM_TYPE,
    "Lr" => &LR_ATOM_TYPE,

    "GLY" => &GLY_ATOM_TYPE,
    "ALA" => &ALA_ATOM_TYPE,
    "VAL" => &VAL_ATOM_TYPE,
    "LEU" => &LEU_ATOM_TYPE,
    "ILE" => &ILE_ATOM_TYPE,
    "SER" => &SER_ATOM_TYPE,
    "THR" => &THR_ATOM_TYPE,
    "ASP" => &ASP_ATOM_TYPE,
    "ASH" => &ASH_ATOM_TYPE,
    "ASN" => &ASN_ATOM_TYPE,
    "GLU" => &GLU_ATOM_TYPE,
    "GLH" => &GLH_ATOM_TYPE,
    "GLN" => &GLN_ATOM_TYPE,
    "LYS" => &LYS_ATOM_TYPE,
    "LYN" => &LYN_ATOM_TYPE,
    "ARG" => &ARG_ATOM_TYPE,
//    "ARN" => &ARN_ATOM_TYPE,
    "CYS" => &CYS_ATOM_TYPE,
    "CYX" => &CYX_ATOM_TYPE,
//    "CYM" => &CYM_ATOM_TYPE,
    "MET" => &MET_ATOM_TYPE,
    "HID" => &HID_ATOM_TYPE,
    "HIE" => &HIE_ATOM_TYPE,
    "HIP" => &HIP_ATOM_TYPE,
    "PHE" => &PHE_ATOM_TYPE,
    "TYR" => &TYR_ATOM_TYPE,
    "TRP" => &TRP_ATOM_TYPE,
    "PRO" => &PRO_ATOM_TYPE,

    "WAT" => &WAT_ATOM_TYPE,

    "ACE" => &ACE_ATOM_TYPE,
    "NME" => &NME_ATOM_TYPE,
    "NHE" => &NHE_ATOM_TYPE,
};

impl FragmentType
{
    pub fn from_str(fragment_type: &str) -> Self
    {
        STR_TO_FRAGMENT_TYPE.get(fragment_type).cloned().expect(&error_type("fragment", fragment_type))
    }

    pub fn get_natom(fragment_type: &str) -> usize
    {
        STR_TO_NATOM.get(fragment_type).cloned().expect(&error_type("fragment", fragment_type))
    }

    pub fn get_atom_type(fragment_type: &str) -> &'static [Element]
    {
        STR_TO_ATOM_TYPE.get(fragment_type).cloned().expect(&error_type("fragment", fragment_type))
    }
}










