//! Contains mathematical, physical, and chemical constants.
use crate::common::error::*;
use phf::phf_map;
use mpi::topology::Rank;
use savefile_derive::Savefile;










pub const ROOT_RANK: Rank = 0;
#[cfg(not(feature = "cuda"))]
pub type Device = dfdx::tensor::Cpu;
#[cfg(feature = "cuda")]
pub type Device = dfdx::tensor::Cuda;










// Mathematical
pub const PI: f64 = 3.141592653589793;
pub const GOLDEN_RATIO1: f64 = 0.618033988749895;
pub const GOLDEN_RATIO2: f64 = 1.618033988749895;










// Physical

// Unit Conversion
pub const C_LIGHT: f64 = 299792458.0;                // (m/s)
pub const H_PLANCK: f64 = 6.62606896E-34;                // (J*s)
pub const BOLTZMANN: f64 = 1.3806504E-23;                // (J/K)
pub const RYBDERG: f64 = 10973731.568527;
pub const N_AVOGADRO: f64 = 6.02214076E+23;

pub const BOHR_TO_ANGSTROM: f64 = 0.52917720859;
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_TO_ANGSTROM;

pub const HARTREE_TO_JOULE: f64 = 2.0 * RYBDERG * H_PLANCK * C_LIGHT;
pub const JOULE_TO_HARTREE: f64 = 1.0 / HARTREE_TO_JOULE;
pub const HARTREE_TO_KJMOL: f64 = 0.001 * HARTREE_TO_JOULE * N_AVOGADRO;
pub const KJMOL_TO_HARTREE: f64 = 1.0 / HARTREE_TO_KJMOL;
pub const HARTREE_TO_KCALMOL: f64 = HARTREE_TO_KJMOL / 4.184;
pub const KCALMOL_TO_HARTREE: f64 = 1.0 / HARTREE_TO_KCALMOL;


pub const AU_TO_FEMTOSECOND: f64 = 1.0E15 / (4.0 * PI * RYBDERG * C_LIGHT);
pub const FEMTOSECOND_TO_AU: f64 = 1.0 / AU_TO_FEMTOSECOND;










// Chemical

#[derive(Clone, Debug, PartialEq, Savefile)]
pub enum Element
{
    H, He,
    Li, Be, B, C, N, O, F, Ne,
    Na, Mg, Al, Si, P, S, Cl, Ar,
    K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr,
    Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe,
    Cs, Ba, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn,
    Fr, Ra, Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg,
    La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu,
    Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr,
}

// 'STR_TO_ELEMENT' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
static STR_TO_ELEMENT: phf::Map<&'static str, Element> = phf_map!
{
    "H" => Element::H,
    "He" => Element::He,

    "Li" => Element::Li,
    "Be" => Element::Be,
    "B" => Element::B,
    "C" => Element::C,
    "N" => Element::N,
    "O" => Element::O,
    "F" => Element::F,
    "Ne" => Element::Ne,

    "Na" => Element::Na,
    "Mg" => Element::Mg,
    "Al" => Element::Al,
    "Si" => Element::Si,
    "P" => Element::P,
    "S" => Element::S,
    "Cl" => Element::Cl,
    "Ar" => Element::Ar,

    "K" => Element::K,
    "Ca" => Element::Ca,
    "Sc" => Element::Sc,
    "Ti" => Element::Ti,
    "V" => Element::V,
    "Cr" => Element::Cr,
    "Mn" => Element::Mn,
    "Fe" => Element::Fe,
    "Co" => Element::Co,
    "Ni" => Element::Ni,
    "Cu" => Element::Cu,
    "Zn" => Element::Zn,
    "Ga" => Element::Ga,
    "Ge" => Element::Ge,
    "As" => Element::As,
    "Se" => Element::Se,
    "Br" => Element::Br,
    "Kr" => Element::Kr,

    "Rb" => Element::Rb,
    "Sr" => Element::Sr,
    "Y" => Element::Y,
    "Zr" => Element::Zr,
    "Nb" => Element::Nb,
    "Mo" => Element::Mo,
    "Tc" => Element::Tc,
    "Ru" => Element::Ru,
    "Rh" => Element::Rh,
    "Pd" => Element::Pd,
    "Ag" => Element::Ag,
    "Cd" => Element::Cd,
    "In" => Element::In,
    "Sn" => Element::Sn,
    "Sb" => Element::Sb,
    "Te" => Element::Te,
    "I" => Element::I,
    "Xe" => Element::Xe,

    "Cs" => Element::Cs,
    "Ba" => Element::Ba,
    "Hf" => Element::Hf,
    "Ta" => Element::Ta,
    "W" => Element::W,
    "Re" => Element::Re,
    "Os" => Element::Os,
    "Ir" => Element::Ir,
    "Pt" => Element::Pt,
    "Au" => Element::Au,
    "Hg" => Element::Hg,
    "Tl" => Element::Tl,
    "Pb" => Element::Pb,
    "Bi" => Element::Bi,
    "Po" => Element::Po,
    "At" => Element::At,
    "Rn" => Element::Rn,

    "Fr" => Element::Fr,
    "Ra" => Element::Ra,
    "Rf" => Element::Rf,
    "Db" => Element::Db,
    "Sg" => Element::Sg,
    "Bh" => Element::Bh,
    "Hs" => Element::Hs,
    "Mt" => Element::Mt,
    "Ds" => Element::Ds,
    "Rg" => Element::Rg,

    "La" => Element::La,
    "Ce" => Element::Ce,
    "Pr" => Element::Pr,
    "Nd" => Element::Nd,
    "Pm" => Element::Pm,
    "Sm" => Element::Sm,
    "Eu" => Element::Eu,
    "Gd" => Element::Gd,
    "Tb" => Element::Tb,
    "Dy" => Element::Dy,
    "Ho" => Element::Ho,
    "Er" => Element::Er,
    "Tm" => Element::Tm,
    "Yb" => Element::Yb,
    "Lu" => Element::Lu,

    "Ac" => Element::Ac,
    "Th" => Element::Th,
    "Pa" => Element::Pa,
    "U" => Element::U,
    "Np" => Element::Np,
    "Pu" => Element::Pu,
    "Am" => Element::Am,
    "Cm" => Element::Cm,
    "Bk" => Element::Bk,
    "Cf" => Element::Cf,
    "Es" => Element::Es,
    "Fm" => Element::Fm,
    "Md" => Element::Md,
    "No" => Element::No,
    "Lr" => Element::Lr,

    "HE" => Element::He,
    "LI" => Element::Li,
    "BE" => Element::Be,
    "NE" => Element::Ne,
    "NA" => Element::Na,
    "MG" => Element::Mg,
    "AL" => Element::Al,
    "SI" => Element::Si,
    "CL" => Element::Cl,
    "AR" => Element::Ar,
    "CA" => Element::Ca,
    "SC" => Element::Sc,
    "TI" => Element::Ti,
    "CR" => Element::Cr,
    "MN" => Element::Mn,
    "FE" => Element::Fe,
    "CO" => Element::Co,
    "NI" => Element::Ni,
    "CU" => Element::Cu,
    "ZN" => Element::Zn,
    "GA" => Element::Ga,
    "GE" => Element::Ge,
    "AS" => Element::As,
    "SE" => Element::Se,
    "BR" => Element::Br,
    "KR" => Element::Kr,
    "RB" => Element::Rb,
    "SR" => Element::Sr,
    "ZR" => Element::Zr,
    "NB" => Element::Nb,
    "MO" => Element::Mo,
    "TC" => Element::Tc,
    "RU" => Element::Ru,
    "RH" => Element::Rh,
    "PD" => Element::Pd,
    "AG" => Element::Ag,
    "CD" => Element::Cd,
    "IN" => Element::In,
    "SN" => Element::Sn,
    "SB" => Element::Sb,
    "TE" => Element::Te,
    "XE" => Element::Xe,
    "CS" => Element::Cs,
    "BA" => Element::Ba,
    "HF" => Element::Hf,
    "TA" => Element::Ta,
    "RE" => Element::Re,
    "OS" => Element::Os,
    "IR" => Element::Ir,
    "PT" => Element::Pt,
    "AU" => Element::Au,
    "HG" => Element::Hg,
    "TL" => Element::Tl,
    "PB" => Element::Pb,
    "BI" => Element::Bi,
    "PO" => Element::Po,
    "AT" => Element::At,
    "RN" => Element::Rn,
    "FR" => Element::Fr,
    "RA" => Element::Ra,
    "RF" => Element::Rf,
    "DB" => Element::Db,
    "SG" => Element::Sg,
    "BH" => Element::Bh,
    "HS" => Element::Hs,
    "MT" => Element::Mt,
    "DS" => Element::Ds,
    "RG" => Element::Rg,
    "LA" => Element::La,
    "CE" => Element::Ce,
    "PR" => Element::Pr,
    "ND" => Element::Nd,
    "PM" => Element::Pm,
    "SM" => Element::Sm,
    "EU" => Element::Eu,
    "GD" => Element::Gd,
    "TB" => Element::Tb,
    "DY" => Element::Dy,
    "HO" => Element::Ho,
    "ER" => Element::Er,
    "TM" => Element::Tm,
    "YB" => Element::Yb,
    "LU" => Element::Lu,
    "AC" => Element::Ac,
    "TH" => Element::Th,
    "PA" => Element::Pa,
    "NP" => Element::Np,
    "PU" => Element::Pu,
    "AM" => Element::Am,
    "CM" => Element::Cm,
    "BK" => Element::Bk,
    "CF" => Element::Cf,
    "ES" => Element::Es,
    "FM" => Element::Fm,
    "MD" => Element::Md,
    "NO" => Element::No,
    "LR" => Element::Lr,
};

// 'STR_TO_ATOMIC_NUMBER' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
static STR_TO_ATOMIC_NUMBER: phf::Map<&'static str, usize> = phf_map!
{
    "H" => 1,
    "He" => 2,

    "Li" => 3,
    "Be" => 4,
    "B" => 5,
    "C" => 6,
    "N" => 7,
    "O" => 8,
    "F" => 9,
    "Ne" => 10,

    "Na" => 11,
    "Mg" => 12,
    "Al" => 13,
    "Si" => 14,
    "P" => 15,
    "S" => 16,
    "Cl" => 17,
    "Ar" => 18,

    "K" => 19,
    "Ca" => 20,
    "Sc" => 21,
    "Ti" => 22,
    "V" => 23,
    "Cr" => 24,
    "Mn" => 25,
    "Fe" => 26,
    "Co" => 27,
    "Ni" => 28,
    "Cu" => 29,
    "Zn" => 30,
    "Ga" => 31,
    "Ge" => 32,
    "As" => 33,
    "Se" => 34,
    "Br" => 35,
    "Kr" => 36,

    "Rb" => 37,
    "Sr" => 38,
    "Y" => 39,
    "Zr" => 40,
    "Nb" => 41,
    "Mo" => 42,
    "Tc" => 43,
    "Ru" => 44,
    "Rh" => 45,
    "Pd" => 46,
    "Ag" => 47,
    "Cd" => 48,
    "In" => 49,
    "Sn" => 50,
    "Sb" => 51,
    "Te" => 52,
    "I" => 53,
    "Xe" => 54,

    "Cs" => 55,
    "Ba" => 56,
    "Hf" => 72,
    "Ta" => 73,
    "W" => 74,
    "Re" => 75,
    "Os" => 76,
    "Ir" => 77,
    "Pt" => 78,
    "Au" => 79,
    "Hg" => 80,
    "Tl" => 81,
    "Pb" => 82,
    "Bi" => 83,
    "Po" => 84,
    "At" => 85,
    "Rn" => 86,

    "Fr" => 87,
    "Ra" => 88,
    "Rf" => 104,
    "Db" => 105,
    "Sg" => 106,
    "Bh" => 107,
    "Hs" => 108,
    "Mt" => 109,
    "Ds" => 110,
    "Rg" => 111,

    "La" => 57,
    "Ce" => 58,
    "Pr" => 59,
    "Nd" => 60,
    "Pm" => 61,
    "Sm" => 62,
    "Eu" => 63,
    "Gd" => 64,
    "Tb" => 65,
    "Dy" => 66,
    "Ho" => 67,
    "Er" => 68,
    "Tm" => 69,
    "Yb" => 70,
    "Lu" => 71,

    "Ac" => 89,
    "Th" => 90,
    "Pa" => 91,
    "U" => 92,
    "Np" => 93,
    "Pu" => 94,
    "Am" => 95,
    "Cm" => 96,
    "Bk" => 97,
    "Cf" => 98,
    "Es" => 99,
    "Fm" => 100,
    "Md" => 101,
    "No" => 102,
    "Lr" => 103,
};

// Refer to Paper "Covalent radii revisited" (Dalton Trans., 2008, 2832-2838)
// Unit: bohr
static STR_TO_ATOMIC_RADIUS: phf::Map<&'static str, f64> = phf_map!
{
    "H" => 0.31 * ANGSTROM_TO_BOHR,
    "He" => 0.28 * ANGSTROM_TO_BOHR,

    "Li" => 1.28 * ANGSTROM_TO_BOHR,
    "Be" => 0.96 * ANGSTROM_TO_BOHR,
    "B" => 0.84 * ANGSTROM_TO_BOHR,
    "C" => 0.76 * ANGSTROM_TO_BOHR,
    "N" => 0.71 * ANGSTROM_TO_BOHR,
    "O" => 0.66 * ANGSTROM_TO_BOHR,
    "F" => 0.57 * ANGSTROM_TO_BOHR,
    "Ne" => 0.58 * ANGSTROM_TO_BOHR,

    "Na" => 1.66 * ANGSTROM_TO_BOHR,
    "Mg" => 1.41 * ANGSTROM_TO_BOHR,
    "Al" => 1.21 * ANGSTROM_TO_BOHR,
    "Si" => 1.11 * ANGSTROM_TO_BOHR,
    "P" => 1.07 * ANGSTROM_TO_BOHR,
    "S" => 1.05 * ANGSTROM_TO_BOHR,
    "Cl" => 1.02 * ANGSTROM_TO_BOHR,
    "Ar" => 1.06 * ANGSTROM_TO_BOHR,

    "K" => 2.03 * ANGSTROM_TO_BOHR,
    "Ca" => 1.76 * ANGSTROM_TO_BOHR,
    "Sc" => 1.70 * ANGSTROM_TO_BOHR,
    "Ti" => 1.60 * ANGSTROM_TO_BOHR,
    "V" => 1.53 * ANGSTROM_TO_BOHR,
    "Cr" => 1.39 * ANGSTROM_TO_BOHR,
    "Mn" => 1.61 * ANGSTROM_TO_BOHR,
    "Fe" => 1.52 * ANGSTROM_TO_BOHR,
    "Co" => 1.50 * ANGSTROM_TO_BOHR,
    "Ni" => 1.24 * ANGSTROM_TO_BOHR,
    "Cu" => 1.32 * ANGSTROM_TO_BOHR,
    "Zn" => 1.22 * ANGSTROM_TO_BOHR,
    "Ga" => 1.22 * ANGSTROM_TO_BOHR,
    "Ge" => 1.20 * ANGSTROM_TO_BOHR,
    "As" => 1.19 * ANGSTROM_TO_BOHR,
    "Se" => 1.20 * ANGSTROM_TO_BOHR,
    "Br" => 1.20 * ANGSTROM_TO_BOHR,
    "Kr" => 1.16 * ANGSTROM_TO_BOHR,

    "Rb" => 2.20 * ANGSTROM_TO_BOHR,
    "Sr" => 1.95 * ANGSTROM_TO_BOHR,
    "Y" => 1.90 * ANGSTROM_TO_BOHR,
    "Zr" => 1.75 * ANGSTROM_TO_BOHR,
    "Nb" => 1.64 * ANGSTROM_TO_BOHR,
    "Mo" => 1.54 * ANGSTROM_TO_BOHR,
    "Tc" => 1.47 * ANGSTROM_TO_BOHR,
    "Ru" => 1.46 * ANGSTROM_TO_BOHR,
    "Rh" => 1.42 * ANGSTROM_TO_BOHR,
    "Pd" => 1.39 * ANGSTROM_TO_BOHR,
    "Ag" => 1.45 * ANGSTROM_TO_BOHR,
    "Cd" => 1.44 * ANGSTROM_TO_BOHR,
    "In" => 1.42 * ANGSTROM_TO_BOHR,
    "Sn" => 1.39 * ANGSTROM_TO_BOHR,
    "Sb" => 1.39 * ANGSTROM_TO_BOHR,
    "Te" => 1.38 * ANGSTROM_TO_BOHR,
    "I" => 1.39 * ANGSTROM_TO_BOHR,
    "Xe" => 1.40 * ANGSTROM_TO_BOHR,

    "Cs" => 2.44 * ANGSTROM_TO_BOHR,
    "Ba" => 2.15 * ANGSTROM_TO_BOHR,
    "La" => 2.07 * ANGSTROM_TO_BOHR,
    "Ce" => 2.04 * ANGSTROM_TO_BOHR,
    "Pr" => 2.03 * ANGSTROM_TO_BOHR,
    "Nd" => 2.01 * ANGSTROM_TO_BOHR,
    "Pm" => 1.99 * ANGSTROM_TO_BOHR,
    "Sm" => 1.98 * ANGSTROM_TO_BOHR,
    "Eu" => 1.98 * ANGSTROM_TO_BOHR,
    "Gd" => 1.96 * ANGSTROM_TO_BOHR,
    "Tb" => 1.94 * ANGSTROM_TO_BOHR,
    "Dy" => 1.92 * ANGSTROM_TO_BOHR,
    "Ho" => 1.92 * ANGSTROM_TO_BOHR,
    "Er" => 1.89 * ANGSTROM_TO_BOHR,
    "Tm" => 1.90 * ANGSTROM_TO_BOHR,
    "Yb" => 1.87 * ANGSTROM_TO_BOHR,
    "Lu" => 1.87 * ANGSTROM_TO_BOHR,
    "Hf" => 1.75 * ANGSTROM_TO_BOHR,
    "Ta" => 1.70 * ANGSTROM_TO_BOHR,
    "W" => 1.62 * ANGSTROM_TO_BOHR,
    "Re" => 1.51 * ANGSTROM_TO_BOHR,
    "Os" => 1.44 * ANGSTROM_TO_BOHR,
    "Ir" => 1.41 * ANGSTROM_TO_BOHR,
    "Pt" => 1.36 * ANGSTROM_TO_BOHR,
    "Au" => 1.36 * ANGSTROM_TO_BOHR,
    "Hg" => 1.32 * ANGSTROM_TO_BOHR,
    "Tl" => 1.45 * ANGSTROM_TO_BOHR,
    "Pb" => 1.46 * ANGSTROM_TO_BOHR,
    "Bi" => 1.48 * ANGSTROM_TO_BOHR,
    "Po" => 1.40 * ANGSTROM_TO_BOHR,
    "At" => 1.50 * ANGSTROM_TO_BOHR,
    "Rn" => 1.50 * ANGSTROM_TO_BOHR,

    "Fr" => 2.60 * ANGSTROM_TO_BOHR,
    "Ra" => 2.21 * ANGSTROM_TO_BOHR,
    "Ac" => 2.15 * ANGSTROM_TO_BOHR,
    "Th" => 2.06 * ANGSTROM_TO_BOHR,
    "Pa" => 2.00 * ANGSTROM_TO_BOHR,
    "U" => 1.96 * ANGSTROM_TO_BOHR,
    "Np" => 1.90 * ANGSTROM_TO_BOHR,
    "Pu" => 1.87 * ANGSTROM_TO_BOHR,
    "Am" => 1.80 * ANGSTROM_TO_BOHR,
    "Cm" => 1.69 * ANGSTROM_TO_BOHR,
};

// 'STR_TO_ATOMIC_MASS' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
// Unit: A.U.
const MASSUNIT: f64 = 1822.88484264550; 
static STR_TO_ATOMIC_MASS: phf::Map<&'static str, f64> = phf_map!
{
    "H" => 1.00794 * MASSUNIT,
    "He" => 4.002602 * MASSUNIT,

    "Li" => 6.941 * MASSUNIT,
    "Be" => 9.012182 * MASSUNIT,
    "B" => 10.811 * MASSUNIT,
    "C" => 12.0107 * MASSUNIT,
    "N" => 14.0067 * MASSUNIT,
    "O" => 15.9994 * MASSUNIT,
    "F" => 18.9984032 * MASSUNIT,
    "Ne" => 20.1797 * MASSUNIT,

    "Na" => 22.98976928 * MASSUNIT,
    "Mg" => 24.305 * MASSUNIT,
    "Al" => 26.9815386 * MASSUNIT,
    "Si" => 28.0855 * MASSUNIT,
    "P" => 30.973762 * MASSUNIT,
    "S" => 32.065 * MASSUNIT,
    "Cl" => 35.453 * MASSUNIT,
    "Ar" => 39.948 * MASSUNIT,

    "K" => 39.0983 * MASSUNIT,
    "Ca" => 40.078 * MASSUNIT,
    "Sc" => 44.955912 * MASSUNIT,
    "Ti" => 47.867 * MASSUNIT,
    "V" => 50.9415 * MASSUNIT,
    "Cr" => 51.9961 * MASSUNIT,
    "Mn" => 54.938045 * MASSUNIT,
    "Fe" => 55.845 * MASSUNIT,
    "Co" => 58.933195 * MASSUNIT,
    "Ni" => 58.6934 * MASSUNIT,
    "Cu" => 63.546 * MASSUNIT,
    "Zn" => 65.38 * MASSUNIT,
    "Ga" => 69.723 * MASSUNIT,
    "Ge" => 72.64 * MASSUNIT,
    "As" => 74.9216 * MASSUNIT,
    "Se" => 78.96 * MASSUNIT,
    "Br" => 79.904 * MASSUNIT,
    "Kr" => 83.798 * MASSUNIT,

    "Rb" => 85.4678 * MASSUNIT,
    "Sr" => 87.62 * MASSUNIT,
    "Y" => 88.90585 * MASSUNIT,
    "Zr" => 91.224 * MASSUNIT,
    "Nb" => 92.90638 * MASSUNIT,
    "Mo" => 95.96 * MASSUNIT,
    "Tc" => 97.9072 * MASSUNIT,
    "Ru" => 101.07 * MASSUNIT,
    "Rh" => 102.9055 * MASSUNIT,
    "Pd" => 106.42 * MASSUNIT,
    "Ag" => 107.8682 * MASSUNIT,
    "Cd" => 112.411 * MASSUNIT,
    "In" => 114.818 * MASSUNIT,
    "Sn" => 118.71 * MASSUNIT,
    "Sb" => 121.76 * MASSUNIT,
    "Te" => 127.6 * MASSUNIT,
    "I" => 126.90447 * MASSUNIT,
    "Xe" => 131.293 * MASSUNIT,

    "Cs" => 132.9054519 * MASSUNIT,
    "Ba" => 137.327 * MASSUNIT,
    "La" => 138.90547 * MASSUNIT,
    "Ce" => 140.116 * MASSUNIT,
    "Pr" => 140.90765 * MASSUNIT,
    "Nd" => 144.242 * MASSUNIT,
    "Pm" => 144.9127 * MASSUNIT,
    "Sm" => 150.36 * MASSUNIT,
    "Eu" => 151.964 * MASSUNIT,
    "Gd" => 157.25 * MASSUNIT,
    "Tb" => 158.92535 * MASSUNIT,
    "Dy" => 162.5 * MASSUNIT,
    "Ho" => 164.93032 * MASSUNIT,
    "Er" => 167.259 * MASSUNIT,
    "Tm" => 168.93421 * MASSUNIT,
    "Yb" => 173.054 * MASSUNIT,
    "Lu" => 174.9668 * MASSUNIT,
    "Hf" => 178.49 * MASSUNIT,
    "Ta" => 180.94788 * MASSUNIT,
    "W" => 183.84 * MASSUNIT,
    "Re" => 186.207 * MASSUNIT,
    "Os" => 190.23 * MASSUNIT,
    "Ir" => 192.217 * MASSUNIT,
    "Pt" => 195.084 * MASSUNIT,
    "Au" => 196.966569 * MASSUNIT,
    "Hg" => 200.59 * MASSUNIT,
    "Tl" => 204.3833 * MASSUNIT,
    "Pb" => 207.2 * MASSUNIT,
    "Bi" => 208.9804 * MASSUNIT,
    "Po" => 208.9824 * MASSUNIT,
    "At" => 209.9871 * MASSUNIT,
    "Rn" => 222.0176 * MASSUNIT,

    "Fr" => 223.0197 * MASSUNIT,
    "Ra" => 226.0254 * MASSUNIT,
    "Ac" => 227.0 * MASSUNIT,
    "Th" => 232.0377 * MASSUNIT,
    "Pa" => 231.03588 * MASSUNIT,
    "U" => 238.02891 * MASSUNIT,
};

impl Element
{
    pub fn from_str(element: &str) -> Self
    {
        STR_TO_ELEMENT.get(element).cloned().expect(&error_type("element", element))
    }

    pub fn get_atomic_number(&self) -> usize
    {
        STR_TO_ATOMIC_NUMBER.get(format!("{:?}", self).as_str()).cloned().expect(&error_getting_property("atomic number", format!("{:?}", self).as_str()))
    }

    pub fn get_atomic_radius(&self) -> f64
    {
        STR_TO_ATOMIC_RADIUS.get(format!("{:?}", self).as_str()).cloned().expect(&error_getting_property("atomic radius", format!("{:?}", self).as_str()))
    }

    pub fn get_atomic_mass(&self) -> f64
    {
        STR_TO_ATOMIC_MASS.get(format!("{:?}", self).as_str()).cloned().expect(&error_getting_property("atomic mass", format!("{:?}", self).as_str()))
    }

    pub fn get_element_number() -> usize
    {
        STR_TO_ATOMIC_NUMBER.len()
    }
}










