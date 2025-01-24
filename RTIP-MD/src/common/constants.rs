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
pub const C_LIGHT: f64 = 299792458.0;
pub const H_PLANCK: f64 = 6.62606896E-34;
pub const BOLTZMANN: f64 = 1.3806504E-23;
pub const RYBDERG: f64 = 10973731.568527;

pub const BOHR_TO_ANGSTROM: f64 = 0.52917720859;
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_TO_ANGSTROM;

pub const HARTREE_TO_JOULE: f64 = 2.0 * RYBDERG * H_PLANCK * C_LIGHT;
pub const JOULE_TO_HARTREE: f64 = 1.0 / HARTREE_TO_JOULE;

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

pub const H_ATOM_TYPE: [Element; 1] = [Element::H];
pub const B_ATOM_TYPE: [Element; 1] = [Element::B];
pub const C_ATOM_TYPE: [Element; 1] = [Element::C];
pub const N_ATOM_TYPE: [Element; 1] = [Element::N];
pub const O_ATOM_TYPE: [Element; 1] = [Element::O];
pub const F_ATOM_TYPE: [Element; 1] = [Element::F];
pub const NA_ATOM_TYPE: [Element; 1] = [Element::Na];
pub const MG_ATOM_TYPE: [Element; 1] = [Element::Mg];
pub const SI_ATOM_TYPE: [Element; 1] = [Element::Si];
pub const P_ATOM_TYPE: [Element; 1] = [Element::P];
pub const S_ATOM_TYPE: [Element; 1] = [Element::S];
pub const CL_ATOM_TYPE: [Element; 1] = [Element::Cl];
pub const K_ATOM_TYPE: [Element; 1] = [Element::K];
pub const CA_ATOM_TYPE: [Element; 1] = [Element::Ca];
pub const V_ATOM_TYPE: [Element; 1] = [Element::V];
pub const CR_ATOM_TYPE: [Element; 1] = [Element::Cr];
pub const MN_ATOM_TYPE: [Element; 1] = [Element::Mn];
pub const FE_ATOM_TYPE: [Element; 1] = [Element::Fe];
pub const CO_ATOM_TYPE: [Element; 1] = [Element::Co];
pub const NI_ATOM_TYPE: [Element; 1] = [Element::Ni];
pub const CU_ATOM_TYPE: [Element; 1] = [Element::Cu];
pub const ZN_ATOM_TYPE: [Element; 1] = [Element::Zn];
pub const AS_ATOM_TYPE: [Element; 1] = [Element::As];
pub const SE_ATOM_TYPE: [Element; 1] = [Element::Se];
pub const BR_ATOM_TYPE: [Element; 1] = [Element::Br];
pub const MO_ATOM_TYPE: [Element; 1] = [Element::Mo];
pub const CD_ATOM_TYPE: [Element; 1] = [Element::Cd];
pub const SN_ATOM_TYPE: [Element; 1] = [Element::Sn];
pub const I_ATOM_TYPE: [Element; 1] = [Element::I];

// 'STR_ELEMENT' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
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

// 'STR_ATOMIC_NUMBER' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
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

// 'STR_ATOMIC_MASS' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
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










// Biological

#[derive(Clone, Debug, PartialEq, Eq, Hash, Savefile)]
pub enum AminoAcid
{
    GLY, ALA, VAL, LEU, ILE,
    SER, THR, ASP, ASH, ASN, GLU, GLH, GLN, LYS, LYN, ARG, ARN,
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
pub const ARN_ATOM_TYPE: [Element; 23] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::C, Element::H, Element::H, Element::N, Element::H, Element::C, Element::N, Element::H, Element::H, Element::N, Element::H, Element::C, Element::O,];
pub const CYS_ATOM_TYPE: [Element; 11] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::S, Element::H, Element::C, Element::O,];
pub const CYX_ATOM_TYPE: [Element; 10] = [Element::N, Element::H, Element::C, Element::H, Element::C, Element::H, Element::H, Element::S, Element::C, Element::O,];
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
    Head (Element),
    Tail (Element),
}

// 'STR_FRAGMENT_TYPE' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
static STR_TO_FRAGMENT_TYPE: phf::Map<&'static str, FragmentType> = phf_map!
{
    "H" => FragmentType::Atom(Element::H),
    "H+" => FragmentType::Atom(Element::H),
    "B" => FragmentType::Atom(Element::B),
    "C" => FragmentType::Atom(Element::C),
    "N" => FragmentType::Atom(Element::N),
    "O" => FragmentType::Atom(Element::O),
    "F" => FragmentType::Atom(Element::F),
    "F-" => FragmentType::Atom(Element::F),
    "Na" => FragmentType::Atom(Element::Na),
    "Na+" => FragmentType::Atom(Element::Na),
    "Mg" => FragmentType::Atom(Element::Mg),
    "Mg2+" => FragmentType::Atom(Element::Mg),
    "Si" => FragmentType::Atom(Element::Si),
    "P" => FragmentType::Atom(Element::P),
    "S" => FragmentType::Atom(Element::S),
    "Cl" => FragmentType::Atom(Element::Cl),
    "Cl-" => FragmentType::Atom(Element::Cl),
    "K" => FragmentType::Atom(Element::K),
    "K+" => FragmentType::Atom(Element::K),
    "Ca" => FragmentType::Atom(Element::Ca),
    "Ca2+" => FragmentType::Atom(Element::Ca),
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
    "As" => FragmentType::Atom(Element::As),
    "Se" => FragmentType::Atom(Element::Se),
    "Se2-" => FragmentType::Atom(Element::Se),
    "Se4+" => FragmentType::Atom(Element::Se),
    "Br" => FragmentType::Atom(Element::Br),
    "Br-" => FragmentType::Atom(Element::Br),
    "Mo" => FragmentType::Atom(Element::Mo),
    "Cd" => FragmentType::Atom(Element::Cd),
    "Sn" => FragmentType::Atom(Element::Sn),
    "Sn2+" => FragmentType::Atom(Element::Sn),
    "I" => FragmentType::Atom(Element::I),
    "I-" => FragmentType::Atom(Element::I),

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
    "ARN" => FragmentType::Residue(AminoAcid::ARN),
    "CYS" => FragmentType::Residue(AminoAcid::CYS),
    "CYX" => FragmentType::Residue(AminoAcid::CYX),
    "MET" => FragmentType::Residue(AminoAcid::MET),
    "HID" => FragmentType::Residue(AminoAcid::HID),
    "HIE" => FragmentType::Residue(AminoAcid::HIE),
    "HIP" => FragmentType::Residue(AminoAcid::HIP),
    "PHE" => FragmentType::Residue(AminoAcid::PHE),
    "TYR" => FragmentType::Residue(AminoAcid::TYR),
    "TRP" => FragmentType::Residue(AminoAcid::TRP),
    "PRO" => FragmentType::Residue(AminoAcid::PRO),

    "WAT" => FragmentType::Molecule(Molecule::WAT),
};

// 'STR_NATOM' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
static STR_TO_NATOM: phf::Map<&'static str, usize> = phf_map!
{
    "H" => 1,
    "H+" => 1,
    "B" => 1,
    "C" => 1,
    "N" => 1,
    "O" => 1,
    "F" => 1,
    "F-" => 1,
    "Na" => 1,
    "Na+" => 1,
    "Mg" => 1,
    "Mg2+" => 1,
    "Si" => 1,
    "P" => 1,
    "S" => 1,
    "Cl" => 1,
    "Cl-" => 1,
    "K" => 1,
    "K+" => 1,
    "Ca" => 1,
    "Ca2+" => 1,
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
    "As" => 1,
    "Se" => 1,
    "Se2-" => 1,
    "Se4+" => 1,
    "Br" => 1,
    "Br-" => 1,
    "Mo" => 1,
    "Cd" => 1,
    "Sn" => 1,
    "Sn2+" => 1,
    "I" => 1,
    "I-" => 1,

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
    "ARN" => 23,
    "CYS" => 11,
    "CYX" => 10,
    "MET" => 17,
    "HID" => 17,
    "HIE" => 17,
    "HIP" => 18,
    "PHE" => 20,
    "TYR" => 21,
    "TRP" => 24,
    "PRO" => 14,

    "WAT" => 3,
};

// 'STR_ATOM_TYPE' is a static structure of type 'phf::Map', initialized by macro 'phf_map'
static STR_TO_ATOM_TYPE: phf::Map<&'static str, &[Element]> = phf_map!
{
    "H" => &H_ATOM_TYPE,
    "H+" => &H_ATOM_TYPE,
    "B" => &B_ATOM_TYPE,
    "C" => &C_ATOM_TYPE,
    "N" => &N_ATOM_TYPE,
    "O" => &O_ATOM_TYPE,
    "F" => &F_ATOM_TYPE,
    "F-" => &F_ATOM_TYPE,
    "Na" => &NA_ATOM_TYPE,
    "Na+" => &NA_ATOM_TYPE,
    "Mg" => &MG_ATOM_TYPE,
    "Mg2+" => &MG_ATOM_TYPE,
    "Si" => &SI_ATOM_TYPE,
    "P" => &P_ATOM_TYPE,
    "S" => &S_ATOM_TYPE,
    "Cl" => &CL_ATOM_TYPE,
    "Cl-" => &CL_ATOM_TYPE,
    "K" => &K_ATOM_TYPE,
    "K+" => &K_ATOM_TYPE,
    "Ca" => &CA_ATOM_TYPE,
    "Ca2+" => &CA_ATOM_TYPE,
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
    "As" => &AS_ATOM_TYPE,
    "Se" => &SE_ATOM_TYPE,
    "Se2-" => &SE_ATOM_TYPE,
    "Se4+" => &SE_ATOM_TYPE,
    "Br" => &BR_ATOM_TYPE,
    "Br-" => &BR_ATOM_TYPE,
    "Mo" => &MO_ATOM_TYPE,
    "Cd" => &CD_ATOM_TYPE,
    "Sn" => &SN_ATOM_TYPE,
    "Sn2+" => &SN_ATOM_TYPE,
    "I" => &I_ATOM_TYPE,
    "I-" => &I_ATOM_TYPE,

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
    "ARN" => &ARN_ATOM_TYPE,
    "CYS" => &CYS_ATOM_TYPE,
    "CYX" => &CYX_ATOM_TYPE,
    "MET" => &MET_ATOM_TYPE,
    "HID" => &HID_ATOM_TYPE,
    "HIE" => &HIE_ATOM_TYPE,
    "HIP" => &HIP_ATOM_TYPE,
    "PHE" => &PHE_ATOM_TYPE,
    "TYR" => &TYR_ATOM_TYPE,
    "TRP" => &TRP_ATOM_TYPE,
    "PRO" => &PRO_ATOM_TYPE,

    "WAT" => &WAT_ATOM_TYPE,
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










