//! Data structures and relative functions for intra-fragment descriptor
use crate::common::constants::{AminoAcid, FragmentType};
use crate::common::error::*;
use crate::nn::protein::Fragment;
use std::collections::HashMap;
use ndarray::{Array1, Array2, Array4, array};
use lazy_static::lazy_static;










/// Derive the intrafragment descriptor for the input fragment based on the atomic coordinates.
///
/// # Parameters
/// ```
/// fragment: the input fragment reference
/// descriptor: the output intrafragment descriptor derived from the atomic coordinates ((natom-2)*3 Vec)
/// gradient: the gradient of the output intrafragment descriptor with respect to the atomic coordinates (I*O Array)
/// ```
pub fn get_intrafragment_descriptor(fragment: &Fragment) -> Option<(Vec<f64>, Array2<f64>)>
{
    match &fragment.fragment_type
    {
        // For amino acid residue
        FragmentType::Residue(amino_acid) =>
        {
            let equi_indices: &'static Vec<Vec<usize>> = AMINO_ACID_TO_EQUI_INDICES.get(&amino_acid).expect(&error_static_hashmap("AminoAcid", "equivalent atoms", "AMINO_ACID_TO_EQUI_INDICES"));
            match amino_acid
            {
                // For the special proline, the indices of the C_alpha atom and the side-chain atom are 10 and 7
                AminoAcid::PRO =>
                {
                    let (descriptor, gradient): (Vec<f64>, Array2<f64>) = get_intraresidue_descriptor(&fragment.coord, 10, 7, equi_indices);
                    return Some((descriptor, gradient));
                },

                // For the other ordinary amino acid residue, the indices of the C_alpha atom and the side-chain atom are 2 and 3
                _ =>
                {
                    let (descriptor, gradient): (Vec<f64>, Array2<f64>) = get_intraresidue_descriptor(&fragment.coord, 2, 3, equi_indices);
                    return Some((descriptor, gradient));
                },
            }
        },

        // For atom or molecule
        _ =>
        {
            return None;
        },
    }
}





/// Based on the atomic coordinates, derive the intraresidue descriptor for the amino acid residue.
///
/// # Parameters
/// ```
/// coord: the input atomic coordinates of the amino acid residue (natom*3 Array)
/// index: the input index of the C_alpha atom
/// index2: the input index of the side-chain atom after removing the C_alpha atom
/// equi_indices: the input indices of the equivalent atoms after removing the C_alpha atom
/// descriptor: the output intraresidue descriptor of the amino acid residue ((natom-2)*3 Vec)
/// gradient: the gradient of the output intraresidue descriptor with respect to the input atomic coordinates (I*O Array)
/// ```
fn get_intraresidue_descriptor(coord: &Array2<f64>, index: usize, index2: usize, equi_indices: &'static Vec<Vec<usize>>) -> (Vec<f64>, Array2<f64>)
{
    let index1: usize = coord.nrows() - 3;

    // Translate the amino acid residue, making the C_alpha atom centered at the origin.
    let (coord_tran, gradient1): (Array2<f64>, &'static Array2<f64>) = center_at_c_alpha(coord, index);                // The second parameter is the index of C_alpha atom originally

    // Schmidt orthonormalization of the fixed coordinate frame
    let s: Array2<f64> = array!
    [
        [ coord_tran[[0,0]], coord_tran[[index1,0]], coord_tran[[index2,0]] ],
        [ coord_tran[[0,1]], coord_tran[[index1,1]], coord_tran[[index2,1]] ],
        [ coord_tran[[0,2]], coord_tran[[index1,2]], coord_tran[[index2,2]] ],
    ];
    let (c, d_c_s): (Array2<f64>, Array4<f64>) = schmidt_orthonormalization(s);

    // Rotate the amino acid residue around the C_alpha atom, making the N atom along the X axis, and the C atom in the XY plane
    let (mut coord_rot, gradient2): (Array2<f64>, Array2<f64>) = rotate_around_c_alpha(coord_tran, index2, c, d_c_s);

    // Average the coordinates of the equivalent atoms to eliminate the permutational symmetry
    let gradient3: Array2<f64> = average_equivalent_atoms(&mut coord_rot, equi_indices);

    // Convert the atomic coordinates of the amino acid residue into intraresidue descriptor
    let (descriptor, gradient4): (Vec<f64>, &'static Array2<f64>) = convert_into_descriptor(coord_rot);

    let gradient: Array2<f64> = gradient1.dot(&gradient2).dot(&gradient3).dot(gradient4);
    (descriptor, gradient)
}










/// Translate the amino acid residue, making the C_alpha atom centered at the origin.
///
/// # Parameters
/// ```
/// coord: the input atomic coordinates of the amino acid residue (natom*3 Array)
/// index: the index of the C_alpha atom
/// coord_tran: the output atomic coordinates of the amino acid residue, with the C_alpha atom centered at the origin ((natom-1)*3 Array)
/// gradient: the gradient of the output atomic coordinates with respect to the input atomic coordinates (I*O Array)
/// ```
///
fn center_at_c_alpha(coord: &Array2<f64>, index: usize) -> (Array2<f64>, &'static Array2<f64>)
{
    let natom: usize = coord.nrows();

    // Translate the C_alpha atom to the origin
    let mut coord_tran: Array2<f64> = Array2::zeros((natom-1, 3));
    for i in 0..natom
    {
        if i < index
        {
            coord_tran[[i,0]] = coord[[i,0]] - coord[[index,0]];
            coord_tran[[i,1]] = coord[[i,1]] - coord[[index,1]];
            coord_tran[[i,2]] = coord[[i,2]] - coord[[index,2]];
        }
        else if i > index
        {
            coord_tran[[i-1,0]] = coord[[i,0]] - coord[[index,0]];
            coord_tran[[i-1,1]] = coord[[i,1]] - coord[[index,1]];
            coord_tran[[i-1,2]] = coord[[i,2]] - coord[[index,2]];
        }
    }

    // Derive the gradient of the output coordinates with respect to the input coordinates
    let gradient: &'static Array2<f64> = NATOM_TO_GRADIENT1.get(&(natom, index)).expect(&error_static_hashmap("natom", "GRADIENT1", "NATOM_TO_GRADIENT1"));

    (coord_tran, gradient)
}





/// Derive the gradient for 'center_at_c_alpha' transformation
///
/// # Parameters
/// ```
/// natom: the input number of atoms of the amino acid residue
/// index: the index of the C_alpha atom
/// gradient: the gradient of the output atomic coordinates with respect to the input atomic coordinates (I*O Array)
/// ```
///
fn center_at_c_alpha_gradient(natom: usize, index: usize) -> Array2<f64>
{
    // Derive the gradient of the output coordinates with respect to the input coordinates
    let mut gradient: Array2<f64> = Array2::zeros((3*natom, 3*natom-3));
    let mut index_3i: usize;
    for i in 0..(natom-1)
    {
        index_3i = 3 * i;
        if i < index
        {
            gradient[[index_3i, index_3i]] = 1.0;
            gradient[[index_3i+1, index_3i+1]] = 1.0;
            gradient[[index_3i+2, index_3i+2]] = 1.0;
        }
        else
        {
            gradient[[index_3i+3, index_3i]] = 1.0;
            gradient[[index_3i+4, index_3i+1]] = 1.0;
            gradient[[index_3i+5, index_3i+2]] = 1.0;
        }
        gradient[[3*index, index_3i]] = -1.0;
        gradient[[3*index+1, index_3i+1]] = -1.0;
        gradient[[3*index+2, index_3i+2]] = -1.0;
    }

    gradient
}





/// Schmidt orthonormalization for 3D Euclidean space
///
/// # Parameters
/// ```
/// s: the three input non-orthonormal axes (3*3 Array, column vector)
/// c: the three output orthonormal axes (3*3 Array, column vector)
/// gradient: the gradient of the output axes with respect to the input axes (3*3*3*3 Array)
/// ```
///
fn schmidt_orthonormalization(s: Array2<f64>) -> (Array2<f64>, Array4<f64>)
{
    let mut gradient: Array4<f64> = Array4::zeros((3, 3, 3, 3));

    // Normalization of the first axis
    let mut norm_1: f64 = 1.0 / (s[[0,0]] * s[[0,0]] + s[[1,0]] * s[[1,0]] + s[[2,0]] * s[[2,0]]).sqrt();                // The inverse of the norm of the first axis
    let mut norm_3: f64 = norm_1.powi(3);
    let mut c: Array2<f64> = array!
    [
        [ s[[0,0]]*norm_1, 0.0, 0.0 ],
        [ s[[1,0]]*norm_1, 0.0, 0.0 ],
        [ s[[2,0]]*norm_1, 0.0, 0.0 ],
    ];
    gradient[[0,0,0,0]] = norm_1 - s[[0,0]] * s[[0,0]] * norm_3;
    gradient[[0,0,1,0]] = -s[[0,0]] * s[[1,0]] * norm_3;
    gradient[[0,0,2,0]] = -s[[0,0]] * s[[2,0]] * norm_3;
    gradient[[1,0,0,0]] = -s[[1,0]] * s[[0,0]] * norm_3;
    gradient[[1,0,1,0]] = norm_1 - s[[1,0]] * s[[1,0]] * norm_3;
    gradient[[1,0,2,0]] = -s[[1,0]] * s[[2,0]] * norm_3;
    gradient[[2,0,0,0]] = -s[[2,0]] * s[[0,0]] * norm_3;
    gradient[[2,0,1,0]] = -s[[2,0]] * s[[1,0]] * norm_3;
    gradient[[2,0,2,0]] = norm_1 - s[[2,0]] * s[[2,0]] * norm_3;

    // Orthogonalization of the second axis
    let mut pro0: f64 = s[[0,1]] * c[[0,0]] + s[[1,1]] * c[[1,0]] + s[[2,1]] * c[[2,0]];                // The projection of the second axis s(:,1) on the new first axis c(:,0)
    let mut d_pro0_s00: f64 = s[[0,1]] * gradient[[0,0,0,0]] + s[[1,1]] * gradient[[0,0,1,0]] + s[[2,1]] * gradient[[0,0,2,0]];
    let mut d_pro0_s10: f64 = s[[0,1]] * gradient[[1,0,0,0]] + s[[1,1]] * gradient[[1,0,1,0]] + s[[2,1]] * gradient[[1,0,2,0]];
    let mut d_pro0_s20: f64 = s[[0,1]] * gradient[[2,0,0,0]] + s[[1,1]] * gradient[[2,0,1,0]] + s[[2,1]] * gradient[[2,0,2,0]];
    let mut sc0: f64 = s[[0,1]] - pro0 * c[[0,0]];
    let mut sc1: f64 = s[[1,1]] - pro0 * c[[1,0]];
    let mut sc2: f64 = s[[2,1]] - pro0 * c[[2,0]];
    let mut d_sc0_s00: f64 = -d_pro0_s00 * c[[0,0]] - pro0 * gradient[[0,0,0,0]];
    let mut d_sc0_s10: f64 = -d_pro0_s10 * c[[0,0]] - pro0 * gradient[[1,0,0,0]];
    let mut d_sc0_s20: f64 = -d_pro0_s20 * c[[0,0]] - pro0 * gradient[[2,0,0,0]];
    let mut d_sc0_s01: f64 = 1.0 - c[[0,0]] * c[[0,0]];
    let mut d_sc0_s11: f64 = -c[[1,0]] * c[[0,0]];
    let mut d_sc0_s21: f64 = -c[[2,0]] * c[[0,0]];
    let mut d_sc1_s00: f64 = -d_pro0_s00 * c[[1,0]] - pro0 * gradient[[0,0,1,0]];
    let mut d_sc1_s10: f64 = -d_pro0_s10 * c[[1,0]] - pro0 * gradient[[1,0,1,0]];
    let mut d_sc1_s20: f64 = -d_pro0_s20 * c[[1,0]] - pro0 * gradient[[2,0,1,0]];
    let mut d_sc1_s01: f64 = -c[[0,0]] * c[[1,0]];
    let mut d_sc1_s11: f64 = 1.0 - c[[1,0]] * c[[1,0]];
    let mut d_sc1_s21: f64 = -c[[2,0]] * c[[1,0]];
    let mut d_sc2_s00: f64 = -d_pro0_s00 * c[[2,0]] - pro0 * gradient[[0,0,2,0]];
    let mut d_sc2_s10: f64 = -d_pro0_s10 * c[[2,0]] - pro0 * gradient[[1,0,2,0]];
    let mut d_sc2_s20: f64 = -d_pro0_s20 * c[[2,0]] - pro0 * gradient[[2,0,2,0]];
    let mut d_sc2_s01: f64 = -c[[0,0]] * c[[2,0]];
    let mut d_sc2_s11: f64 = -c[[1,0]] * c[[2,0]];
    let mut d_sc2_s21: f64 = 1.0 - c[[2,0]] * c[[2,0]];

    // Normalization of the second axis
    norm_1 = 1.0 / (sc0 * sc0 + sc1 * sc1 + sc2 * sc2).sqrt();                // The inverse of the norm of the second axis
    norm_3 = norm_1.powi(3);
    c[[0,1]] = sc0 * norm_1;
    c[[1,1]] = sc1 * norm_1;
    c[[2,1]] = sc2 * norm_1;
    let mut d_norm_s00: f64 = sc0 * d_sc0_s00 + sc1 * d_sc1_s00 + sc2 * d_sc2_s00;
    let mut d_norm_s10: f64 = sc0 * d_sc0_s10 + sc1 * d_sc1_s10 + sc2 * d_sc2_s10;
    let mut d_norm_s20: f64 = sc0 * d_sc0_s20 + sc1 * d_sc1_s20 + sc2 * d_sc2_s20;
    let mut d_norm_s01: f64 = sc0 * d_sc0_s01 + sc1 * d_sc1_s01 + sc2 * d_sc2_s01;
    let mut d_norm_s11: f64 = sc0 * d_sc0_s11 + sc1 * d_sc1_s11 + sc2 * d_sc2_s11;
    let mut d_norm_s21: f64 = sc0 * d_sc0_s21 + sc1 * d_sc1_s21 + sc2 * d_sc2_s21;
    gradient[[0,0,0,1]] = d_sc0_s00 * norm_1 - sc0 * d_norm_s00 * norm_3;
    gradient[[0,0,1,1]] = d_sc1_s00 * norm_1 - sc1 * d_norm_s00 * norm_3;
    gradient[[0,0,2,1]] = d_sc2_s00 * norm_1 - sc2 * d_norm_s00 * norm_3;
    gradient[[0,1,0,1]] = d_sc0_s01 * norm_1 - sc0 * d_norm_s01 * norm_3;
    gradient[[0,1,1,1]] = d_sc1_s01 * norm_1 - sc1 * d_norm_s01 * norm_3;
    gradient[[0,1,2,1]] = d_sc2_s01 * norm_1 - sc2 * d_norm_s01 * norm_3;
    gradient[[1,0,0,1]] = d_sc0_s10 * norm_1 - sc0 * d_norm_s10 * norm_3;
    gradient[[1,0,1,1]] = d_sc1_s10 * norm_1 - sc1 * d_norm_s10 * norm_3;
    gradient[[1,0,2,1]] = d_sc2_s10 * norm_1 - sc2 * d_norm_s10 * norm_3;
    gradient[[1,1,0,1]] = d_sc0_s11 * norm_1 - sc0 * d_norm_s11 * norm_3;
    gradient[[1,1,1,1]] = d_sc1_s11 * norm_1 - sc1 * d_norm_s11 * norm_3;
    gradient[[1,1,2,1]] = d_sc2_s11 * norm_1 - sc2 * d_norm_s11 * norm_3;
    gradient[[2,0,0,1]] = d_sc0_s20 * norm_1 - sc0 * d_norm_s20 * norm_3;
    gradient[[2,0,1,1]] = d_sc1_s20 * norm_1 - sc1 * d_norm_s20 * norm_3;
    gradient[[2,0,2,1]] = d_sc2_s20 * norm_1 - sc2 * d_norm_s20 * norm_3;
    gradient[[2,1,0,1]] = d_sc0_s21 * norm_1 - sc0 * d_norm_s21 * norm_3;
    gradient[[2,1,1,1]] = d_sc1_s21 * norm_1 - sc1 * d_norm_s21 * norm_3;
    gradient[[2,1,2,1]] = d_sc2_s21 * norm_1 - sc2 * d_norm_s21 * norm_3;

    // Orthogonalization of the third axis
    pro0 = s[[0,2]] * c[[0,0]] + s[[1,2]] * c[[1,0]] + s[[2,2]] * c[[2,0]];                // The projection of the third axis s(:,2) on the new first axis c(:,0)
    let pro1: f64 = s[[0,2]] * c[[0,1]] + s[[1,2]] * c[[1,1]] + s[[2,2]] * c[[2,1]];                // The projection of the third axis s(:,2) on the new second axis c(:,1)
    d_pro0_s00 = s[[0,2]] * gradient[[0,0,0,0]] + s[[1,2]] * gradient[[0,0,1,0]] + s[[2,2]] * gradient[[0,0,2,0]];
    d_pro0_s10 = s[[0,2]] * gradient[[1,0,0,0]] + s[[1,2]] * gradient[[1,0,1,0]] + s[[2,2]] * gradient[[1,0,2,0]];
    d_pro0_s20 = s[[0,2]] * gradient[[2,0,0,0]] + s[[1,2]] * gradient[[2,0,1,0]] + s[[2,2]] * gradient[[2,0,2,0]];
    let d_pro1_s00: f64 = s[[0,2]] * gradient[[0,0,0,1]] + s[[1,2]] * gradient[[0,0,1,1]] + s[[2,2]] * gradient[[0,0,2,1]];
    let d_pro1_s10: f64 = s[[0,2]] * gradient[[1,0,0,1]] + s[[1,2]] * gradient[[1,0,1,1]] + s[[2,2]] * gradient[[1,0,2,1]];
    let d_pro1_s20: f64 = s[[0,2]] * gradient[[2,0,0,1]] + s[[1,2]] * gradient[[2,0,1,1]] + s[[2,2]] * gradient[[2,0,2,1]];
    let d_pro1_s01: f64 = s[[0,2]] * gradient[[0,1,0,1]] + s[[1,2]] * gradient[[0,1,1,1]] + s[[2,2]] * gradient[[0,1,2,1]];
    let d_pro1_s11: f64 = s[[0,2]] * gradient[[1,1,0,1]] + s[[1,2]] * gradient[[1,1,1,1]] + s[[2,2]] * gradient[[1,1,2,1]];
    let d_pro1_s21: f64 = s[[0,2]] * gradient[[2,1,0,1]] + s[[1,2]] * gradient[[2,1,1,1]] + s[[2,2]] * gradient[[2,1,2,1]];
    sc0 = s[[0,2]] - pro0 * c[[0,0]] - pro1 * c[[0,1]];
    sc1 = s[[1,2]] - pro0 * c[[1,0]] - pro1 * c[[1,1]];
    sc2 = s[[2,2]] - pro0 * c[[2,0]] - pro1 * c[[2,1]];
    d_sc0_s00 = -d_pro0_s00 * c[[0,0]] - pro0 * gradient[[0,0,0,0]] - d_pro1_s00 * c[[0,1]] - pro1 * gradient[[0,0,0,1]];
    d_sc0_s10 = -d_pro0_s10 * c[[0,0]] - pro0 * gradient[[1,0,0,0]] - d_pro1_s10 * c[[0,1]] - pro1 * gradient[[1,0,0,1]];
    d_sc0_s20 = -d_pro0_s20 * c[[0,0]] - pro0 * gradient[[2,0,0,0]] - d_pro1_s20 * c[[0,1]] - pro1 * gradient[[2,0,0,1]];
    d_sc0_s01 = -d_pro1_s01 * c[[0,1]] - pro1 * gradient[[0,1,0,1]];
    d_sc0_s11 = -d_pro1_s11 * c[[0,1]] - pro1 * gradient[[1,1,0,1]];
    d_sc0_s21 = -d_pro1_s21 * c[[0,1]] - pro1 * gradient[[2,1,0,1]];
    let d_sc0_s02: f64 = 1.0 - c[[0,0]] * c[[0,0]] - c[[0,1]] * c[[0,1]];
    let d_sc0_s12: f64 = -c[[1,0]] * c[[0,0]] - c[[1,1]] * c[[0,1]];
    let d_sc0_s22: f64 = -c[[2,0]] * c[[0,0]] - c[[2,1]] * c[[0,1]];
    d_sc1_s00 = -d_pro0_s00 * c[[1,0]] - pro0 * gradient[[0,0,1,0]] - d_pro1_s00 * c[[1,1]] - pro1 * gradient[[0,0,1,1]];
    d_sc1_s10 = -d_pro0_s10 * c[[1,0]] - pro0 * gradient[[1,0,1,0]] - d_pro1_s10 * c[[1,1]] - pro1 * gradient[[1,0,1,1]];
    d_sc1_s20 = -d_pro0_s20 * c[[1,0]] - pro0 * gradient[[2,0,1,0]] - d_pro1_s20 * c[[1,1]] - pro1 * gradient[[2,0,1,1]];
    d_sc1_s01 = -d_pro1_s01 * c[[1,1]] - pro1 * gradient[[0,1,1,1]];
    d_sc1_s11 = -d_pro1_s11 * c[[1,1]] - pro1 * gradient[[1,1,1,1]];
    d_sc1_s21 = -d_pro1_s21 * c[[1,1]] - pro1 * gradient[[2,1,1,1]];
    let d_sc1_s02: f64 = -c[[0,0]] * c[[1,0]] - c[[0,1]] * c[[1,1]];
    let d_sc1_s12: f64 = 1.0 - c[[1,0]] * c[[1,0]] - c[[1,1]] * c[[1,1]];
    let d_sc1_s22: f64 = -c[[2,0]] * c[[1,0]] - c[[2,1]] * c[[1,1]];
    d_sc2_s00 = -d_pro0_s00 * c[[2,0]] - pro0 * gradient[[0,0,2,0]] - d_pro1_s00 * c[[2,1]] - pro1 * gradient[[0,0,2,1]];
    d_sc2_s10 = -d_pro0_s10 * c[[2,0]] - pro0 * gradient[[1,0,2,0]] - d_pro1_s10 * c[[2,1]] - pro1 * gradient[[1,0,2,1]];
    d_sc2_s20 = -d_pro0_s20 * c[[2,0]] - pro0 * gradient[[2,0,2,0]] - d_pro1_s20 * c[[2,1]] - pro1 * gradient[[2,0,2,1]];
    d_sc2_s01 = -d_pro1_s01 * c[[2,1]] - pro1 * gradient[[0,1,2,1]];
    d_sc2_s11 = -d_pro1_s11 * c[[2,1]] - pro1 * gradient[[1,1,2,1]];
    d_sc2_s21 = -d_pro1_s21 * c[[2,1]] - pro1 * gradient[[2,1,2,1]];
    let d_sc2_s02: f64 = -c[[0,0]] * c[[2,0]] - c[[0,1]] * c[[2,1]];
    let d_sc2_s12: f64 = -c[[1,0]] * c[[2,0]] - c[[1,1]] * c[[2,1]];
    let d_sc2_s22: f64 = 1.0 - c[[2,0]] * c[[2,0]] - c[[2,1]] * c[[2,1]];

    // Normalization of the third axis 
    norm_1 = 1.0 / (sc0 * sc0 + sc1 * sc1 + sc2 * sc2).sqrt();                // The inverse of the norm of the third axis
    norm_3 = norm_1.powi(3);
    c[[0,2]] = sc0 * norm_1;
    c[[1,2]] = sc1 * norm_1;
    c[[2,2]] = sc2 * norm_1;
    d_norm_s00 = sc0 * d_sc0_s00 + sc1 * d_sc1_s00 + sc2 * d_sc2_s00;
    d_norm_s10 = sc0 * d_sc0_s10 + sc1 * d_sc1_s10 + sc2 * d_sc2_s10;
    d_norm_s20 = sc0 * d_sc0_s20 + sc1 * d_sc1_s20 + sc2 * d_sc2_s20;
    d_norm_s01 = sc0 * d_sc0_s01 + sc1 * d_sc1_s01 + sc2 * d_sc2_s01;
    d_norm_s11 = sc0 * d_sc0_s11 + sc1 * d_sc1_s11 + sc2 * d_sc2_s11;
    d_norm_s21 = sc0 * d_sc0_s21 + sc1 * d_sc1_s21 + sc2 * d_sc2_s21;
    let d_norm_s02: f64 = sc0 * d_sc0_s02 + sc1 * d_sc1_s02 + sc2 * d_sc2_s02;
    let d_norm_s12: f64 = sc0 * d_sc0_s12 + sc1 * d_sc1_s12 + sc2 * d_sc2_s12;
    let d_norm_s22: f64 = sc0 * d_sc0_s22 + sc1 * d_sc1_s22 + sc2 * d_sc2_s22;
    gradient[[0,0,0,2]] = d_sc0_s00 * norm_1 - sc0 * d_norm_s00 * norm_3;
    gradient[[0,0,1,2]] = d_sc1_s00 * norm_1 - sc1 * d_norm_s00 * norm_3;
    gradient[[0,0,2,2]] = d_sc2_s00 * norm_1 - sc2 * d_norm_s00 * norm_3;
    gradient[[0,1,0,2]] = d_sc0_s01 * norm_1 - sc0 * d_norm_s01 * norm_3;
    gradient[[0,1,1,2]] = d_sc1_s01 * norm_1 - sc1 * d_norm_s01 * norm_3;
    gradient[[0,1,2,2]] = d_sc2_s01 * norm_1 - sc2 * d_norm_s01 * norm_3;
    gradient[[0,2,0,2]] = d_sc0_s02 * norm_1 - sc0 * d_norm_s02 * norm_3;
    gradient[[0,2,1,2]] = d_sc1_s02 * norm_1 - sc1 * d_norm_s02 * norm_3;
    gradient[[0,2,2,2]] = d_sc2_s02 * norm_1 - sc2 * d_norm_s02 * norm_3;
    gradient[[1,0,0,2]] = d_sc0_s10 * norm_1 - sc0 * d_norm_s10 * norm_3;
    gradient[[1,0,1,2]] = d_sc1_s10 * norm_1 - sc1 * d_norm_s10 * norm_3;
    gradient[[1,0,2,2]] = d_sc2_s10 * norm_1 - sc2 * d_norm_s10 * norm_3;
    gradient[[1,1,0,2]] = d_sc0_s11 * norm_1 - sc0 * d_norm_s11 * norm_3;
    gradient[[1,1,1,2]] = d_sc1_s11 * norm_1 - sc1 * d_norm_s11 * norm_3;
    gradient[[1,1,2,2]] = d_sc2_s11 * norm_1 - sc2 * d_norm_s11 * norm_3;
    gradient[[1,2,0,2]] = d_sc0_s12 * norm_1 - sc0 * d_norm_s12 * norm_3;
    gradient[[1,2,1,2]] = d_sc1_s12 * norm_1 - sc1 * d_norm_s12 * norm_3;
    gradient[[1,2,2,2]] = d_sc2_s12 * norm_1 - sc2 * d_norm_s12 * norm_3;
    gradient[[2,0,0,2]] = d_sc0_s20 * norm_1 - sc0 * d_norm_s20 * norm_3;
    gradient[[2,0,1,2]] = d_sc1_s20 * norm_1 - sc1 * d_norm_s20 * norm_3;
    gradient[[2,0,2,2]] = d_sc2_s20 * norm_1 - sc2 * d_norm_s20 * norm_3;
    gradient[[2,1,0,2]] = d_sc0_s21 * norm_1 - sc0 * d_norm_s21 * norm_3;
    gradient[[2,1,1,2]] = d_sc1_s21 * norm_1 - sc1 * d_norm_s21 * norm_3;
    gradient[[2,1,2,2]] = d_sc2_s21 * norm_1 - sc2 * d_norm_s21 * norm_3;
    gradient[[2,2,0,2]] = d_sc0_s22 * norm_1 - sc0 * d_norm_s22 * norm_3;
    gradient[[2,2,1,2]] = d_sc1_s22 * norm_1 - sc1 * d_norm_s22 * norm_3;
    gradient[[2,2,2,2]] = d_sc2_s22 * norm_1 - sc2 * d_norm_s22 * norm_3;

    (c, gradient)
}





/// Rotate the amino acid residue around the C_alpha atom, making the N atom along the X axis, and the C atom in the XY plane
///
/// # Parameters
/// ```
/// coord_tran: the input atomic coordinates of the amino acid residue, with the C_alpha atom at the origin ((natom-1)*3 Array)
/// index2: the index of the side chain atom
/// c: the 3*3 rotation matrix
/// d_c_s: the gradient of c with respect to s (please see function schmidt_orthonormalization)
/// coord_rot: the output atomic coordinates of the amino acid residue, with the N atom along the X axis, and the C atom in the XY plane ((natom-1)*3 Array)
/// gradient: the gradient of the output atomic coordinates with respect to the input atomic coordinates (I*O Array)
/// ```
///
fn rotate_around_c_alpha(coord_tran: Array2<f64>, index2: usize, c: Array2<f64>, d_c_s: Array4<f64>) -> (Array2<f64>, Array2<f64>)
{
    let natom: usize = coord_tran.nrows();
    let index1: usize = natom - 2;
    let coord_rot: Array2<f64> = coord_tran.dot(&c);
    let mut gradient: Array2<f64> = Array2::zeros((natom*3, natom*3));

    for m in 0..natom
    {
        for n in 0..3
        {
            for i in 0..natom
            {
                for j in 0..3
                {
                    if m == 0                // The index of N atom is always 0
                    {
                        gradient[[m*3+n, i*3+j]] = coord_tran[[i,0]] * d_c_s[[n,0,0,j]] + coord_tran[[i,1]] * d_c_s[[n,0,1,j]] + coord_tran[[i,2]] * d_c_s[[n,0,2,j]];
                    }
                    if m == index1
                    {
                        gradient[[m*3+n, i*3+j]] = coord_tran[[i,0]] * d_c_s[[n,1,0,j]] + coord_tran[[i,1]] * d_c_s[[n,1,1,j]] + coord_tran[[i,2]] * d_c_s[[n,1,2,j]];
                    }
                    if m == index2
                    {
                        gradient[[m*3+n, i*3+j]] = coord_tran[[i,0]] * d_c_s[[n,2,0,j]] + coord_tran[[i,1]] * d_c_s[[n,2,1,j]] + coord_tran[[i,2]] * d_c_s[[n,2,2,j]];
                    }
                    if m == i
                    {
                        gradient[[m*3+n, i*3+j]] += c[[n, j]];
                    }
                }
            }
        }
    }

    (coord_rot, gradient)
}





/*
/// Average the coordinates of the equivalent atoms to eliminate the permutational symmetry
/// The basis function is x^(l)
///
/// # Parameters
/// ```
/// coord: the input and output atomic coordinates of the amino acid residue ((natom-1)*3 Array)
/// equi_index: the indices of the equivalent atoms
/// gradient: the gradient of the output atomic coordinates with respect to the input atomic coordinates (I*O Array)
/// ```
///
fn average_equivalent_atoms(coord: &mut Array2<f64>, equi_index: &'static Vec<Vec<usize>>) -> Array2<f64>
{
    let mut gradient: Array2<f64> = Array2::eye(coord.nrows()*3);

    let mut n: usize;
    for i in 0..equi_index.len()                // For each group of equivalent atoms
    {
        n = equi_index[i].len();
        let mut power: Array2<f64> = Array2::ones((n, n+1));
        let mut average: Array1<f64>;
        for j in 0..3                // For XYZ coordinates
        {
            for k in 0..n                // For each equivalent atom
            {
                for l in 1..(n+1)                // For each power
                {
                    power[[k,l]] = coord[[equi_index[i][k], j]].powi(l.try_into().unwrap());
                }
            }
            average = power.mean_axis(Axis(0)).unwrap();

            for k in 0..n                // For each input equivalent atom
            {
                coord[[equi_index[i][k], j]] = average[[k+1]];
                for l in 0..n                // For each output equivalent atom
                {
                    gradient[[equi_index[i][k]*3+j, equi_index[i][l]*3+j]] = power[[k,l]] * ((l+1) as f64) / (n as f64);
                }
            }
        }
    }

    gradient
}
*/





/*
/// Average the coordinates of the equivalent atoms to eliminate the permutational symmetry
/// The basis function is 10sin(x/10l)
///
/// # Parameters
/// ```
/// coord: the input and output atomic coordinates of the amino acid residue ((natom-1)*3 Array)
/// equi_index: the indices of the equivalent atoms
/// gradient: the gradient of the output atomic coordinates with respect to the input atomic coordinates (I*O Array)
/// ```
///
fn average_equivalent_atoms(coord: &mut Array2<f64>, equi_index: &'static Vec<Vec<usize>>) -> Array2<f64>
{
    let mut gradient: Array2<f64> = Array2::eye(coord.nrows()*3);

    let mut n: usize;
    let mut x: f64;
    for i in 0..equi_index.len()                // For each group of equivalent atoms
    {
        n = equi_index[i].len();
        let mut sum: Array1<f64>;
        for j in 0..3                // For XYZ coordinates
        {
            sum = Array1::zeros(n);
            for k in 0..n                // For each input equivalent atom
            {
                for l in 0..n                // For each basis function
                {
                    x = coord[[equi_index[i][k], j]] / (10 * (l+1)) as f64;
                    sum[l] += x.sin();
                    gradient[[equi_index[i][k]*3+j, equi_index[i][l]*3+j]] = x.cos() / (n * (l+1)) as f64;
                }
            }

            for k in 0..n                // For each output equivalent atom
            {
                coord[[equi_index[i][k], j]] = sum[k] * 10.0 / (n as f64);
            }
        }
    }

    gradient
}
*/





/// Average the coordinates of the equivalent atoms to eliminate the permutational symmetry
/// The basis function is 5tanh(lx/5)
///
/// # Parameters
/// ```
/// coord: the input and output atomic coordinates of the amino acid residue ((natom-1)*3 Array)
/// equi_indices: the input indices of the equivalent atoms
/// gradient: the gradient of the output atomic coordinates with respect to the input atomic coordinates (I*O Array)
/// ```
///
fn average_equivalent_atoms(coord: &mut Array2<f64>, equi_indices: &'static Vec<Vec<usize>>) -> Array2<f64>
{
    let mut gradient: Array2<f64> = Array2::eye(coord.nrows()*3);

    let mut n: usize;
    let mut x: f64;
    for i in 0..equi_indices.len()                // For each group of equivalent atoms
    {
        n = equi_indices[i].len();
        let mut sum: Array1<f64>;
        for j in 0..3                // For XYZ coordinates
        {
            sum = Array1::zeros(n);
            for k in 0..n                // For each input equivalent atom
            {
                for l in 0..n                // For each basis function
                {
                    x = coord[[equi_indices[i][k], j]] * ((l+1) as f64) / 5.0;
                    sum[l] += x.tanh();
                    gradient[[equi_indices[i][k]*3+j, equi_indices[i][l]*3+j]] = ((l+1) as f64) / (n as f64) / x.cosh().powi(2);
                }
            }

            for k in 0..n                // For each output equivalent atom
            {
                coord[[equi_indices[i][k], j]] = sum[k] * 5.0 / (n as f64);
            }
        }
    }

    gradient
}





/// Convert the atomic coordinates of the amino acid residue into intraresidue descriptor
///
/// # Parameters
/// ```
/// coord_rot: the input atomic coordinates of the amino acid residue, with the N atom along the X axis, and the C atom in the XY plane ((natom-1)*3 Array)
/// index1: the index of the C atom
/// descriptor: the output descriptor of the amino acid residue ((natom-2)*3 Vec)
/// gradient: the gradient of the output descriptor with respect to the input atomic coordinates (I*O Array)
/// ```
///
fn convert_into_descriptor(coord_rot: Array2<f64>) -> (Vec<f64>, &'static Array2<f64>)
{
    let natom: usize = coord_rot.nrows();
    let mut descriptor: Vec<f64> = Vec::with_capacity((natom-1)*3);

    descriptor.push(coord_rot[[0,0]]);
    for i in 1..natom
    {
        if i == (natom-2)
        {
            descriptor.push(coord_rot[[i,0]]);
            descriptor.push(coord_rot[[i,1]]);
        }
        else
        {
            descriptor.push(coord_rot[[i,0]]);
            descriptor.push(coord_rot[[i,1]]);
            descriptor.push(coord_rot[[i,2]]);
        }
    }

    let gradient: &'static Array2<f64> = NATOM_TO_GRADIENT4.get(&natom).expect(&error_static_hashmap("natom", "GRADIENT4", "NATOM_TO_GRADIENT4"));

    (descriptor, gradient)
}





/// Derive the gradient for 'convert_into_descriptor' transformation
///
/// # Parameters
/// ```
/// natom: the input number of atoms of the amino acid residue
/// gradient: the gradient of the output descriptor with respect to the input atomic coordinates (I*O Array)
/// ```
///
fn convert_into_descriptor_gradient(natom: usize) -> Array2<f64>
{
    let mut gradient: Array2<f64> = Array2::zeros((natom*3, (natom-1)*3));

    gradient[[0,0]] = 1.0;
    for i in 1..natom
    {
        if i < (natom-2)
        {
            gradient[[i*3, i*3-2]] = 1.0;
            gradient[[i*3+1, i*3-1]] = 1.0;
            gradient[[i*3+2, i*3]] = 1.0;
        }
        if i == (natom-2)
        {
            gradient[[i*3, i*3-2]] = 1.0;
            gradient[[i*3+1, i*3-1]] = 1.0;
        }
        if i > (natom-2)
        {
            gradient[[i*3, i*3-3]] = 1.0;
            gradient[[i*3+1, i*3-2]] = 1.0;
            gradient[[i*3+2, i*3-1]] = 1.0;
        }
    }

    gradient
}










lazy_static!
{
    // Based on the number of atoms and the index of the C_alpha atom, derive the gradient for 'center_at_c_alpha_gradient' transformation
    static ref GRADIENT1_7_2: Array2<f64> = center_at_c_alpha_gradient(7, 2);
    static ref GRADIENT1_10_2: Array2<f64> = center_at_c_alpha_gradient(10, 2);
    static ref GRADIENT1_11_2: Array2<f64> = center_at_c_alpha_gradient(11, 2);
    static ref GRADIENT1_12_2: Array2<f64> = center_at_c_alpha_gradient(12, 2);
    static ref GRADIENT1_13_2: Array2<f64> = center_at_c_alpha_gradient(13, 2);
    static ref GRADIENT1_14_2: Array2<f64> = center_at_c_alpha_gradient(14, 2);
    static ref GRADIENT1_15_2: Array2<f64> = center_at_c_alpha_gradient(15, 2);
    static ref GRADIENT1_16_2: Array2<f64> = center_at_c_alpha_gradient(16, 2);
    static ref GRADIENT1_17_2: Array2<f64> = center_at_c_alpha_gradient(17, 2);
    static ref GRADIENT1_18_2: Array2<f64> = center_at_c_alpha_gradient(18, 2);
    static ref GRADIENT1_19_2: Array2<f64> = center_at_c_alpha_gradient(19, 2);
    static ref GRADIENT1_20_2: Array2<f64> = center_at_c_alpha_gradient(20, 2);
    static ref GRADIENT1_21_2: Array2<f64> = center_at_c_alpha_gradient(21, 2);
    static ref GRADIENT1_22_2: Array2<f64> = center_at_c_alpha_gradient(22, 2);
    static ref GRADIENT1_23_2: Array2<f64> = center_at_c_alpha_gradient(23, 2);
    static ref GRADIENT1_24_2: Array2<f64> = center_at_c_alpha_gradient(24, 2);
    static ref GRADIENT1_14_10: Array2<f64> = center_at_c_alpha_gradient(14, 10);

    static ref NATOM_TO_GRADIENT1: HashMap<(usize, usize), &'static Array2<f64>> =
    {
        let mut natom_to_gradient1: HashMap<(usize, usize), &'static Array2<f64>> = HashMap::new();

        let gradient1_7_2: &'static Array2<f64> = &GRADIENT1_7_2;
        let gradient1_10_2: &'static Array2<f64> = &GRADIENT1_10_2;
        let gradient1_11_2: &'static Array2<f64> = &GRADIENT1_11_2;
        let gradient1_12_2: &'static Array2<f64> = &GRADIENT1_12_2;
        let gradient1_13_2: &'static Array2<f64> = &GRADIENT1_13_2;
        let gradient1_14_2: &'static Array2<f64> = &GRADIENT1_14_2;
        let gradient1_15_2: &'static Array2<f64> = &GRADIENT1_15_2;
        let gradient1_16_2: &'static Array2<f64> = &GRADIENT1_16_2;
        let gradient1_17_2: &'static Array2<f64> = &GRADIENT1_17_2;
        let gradient1_18_2: &'static Array2<f64> = &GRADIENT1_18_2;
        let gradient1_19_2: &'static Array2<f64> = &GRADIENT1_19_2;
        let gradient1_20_2: &'static Array2<f64> = &GRADIENT1_20_2;
        let gradient1_21_2: &'static Array2<f64> = &GRADIENT1_21_2;
        let gradient1_22_2: &'static Array2<f64> = &GRADIENT1_22_2;
        let gradient1_23_2: &'static Array2<f64> = &GRADIENT1_23_2;
        let gradient1_24_2: &'static Array2<f64> = &GRADIENT1_24_2;
        let gradient1_14_10: &'static Array2<f64> = &GRADIENT1_14_10;

        natom_to_gradient1.insert((7, 2), gradient1_7_2);
        natom_to_gradient1.insert((10, 2), gradient1_10_2);
        natom_to_gradient1.insert((11, 2), gradient1_11_2);
        natom_to_gradient1.insert((12, 2), gradient1_12_2);
        natom_to_gradient1.insert((13, 2), gradient1_13_2);
        natom_to_gradient1.insert((14, 2), gradient1_14_2);
        natom_to_gradient1.insert((15, 2), gradient1_15_2);
        natom_to_gradient1.insert((16, 2), gradient1_16_2);
        natom_to_gradient1.insert((17, 2), gradient1_17_2);
        natom_to_gradient1.insert((18, 2), gradient1_18_2);
        natom_to_gradient1.insert((19, 2), gradient1_19_2);
        natom_to_gradient1.insert((20, 2), gradient1_20_2);
        natom_to_gradient1.insert((21, 2), gradient1_21_2);
        natom_to_gradient1.insert((22, 2), gradient1_22_2);
        natom_to_gradient1.insert((23, 2), gradient1_23_2);
        natom_to_gradient1.insert((24, 2), gradient1_24_2);
        natom_to_gradient1.insert((14, 10), gradient1_14_10);

        natom_to_gradient1
    };





    // Based on the number of atoms, derive the gradient for the 'convert_into_descriptor' transformation
    static ref GRADIENT4_6: Array2<f64> = convert_into_descriptor_gradient(6);
    static ref GRADIENT4_9: Array2<f64> = convert_into_descriptor_gradient(9);
    static ref GRADIENT4_10: Array2<f64> = convert_into_descriptor_gradient(10);
    static ref GRADIENT4_11: Array2<f64> = convert_into_descriptor_gradient(11);
    static ref GRADIENT4_12: Array2<f64> = convert_into_descriptor_gradient(12);
    static ref GRADIENT4_13: Array2<f64> = convert_into_descriptor_gradient(13);
    static ref GRADIENT4_14: Array2<f64> = convert_into_descriptor_gradient(14);
    static ref GRADIENT4_15: Array2<f64> = convert_into_descriptor_gradient(15);
    static ref GRADIENT4_16: Array2<f64> = convert_into_descriptor_gradient(16);
    static ref GRADIENT4_17: Array2<f64> = convert_into_descriptor_gradient(17);
    static ref GRADIENT4_18: Array2<f64> = convert_into_descriptor_gradient(18);
    static ref GRADIENT4_19: Array2<f64> = convert_into_descriptor_gradient(19);
    static ref GRADIENT4_20: Array2<f64> = convert_into_descriptor_gradient(20);
    static ref GRADIENT4_21: Array2<f64> = convert_into_descriptor_gradient(21);
    static ref GRADIENT4_22: Array2<f64> = convert_into_descriptor_gradient(22);
    static ref GRADIENT4_23: Array2<f64> = convert_into_descriptor_gradient(23);

    static ref NATOM_TO_GRADIENT4: HashMap<usize, &'static Array2<f64>> =
    {
        let mut natom_to_gradient4: HashMap<usize, &'static Array2<f64>> = HashMap::new();

        let gradient4_6: &'static Array2<f64> = &GRADIENT4_6;
        let gradient4_9: &'static Array2<f64> = &GRADIENT4_9;
        let gradient4_10: &'static Array2<f64> = &GRADIENT4_10;
        let gradient4_11: &'static Array2<f64> = &GRADIENT4_11;
        let gradient4_12: &'static Array2<f64> = &GRADIENT4_12;
        let gradient4_13: &'static Array2<f64> = &GRADIENT4_13;
        let gradient4_14: &'static Array2<f64> = &GRADIENT4_14;
        let gradient4_15: &'static Array2<f64> = &GRADIENT4_15;
        let gradient4_16: &'static Array2<f64> = &GRADIENT4_16;
        let gradient4_17: &'static Array2<f64> = &GRADIENT4_17;
        let gradient4_18: &'static Array2<f64> = &GRADIENT4_18;
        let gradient4_19: &'static Array2<f64> = &GRADIENT4_19;
        let gradient4_20: &'static Array2<f64> = &GRADIENT4_20;
        let gradient4_21: &'static Array2<f64> = &GRADIENT4_21;
        let gradient4_22: &'static Array2<f64> = &GRADIENT4_22;
        let gradient4_23: &'static Array2<f64> = &GRADIENT4_23;

        natom_to_gradient4.insert(6, gradient4_6);
        natom_to_gradient4.insert(9, gradient4_9);
        natom_to_gradient4.insert(10, gradient4_10);
        natom_to_gradient4.insert(11, gradient4_11);
        natom_to_gradient4.insert(12, gradient4_12);
        natom_to_gradient4.insert(13, gradient4_13);
        natom_to_gradient4.insert(14, gradient4_14);
        natom_to_gradient4.insert(15, gradient4_15);
        natom_to_gradient4.insert(16, gradient4_16);
        natom_to_gradient4.insert(17, gradient4_17);
        natom_to_gradient4.insert(18, gradient4_18);
        natom_to_gradient4.insert(19, gradient4_19);
        natom_to_gradient4.insert(20, gradient4_20);
        natom_to_gradient4.insert(21, gradient4_21);
        natom_to_gradient4.insert(22, gradient4_22);
        natom_to_gradient4.insert(23, gradient4_23);

        natom_to_gradient4
    };





    // Based on the type of AminoAcid, achieve the indices of the equivalent atoms (after removing the C_alpha atom) for the amino acid residue
    static ref EQUI_INDICES_OF_GLY: Vec<Vec<usize>> = Vec::new();
    static ref EQUI_INDICES_OF_ALA: Vec<Vec<usize>> = vec![vec![4, 5, 6]];
    static ref EQUI_INDICES_OF_VAL: Vec<Vec<usize>> = vec![vec![5, 9], vec![6, 7, 8, 10, 11, 12]];
    static ref EQUI_INDICES_OF_LEU: Vec<Vec<usize>> = vec![vec![4, 5], vec![8, 12], vec![9, 10, 11, 13, 14, 15]];
    static ref EQUI_INDICES_OF_ILE: Vec<Vec<usize>> = vec![vec![6, 7, 8], vec![10, 11], vec![13, 14, 15]];
    static ref EQUI_INDICES_OF_SER: Vec<Vec<usize>> = vec![vec![4, 5]];
    static ref EQUI_INDICES_OF_THR: Vec<Vec<usize>> = vec![vec![6, 7, 8]];
    static ref EQUI_INDICES_OF_ASP: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 8]];
    static ref EQUI_INDICES_OF_ASH: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 8]];
    static ref EQUI_INDICES_OF_ASN: Vec<Vec<usize>> = vec![vec![4, 5], vec![9, 10]];
    static ref EQUI_INDICES_OF_GLU: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 8], vec![10, 11]];
    static ref EQUI_INDICES_OF_GLH: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 8], vec![10, 11]];
    static ref EQUI_INDICES_OF_GLN: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 8], vec![12, 13]];
    static ref EQUI_INDICES_OF_LYS: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 8], vec![10, 11], vec![13, 14], vec![16, 17, 18]];
    static ref EQUI_INDICES_OF_LYN: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 8], vec![10, 11], vec![13, 14], vec![16, 17]];
    static ref EQUI_INDICES_OF_ARG: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 8], vec![10, 11], vec![15, 18], vec![16, 17, 19, 20]];
    static ref EQUI_INDICES_OF_ARN: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 8], vec![10, 11], vec![15, 18], vec![16, 17, 19]];
    static ref EQUI_INDICES_OF_CYS: Vec<Vec<usize>> = vec![vec![4, 5]];
    static ref EQUI_INDICES_OF_CYX: Vec<Vec<usize>> = vec![vec![4, 5]];
    static ref EQUI_INDICES_OF_MET: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 8], vec![11, 12, 13]];
    static ref EQUI_INDICES_OF_HID: Vec<Vec<usize>> = vec![vec![4, 5]];
    static ref EQUI_INDICES_OF_HIE: Vec<Vec<usize>> = vec![vec![4, 5]];
    static ref EQUI_INDICES_OF_HIP: Vec<Vec<usize>> = vec![vec![4, 5]];
    static ref EQUI_INDICES_OF_PHE: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 15], vec![8, 16], vec![9, 13], vec![10, 14]];
    static ref EQUI_INDICES_OF_TYR: Vec<Vec<usize>> = vec![vec![4, 5], vec![7, 16], vec![8, 17], vec![9, 14], vec![10, 15]];
    static ref EQUI_INDICES_OF_TRP: Vec<Vec<usize>> = vec![vec![4, 5]];
    static ref EQUI_INDICES_OF_PRO: Vec<Vec<usize>> = vec![vec![2, 3], vec![5, 6], vec![8, 9]];

    static ref AMINO_ACID_TO_EQUI_INDICES: HashMap<AminoAcid, &'static Vec<Vec<usize>>> =
    {
        let mut amino_acid_to_equi_indices: HashMap<AminoAcid, &'static Vec<Vec<usize>>> = HashMap::new();

        let equi_indices_of_gly: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_GLY;
        let equi_indices_of_ala: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_ALA;
        let equi_indices_of_val: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_VAL;
        let equi_indices_of_leu: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_LEU;
        let equi_indices_of_ile: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_ILE;
        let equi_indices_of_ser: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_SER;
        let equi_indices_of_thr: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_THR;
        let equi_indices_of_asp: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_ASP;
        let equi_indices_of_ash: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_ASH;
        let equi_indices_of_asn: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_ASN;
        let equi_indices_of_glu: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_GLU;
        let equi_indices_of_glh: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_GLH;
        let equi_indices_of_gln: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_GLN;
        let equi_indices_of_lys: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_LYS;
        let equi_indices_of_lyn: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_LYN;
        let equi_indices_of_arg: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_ARG;
        let equi_indices_of_arn: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_ARN;
        let equi_indices_of_cys: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_CYS;
        let equi_indices_of_cyx: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_CYX;
        let equi_indices_of_met: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_MET;
        let equi_indices_of_hid: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_HID;
        let equi_indices_of_hie: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_HIE;
        let equi_indices_of_hip: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_HIP;
        let equi_indices_of_phe: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_PHE;
        let equi_indices_of_tyr: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_TYR;
        let equi_indices_of_trp: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_TRP;
        let equi_indices_of_pro: &'static Vec<Vec<usize>> = &EQUI_INDICES_OF_PRO;

        amino_acid_to_equi_indices.insert(AminoAcid::GLY, equi_indices_of_gly);
        amino_acid_to_equi_indices.insert(AminoAcid::ALA, equi_indices_of_ala);
        amino_acid_to_equi_indices.insert(AminoAcid::VAL, equi_indices_of_val);
        amino_acid_to_equi_indices.insert(AminoAcid::LEU, equi_indices_of_leu);
        amino_acid_to_equi_indices.insert(AminoAcid::ILE, equi_indices_of_ile);
        amino_acid_to_equi_indices.insert(AminoAcid::SER, equi_indices_of_ser);
        amino_acid_to_equi_indices.insert(AminoAcid::THR, equi_indices_of_thr);
        amino_acid_to_equi_indices.insert(AminoAcid::ASP, equi_indices_of_asp);
        amino_acid_to_equi_indices.insert(AminoAcid::ASH, equi_indices_of_ash);
        amino_acid_to_equi_indices.insert(AminoAcid::ASN, equi_indices_of_asn);
        amino_acid_to_equi_indices.insert(AminoAcid::GLU, equi_indices_of_glu);
        amino_acid_to_equi_indices.insert(AminoAcid::GLH, equi_indices_of_glh);
        amino_acid_to_equi_indices.insert(AminoAcid::GLN, equi_indices_of_gln);
        amino_acid_to_equi_indices.insert(AminoAcid::LYS, equi_indices_of_lys);
        amino_acid_to_equi_indices.insert(AminoAcid::LYN, equi_indices_of_lyn);
        amino_acid_to_equi_indices.insert(AminoAcid::ARG, equi_indices_of_arg);
        amino_acid_to_equi_indices.insert(AminoAcid::ARN, equi_indices_of_arn);
        amino_acid_to_equi_indices.insert(AminoAcid::CYS, equi_indices_of_cys);
        amino_acid_to_equi_indices.insert(AminoAcid::CYX, equi_indices_of_cyx);
        amino_acid_to_equi_indices.insert(AminoAcid::MET, equi_indices_of_met);
        amino_acid_to_equi_indices.insert(AminoAcid::HID, equi_indices_of_hid);
        amino_acid_to_equi_indices.insert(AminoAcid::HIE, equi_indices_of_hie);
        amino_acid_to_equi_indices.insert(AminoAcid::HIP, equi_indices_of_hip);
        amino_acid_to_equi_indices.insert(AminoAcid::PHE, equi_indices_of_phe);
        amino_acid_to_equi_indices.insert(AminoAcid::TYR, equi_indices_of_tyr);
        amino_acid_to_equi_indices.insert(AminoAcid::TRP, equi_indices_of_trp);
        amino_acid_to_equi_indices.insert(AminoAcid::PRO, equi_indices_of_pro);

        amino_acid_to_equi_indices
    };
}










