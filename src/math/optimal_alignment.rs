//! A module for solving the optimal alighment problem by the singular value decomposition (SVD) method
//! Of note, both the rotation and inversion are taken into account.

use crate::common::error::*;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::SVDInto;





/// Input the atomic coordinates of two systems (coord1, coord2),
/// output the minimum alignment distance between the two atomic coordinates
/// under corresponding transformation (orth, tran), which makes ( coord2 - (coord1 * orth + tran) ) minimum.
///
/// # Parameters
/// ```
/// coord1: the atomic coordinates of System 1 (natom * 3)
/// coord2: the atomic coordinates of System 2 (natom * 3)
/// dist: the output minimum alignment distance between coord1 and coord2
/// vec: the output alignment vector from coord1 to coord2
/// orth: the output orthogonal matrix (3 * 3)
/// tran: the output translational vector (1 * 3)
/// ```
pub fn svd_dist_vec_orth_tran(coord1: &Array2<f64>, coord2: &Array2<f64>) -> (f64, Array2<f64>, Array2<f64>, Array1<f64>)
{
    // Move the geometric centers of the two systems to the origin
    let o1: Array1<f64> = coord1.mean_axis(Axis(0)).expect(&error_none_value("coord1"));
    let o2: Array1<f64> = coord2.mean_axis(Axis(0)).expect(&error_none_value("coord2"));
    let coord1: Array2<f64> = coord1 - &o1;
    let coord2: Array2<f64> = coord2 - &o2;

    // Singular value decomposition
    let b: Array2<f64> = coord1.t().dot(&coord2);
    let (u, _sigma, vt) = b.svd_into(true, true).expect(&error_none_value("(u, sigma, vt)"));
    let u: Array2<f64> = u.expect(&error_none_value("u"));
    let vt: Array2<f64> = vt.expect(&error_none_value("vt"));

    // Calculate the orthogonal matrix, translational matrix, the alignment vector, and the alignment distance
    let orth: Array2<f64> = u.dot(&vt);
    let tran: Array1<f64> = o2 - o1.dot(&orth);
    let vec: Array2<f64> = coord2 - coord1.dot(&orth);
    let dist: f64 = vec.mapv(|a| a.powi(2)).sum().sqrt();

    (dist, vec, orth, tran)
}










