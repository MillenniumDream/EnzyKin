//! About the derivative of complex spherical harmonics

use num::Complex;
use sphrs::{SphrsFloat, SHCoordinates, sh};





/// Calculate the derivative of complex spherical harmonics with respect to theta (dY/dtheta)
///
/// # Parameters
/// ```
/// l: degree of complex spherical harmonics
/// m: order of complex spherical harmonics
/// p: reference to the coordinate (with information about theta and phi)
/// ```
///
/// # Examples
/// ```
/// ```
pub fn dsh_dtheta<T: SphrsFloat>(l: i64, m: i64, p: &impl SHCoordinates<T>) -> Complex<T>
{
    assert!(l >= 0);
    assert!(m.abs() <= l);

    // If l is 0, the complex spherical harmonics is constant, return 0
    if l == 0
    {
        Complex::new(T::zero(), T::zero())
    }
    // Otherwise, calculate the derivative
    else
    {
        // 0.5 + 0.0i
        let half: Complex<T> = Complex::new(T::from_f64(0.5).unwrap(), T::zero());

        // If m is -l, only the (m+1) term is nonzero
        if m == -l
        {
            let tmp1: Complex<T> = Complex::cis(-p.phi());
            let u1: i64 = (l - m) * (l + m + 1);
            let k1: Complex<T> = Complex::new(T::from_i64(u1).unwrap().sqrt(), T::zero());
            let y1: Complex<T> = sh(l, m+1, p);

            half * tmp1 * k1 * y1
        }
        // If m is l, only the (m-1) term is nonzero
        else if m == l
        {
            let tmp2: Complex<T> = Complex::cis(p.phi());
            let u2: i64 = (l + m) * (l - m + 1);
            let k2: Complex<T> = Complex::new(T::from_i64(u2).unwrap().sqrt(), T::zero());
            let y2: Complex<T> = sh(l, m-1, p);

            -half * tmp2 * k2 * y2
        }
        // Otherwise, both the (m+1) term and the (m-1) term is nonzero
        else
        {
            let tmp1: Complex<T> = Complex::cis(-p.phi());
            let tmp2: Complex<T> = Complex::cis(p.phi());
            let u1: i64 = (l - m) * (l + m + 1);
            let u2: i64 = (l + m) * (l - m + 1);
            let k1: Complex<T> = Complex::new(T::from_i64(u1).unwrap().sqrt(), T::zero());
            let k2: Complex<T> = Complex::new(T::from_i64(u2).unwrap().sqrt(), T::zero());
            let y1: Complex<T> = sh(l, m+1, p);
            let y2: Complex<T> = sh(l, m-1, p);

            half * (tmp1 * k1 * y1 - tmp2 * k2 * y2)
        }
    }
}





/// Calculate the derivative of complex spherical harmonics with respect to phi (dY/dphi)
///
/// # Parameters
/// ```
/// l: degree of complex spherical harmonics
/// m: order of complex spherical harmonics
/// p: reference to the coordinate (with information about theta and phi)
/// ```
///
/// # Examples
/// ```
/// ```
pub fn dsh_dphi<T: SphrsFloat>(l: i64, m: i64, p: &impl SHCoordinates<T>) -> Complex<T>
{
    assert!(l >= 0);
    assert!(m.abs() <= l);

    let k: Complex<T> = Complex::new(T::zero(), T::from_i64(m).unwrap());
    k * sh(l, m, p)
}





/// Calculate the derivative of complex spherical harmonics with respect to x, y, and z (dY/dx, dY/dy, dY/dz)
///
/// # Parameters
/// ```
/// l: degree of complex spherical harmonics
/// m: order of complex spherical harmonics
/// p: reference to the coordinate (with information about x, y, and z)
/// ```
///
/// # Examples
/// ```
/// ```
pub fn dsh_dxyz<T: SphrsFloat>(l: i64, m: i64, p: &impl SHCoordinates<T>) -> (Complex<T>, Complex<T>, Complex<T>)
{
    assert!(l >= 0);
    assert!(m.abs() <= l);

    // If l is 0, the complex spherical harmonics is constant, return 0
    if l == 0
    {
        ( Complex::new(T::zero(), T::zero()), Complex::new(T::zero(), T::zero()), Complex::new(T::zero(), T::zero()) )
    }
    else
    {
        let mut u1: i64;
        let mut u2: i64;
        let v1: Complex<T> = Complex::new(T::from_i64( (2 * l + 1) * (2 * l + 3) ).unwrap().sqrt(), T::zero());
        let v2: Complex<T> = Complex::new(T::from_i64( (2 * l + 1) * (2 * l - 1) ).unwrap().sqrt(), T::zero());
        let mut k1: Complex<T>;
        let mut k2: Complex<T>;
        let mut y1: Complex<T>;
        let mut y2: Complex<T>;

        // Calculate dY/dz
        let dy_dz: Complex<T> = if m == -l || m == l
        {
            u1 = (l - m + 1) * (l + m + 1);
            k1 = Complex::new(T::from_i64(l).unwrap(), T::zero()) * Complex::new(T::from_i64(u1).unwrap().sqrt(), T::zero()) / v1 / p.r();
            y1 = sh(l+1, m, p);
            -k1 * y1
        }
        else
        {
            u1 = (l - m + 1) * (l + m + 1);
            u2 = (l - m) * (l + m);
            k1 = Complex::new(T::from_i64(l).unwrap(), T::zero()) * Complex::new(T::from_i64(u1).unwrap().sqrt(), T::zero()) / v1 / p.r();
            k2 = Complex::new(T::from_i64(l+1).unwrap(), T::zero()) * Complex::new(T::from_i64(u2).unwrap().sqrt(), T::zero()) / v2 / p.r();
            y1 = sh(l+1, m, p);
            y2 = sh(l-1, m, p);
            -k1 * y1 + k2 * y2
        };

        // Calculate (dY/dx + idY/dy)
        let dy_dxy1: Complex<T> = if m == l || m == (l-1)
        {
            u1 = (l + m + 1) * (l + m + 2);
            k1 = Complex::new(T::from_i64(l).unwrap(), T::zero()) * Complex::new(T::from_i64(u1).unwrap().sqrt(), T::zero()) / v1 / p.r();
            y1 = sh(l+1, m+1, p);
            k1 * y1
        }
        else
        {
            u1 = (l + m + 1) * (l + m + 2);
            u2 = (l - m) * (l - m - 1);
            k1 = Complex::new(T::from_i64(l).unwrap(), T::zero()) * Complex::new(T::from_i64(u1).unwrap().sqrt(), T::zero()) / v1 / p.r();
            k2 = Complex::new(T::from_i64(l+1).unwrap(), T::zero()) * Complex::new(T::from_i64(u2).unwrap().sqrt(), T::zero()) / v2 / p.r();
            y1 = sh(l+1, m+1, p);
            y2 = sh(l-1, m+1, p);
            k1 * y1 + k2 * y2
        };

        // Calculate (dY/dx - idY/dy)
        let dy_dxy2: Complex<T> = if m == (-l) || m == (-l+1)
        {
            u1 = (l - m + 1) * (l - m + 2);
            k1 = Complex::new(T::from_i64(l).unwrap(), T::zero()) * Complex::new(T::from_i64(u1).unwrap().sqrt(), T::zero()) / v1 / p.r();
            y1 = sh(l+1, m-1, p);
            -k1 * y1
        }
        else
        {
            u1 = (l - m + 1) * (l - m + 2);
            u2 = (l + m) * (l + m - 1);
            k1 = Complex::new(T::from_i64(l).unwrap(), T::zero()) * Complex::new(T::from_i64(u1).unwrap().sqrt(), T::zero()) / v1 / p.r();
            k2 = Complex::new(T::from_i64(l+1).unwrap(), T::zero()) * Complex::new(T::from_i64(u2).unwrap().sqrt(), T::zero()) / v2 / p.r();
            y1 = sh(l+1, m-1, p);
            y2 = sh(l-1, m-1, p);
            -k1 * y1 - k2 * y2
        };

        // Calculate dY/dx and dY/dy
        let dy_dx: Complex<T> = (dy_dxy1 + dy_dxy2) / Complex::new(T::from_f64(2.0).unwrap(), T::zero());
        let dy_dy: Complex<T> = (dy_dxy1 - dy_dxy2) / Complex::new(T::zero(), T::from_f64(2.0).unwrap());

        (dy_dx, dy_dy, dy_dz)
    }
}










