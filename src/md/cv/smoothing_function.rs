//! About the bond orientational order parameter introduced by Steinhardt et al., as collective variable (CV) for enhanced sampling.










/// Basic structure for quintic smoothing function, containing the inner cutoff (Rin) and outer cutoff (Rcut)
///
/// # Fields
/// ```
/// r_in: given distance for inner cutoff (Unit: bohr)
/// r_cut: given distance for outer cutoff (Unit: bohr)
/// inv_delta_r: 1 / (r_cut - r_in) (Unit: bohr)
/// ```
#[derive(Debug)]
pub struct QuinticSmooth
{
    pub r_in: f64,
    pub r_cut: f64,
    pub inv_delta_r: f64,
}





impl QuinticSmooth
{
    /// Construct a new QuinticSmooth directly
    ///
    /// # Parameters
    /// ```
    /// r_in: given distance for inner cutoff (Unit: bohr)
    /// r_cut: given distance for outer cutoff (Unit: bohr)
    /// ```
    ///
    /// # Examples
    /// ```
    /// let quintic_smooth: QuinticSmooth = QuinticSmooth::new(r_in, r_cut);
    /// ```
    pub fn new(r_in: f64, r_cut: f64) -> Self
    {
        assert!(r_in.abs() < r_cut.abs());

        QuinticSmooth
        {
            r_in: r_in.abs(),
            r_cut: r_cut.abs(),
            inv_delta_r: 1.0 / (r_cut - r_in),
        }
    }



    /// Calculate the quintic smoothing function value for the input distance
    ///
    /// # Parameters
    /// ```
    /// r: the input distance (Unit: bohr)
    /// w: the output smoothing function value
    /// ```
    ///
    /// # Examples
    /// ```
    /// let w: f64 = quintic_smooth.f(r);
    /// ```
    pub fn f(&self, r: f64) -> f64
    {
        if r <= self.r_in
        {
            1.0
        }
        else if r >= self.r_cut
        {
            0.0
        }
        else
        {
            let x: f64 = (r - self.r_in) * self.inv_delta_r;
            -6.0 * x.powi(5) + 15.0 * x.powi(4) - 10.0 * x.powi(3) + 1.0
        }
    }



    /// Calculate derivative of quintic smoothing function for the input distance
    ///
    /// # Parameters
    /// ```
    /// r: the input distance (Unit: bohr)
    /// dw: the output derivative of the smoothing function
    /// ```
    ///
    /// # Examples
    /// ```
    /// let dw: f64 = quintic_smooth.df(r);
    /// ```
    pub fn df(&self, r: f64) -> f64
    {
        if r <= self.r_in
        {
            0.0
        }
        else if r >= self.r_cut
        {
            0.0
        }
        else
        {
            let x: f64 = (r - self.r_in) * self.inv_delta_r;
            ( -30.0 * x.powi(4) + 60.0 * x.powi(3) - 30.0 * x.powi(2) ) * self.inv_delta_r
        }
    }
}










