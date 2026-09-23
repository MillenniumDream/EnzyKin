use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use crate::common::constants::*;
use crate::common::error::*;
use crate::common::traits::*;
use crate::io::input::Para;
use crate::pes_exploration::system::System;
use mpi::traits::Communicator;
use mpi::collective::Root;
use ndarray::{Array1, Array2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;





/// 1D biasing force for extended adaptive biasing force and adaptive thermoleveling force (eABF-ATF)
///
/// # Fields
/// ```
/// cv_min: minimum CV value in eABF-ATF (Unit: bohr)
/// cv_max: maximum CV value in eABF-ATF (Unit: bohr)
/// cv_bin: CV bin in eABF-ATF (Unit: bohr)
/// num_bin: number of CV bins
/// num_half_full: half of number of full sample for the biasing force (of type f64 for convenient)
/// num_full: number of full sampling for the biasing force (of type f64 for convenient)
/// accum_num: accumulated number of biasing force in each CV bin (of type f64 for convenient)
/// accum_force: accumulated sum of biasing force in each CV bin (Unit: A.U.)
/// delta_force: kB*delta_T/cv_bin, basic unit for the thermoleveling force (Unit: A.U.)
/// k: elastic coefficient of the harmonic potential of the boundary (A.U.)
/// ```
#[derive(Debug)]
pub struct BiasingForce1D
{
    pub cv_min: f64,
    pub cv_max: f64,
    pub cv_bin: f64,
    pub num_bin: usize,
    pub num_half_full: f64,
    pub num_full: f64,
    pub accum_num: Vec<f64>,
    pub accum_force: Vec<f64>,
    pub delta_force: f64,
    pub k: Option<f64>,
}





/// 1D mean force for extended adaptive biasing force and adaptive thermoleveling force (eABF-ATF)
///
/// # Fields
/// ```
/// cv_min: minimum CV value in eABF-ATF (Unit: bohr)
/// cv_max: maximum CV value in eABF-ATF (Unit: bohr)
/// cv_bin: CV bin in eABF-ATF (Unit: bohr)
/// num_bin: number of CV bins
/// num_full: number of full sampling for the mean force (of type f64 for convenient)
/// accum_num: accumulated number of sampling force in each CV bin (of type f64 for convenient)
/// accum_force: accumulated sum of sampling force in each CV bin (Unit: A.U.)
/// delta_energy: kB*delta_T, basic unit for the thermoleveling energy (Unit: A.U.)
/// ```
#[derive(Debug)]
pub struct MeanForce1D
{
    pub cv_min: f64,
    pub cv_max: f64,
    pub cv_bin: f64,
    pub num_bin: usize,
    pub num_full: f64,
    pub accum_num: Vec<f64>,
    pub accum_force: Vec<f64>,
    pub delta_energy: f64,
}










impl BiasingForce1D
{
    /// Construct an initialized BiasingForce1D for eABF-ATF free energy calculation.
    ///
    /// # Parameters
    /// ```
    /// para: the input parameters for construction
    /// cv_min: minimum CV value in eABF-ATF (Unit: bohr)
    /// cv_max: maximum CV value in eABF-ATF (Unit: bohr)
    /// ```
    ///
    /// # Examples
    /// ```
    /// let biasing_force_1d: BiasingForce1D = BiasingForce1D::from_range(&para, cv_min, cv_max);
    /// ```
    pub fn from_range(para: &Para, cv_min: f64, cv_max: f64) -> BiasingForce1D
    {
        // Read in the needed parameters from para
        let mut cv_bin: f64 = para.md_para.eabf_atf_para.cv_bin * ANGSTROM_TO_BOHR;                // reference CV bin in eABF-ATF (Unit: bohr)
        let num_half_full: usize = para.md_para.eabf_atf_para.num_half_full;                // half of number of full sample for the biasing force

        // Get the number of CV bin and reset the CV bin
        let num_bin: f64 = ((cv_max - cv_min) / cv_bin).round();                // Get the number of CV bin according to the CV range
        cv_bin = (cv_max - cv_min) / num_bin;                // Reset the CV bin
        let num_bin: usize = num_bin as usize;

        // Calculate the basic unit of thermoleveling force
        let delta_force: f64 = para.md_para.eabf_atf_para.gamma * BOLTZMANN * para.md_para.thermostat.temp_bath * JOULE_TO_HARTREE / cv_bin;

        // Calculate the elastic coefficient of the harmonic potential of the boundary (A.U.)
        let k: Option<f64> = match para.md_para.extended_md_para.boundary
        {
            Some(multiplier) => Some( multiplier * (BOLTZMANN * para.md_para.thermostat.temp_bath * JOULE_TO_HARTREE) / (para.md_para.extended_md_para.sigma * ANGSTROM_TO_BOHR).powi(2) ),
            None => None,
        };

        BiasingForce1D
        {
            cv_min,
            cv_max,
            cv_bin,
            num_bin,
            num_half_full: num_half_full as f64,
            num_full: 2.0 * num_half_full as f64,
            accum_num: vec![0.0; num_bin],
            accum_force: vec![0.0; num_bin],
            delta_force,
            k,
        }
    }





    /// Accumulate a new biasing force to BiasingForce1D
    ///
    /// # Parameters
    /// ```
    /// lambda: the value of the extended degree of freedom for the biasing force (Unit: bohr)
    /// force: the value of the biasing force (Unit: A.U.)
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// biasing_force_1d.accumulate_biasing_force(lambda, force);
    /// ```
    fn accumulate_biasing_force(&mut self, lambda: f64, force: f64)
    {
        // If the input lambda is within the CV range, accumulate the force; otherwise, ignore the force
        if (lambda > self.cv_min) && (lambda < self.cv_max)
        {
            let index: usize = ( (lambda - self.cv_min) / self.cv_bin ).floor() as usize;                // Get the index of the input lambda
            self.accum_num[index] += 1.0;
            self.accum_force[index] += force;
        }
    }





    /// Based on the BiasingForce1D, get the current biasing force for a given lambda
    ///
    /// # Parameters
    /// ```
    /// lambda: the given value of the extended degree of freedom (Unit: bohr)
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// let biasing_force: f64 = biasing_force_1d.get_biasing_force(lambda);
    /// ```
    fn get_biasing_force(&self, lambda: f64) -> f64
    {
        // If the input lambda is within the CV range, calculate the force
        if (lambda > self.cv_min) && (lambda < self.cv_max)
        {
            let index: usize = ( (lambda - self.cv_min) / self.cv_bin ).floor() as usize;                // Get the index of the input lambda
            
            // If num <= num_full/2, return 0
            if self.accum_num[index] <= self.num_half_full
            {
                0.0
            }
            // If num >= num_full, return the average force
            else if self.accum_num[index] >= self.num_full
            {
                self.accum_force[index] / self.accum_num[index]
            }
            // If num_full/2 < num < num_full, return the scaled force
            else
            {
                (self.accum_force[index] / self.accum_num[index]) * (self.accum_num[index] / self.num_half_full - 1.0)
            }
        }
        // Otherwise, return the boundary force
        else
        {
            match self.k
            {
                // If there is a boundary, return the boundary force
                Some(k) =>
                {
                    if lambda < (self.cv_min + 1e-6)
                    {
                        k * (self.cv_min - lambda)
                    }
                    else
                    {
                        k * (self.cv_max - lambda)
                    }
                },

                // Otherwise, return 0
                None => 0.0,
            }
        }
    }





    /// Based on the BiasingForce1D, return the whole biasing potential
    ///
    /// # Parameters
    /// ```
    /// biasing_pot: the whole biasing potential
    /// ```
    ///
    /// # Examples
    /// ```
    /// let biasing_pot: Vec<f64> = biasing_force_1d.get_biasing_pot();
    /// ```
    fn get_biasing_pot(&self) -> Vec<f64>
    {
        let mut force: Vec<f64> = Vec::with_capacity(self.num_bin);
        for i in 0..self.num_bin
        {
            // If num <= num_full/2, return the initial force
            if self.accum_num[i] <= self.num_half_full
            {
                force.push(0.0);
            }
            // If num >= num_full, return the average force
            else if self.accum_num[i] >= self.num_full
            {
                force.push(self.accum_force[i] / self.accum_num[i]);
            }
            // If num_full/2 < num < num_full, return the scaled force
            else
            {
                force.push( (self.accum_force[i] / self.accum_num[i]) * (self.accum_num[i] / self.num_half_full - 1.0) );
            }
        }

        let mut biasing_pot: Vec<f64> = Vec::with_capacity(self.num_bin);                // Locate at center of each bin
        biasing_pot.push(0.0);
        for i in 0..(self.num_bin-1)
        {
            biasing_pot.push( biasing_pot[i] - (force[i] + force[i+1]) * self.cv_bin / 2.0 );
        }

        biasing_pot
    }





    /// Based on the BiasingForce1D, get the current thermoleveling force for a given lambda
    ///
    /// # Parameters
    /// ```
    /// lambda: the given value of the extended degree of freedom (Unit: bohr)
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// let thermoleveling_force: f64 = biasing_force_1d.get_thermoleveling_force(lambda);
    /// ```
    fn get_thermoleveling_force(&self, lambda: f64) -> f64
    {
        // If the input lambda is within the CV range, calculate the force
        if (lambda > self.cv_min) && (lambda < self.cv_max)
        {
            let index: usize = ( (lambda - self.cv_min) / self.cv_bin ).floor() as usize;                // Get the index of the input lambda
            
            // For the first bin
            if index == 0
            {
                // If the accumulated number is smallest, return 0
                if self.accum_num[index] < self.accum_num[index+1]
                {
                    0.0
                }
                else
                {
                    self.delta_force * ( (self.accum_num[index] + self.num_full) / (self.accum_num[index+1] + self.num_full) ).ln()
                }
            }

            // For the last bin
            else if index == self.num_bin - 1
            {
                // If the accumulated number is smallest, return 0
                if self.accum_num[index] < self.accum_num[index-1]
                {
                    0.0
                }
                else
                {
                    self.delta_force * ( (self.accum_num[index-1] + self.num_full) / (self.accum_num[index] + self.num_full) ).ln()
                }
            }

            // For the in-between bin
            else
            {
                // If the accumulated number is smallest, return 0
                if (self.accum_num[index] < self.accum_num[index-1]) && (self.accum_num[index] < self.accum_num[index+1])
                {
                    0.0
                }
                // If the accumulated number is largest, return the larger side
                else if (self.accum_num[index] > self.accum_num[index-1]) && (self.accum_num[index] > self.accum_num[index+1])
                {
                    if self.accum_num[index-1] < self.accum_num[index+1]
                    {
                        self.delta_force * ( (self.accum_num[index-1] + self.num_full) / (self.accum_num[index] + self.num_full) ).ln()
                    }
                    else
                    {
                        self.delta_force * ( (self.accum_num[index] + self.num_full) / (self.accum_num[index+1] + self.num_full) ).ln()
                    }
                }
                // If the accumulated number is in middle, return the larger side
                else
                {
                    let force1: f64 = self.delta_force * ( (self.accum_num[index-1] + self.num_full) / (self.accum_num[index] + self.num_full) ).ln();
                    let force2: f64 = self.delta_force * ( (self.accum_num[index] + self.num_full) / (self.accum_num[index+1] + self.num_full) ).ln();
                    if force1.abs() > force2.abs()
                    {
                        force1
                    }
                    else
                    {
                        force2
                    }
                }
            }
        }


        // Otherwise, return 0
        else
        {
            0.0
        }
    }
}










impl MeanForce1D
{
    /// Construct an initialized MeanForce1D for eABF-ATF free energy calculation.
    ///
    /// # Parameters
    /// ```
    /// biasing_force_1d: the input reference biasing_force_1d for construction
    /// temp_bath: temperature of the bath (Unit: K)
    /// ```
    ///
    /// # Examples
    /// ```
    /// let mean_force_1d: MeanForce1D = MeanForce1D::from_biasing_force(&biasing_force_1d, temp_bath);
    /// ```
    fn from_biasing_force(biasing_force_1d: &BiasingForce1D, temp_bath: f64) -> MeanForce1D
    {
        MeanForce1D
        {
            cv_min: biasing_force_1d.cv_min,
            cv_max: biasing_force_1d.cv_max,
            cv_bin: biasing_force_1d.cv_bin,
            num_bin: biasing_force_1d.num_bin,
            num_full: biasing_force_1d.num_full,
            accum_num: vec![0.0; biasing_force_1d.num_bin],
            accum_force: vec![0.0; biasing_force_1d.num_bin],
            delta_energy: BOLTZMANN * temp_bath * JOULE_TO_HARTREE,
        }
    }





    /// Accumulate a new sampling force to MeanForce1D
    ///
    /// # Parameters
    /// ```
    /// xi: the value of the real reaction coordinate for the sampling force (Unit: bohr)
    /// force: the value of the sampling force (Unit: A.U.)
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// mean_force_1d.accumulate_sampling_force(xi, force);
    /// ```
    fn accumulate_sampling_force(&mut self, xi: f64, force: f64)
    {
        // If the input xi is within the CV range, accumulate the force; otherwise, ignore the force
        if (xi > self.cv_min) && (xi < self.cv_max)
        {
            let index: usize = ( (xi - self.cv_min) / self.cv_bin ).floor() as usize;                // Get the index of the input xi
            self.accum_num[index] += 1.0;
            self.accum_force[index] += force;
        }
    }





    /// Based on the MeanForce1D, calculate the free energy along CV
    ///
    /// # Parameters
    /// ```
    /// free_energy: the output free energy along CV
    /// ```
    ///
    /// # Examples
    /// ```
    /// let free_energy: Vec<f64> = mean_force_1d.get_free_energy();
    /// ```
    fn get_free_energy(&self) -> Vec<f64>
    {
        let mut force: Vec<f64> = Vec::with_capacity(self.num_bin);
        for i in 0..self.num_bin
        {
            // For full sampling bins
            if self.accum_num[i] > self.num_full
            {
                force.push(self.accum_force[i] / self.accum_num[i]);
            }
            // Otherwise
            else
            {
                force.push(0.0);
            }
        }

        let mut free_energy: Vec<f64> = Vec::with_capacity(self.num_bin);                // Locate at center of each bin
        free_energy.push(0.0);
        for i in 0..(self.num_bin-1)
        {
            if self.accum_num[i] > self.num_full && self.accum_num[i+1] > self.num_full
            {
                free_energy.push( free_energy[i] - (force[i] + force[i+1]) * self.cv_bin / 2.0 - self.delta_energy * (self.accum_num[i+1] / self.accum_num[i]).ln() );
            }
            else
            {
                free_energy.push( free_energy[i] );
            }
        }

        free_energy
    }
}










pub fn eabf_atf_1d<C: Communicator, P: PES, CV: ColVar>(comm: &C, real_pes: &P, cv: &CV, para: &Para, output_path: &PathBuf)
{
    // Specify the output files
    let str_output_file: PathBuf = output_path.join("eabf_atf.pdb");
    let eabf_atf_output_file: PathBuf = output_path.join("eabf_atf.out");
    let bias_pot_output_file: PathBuf = output_path.join("biasing_potential.out");
    let free_energy_output_file: PathBuf = output_path.join("free_energy.out");

    // Parallel processing
    let rank = comm.rank();
    let root_process = comm.process_at_rank(ROOT_RANK);



    // Define the variables for free energy calculation
    let mut s: System = cv.get_ini_str();

    let dt_au_half: f64 = 0.5 * para.md_para.dt * FEMTOSECOND_TO_AU;                // A.U.
    let mut t: f64 = 0.0;                // fs

    let n_free: usize = match s.cell
    {
        Some(_) => 3 * s.natom - 3,                // For NVT periodic system, non-conservation of angular momentum
        None => 3 * s.natom - 3,                // For NVT non-periodic system, doesn't remove angular momentum drift explicitly
    };

    let mut kin: f64;                // Current kinetic energy (Hartree)
    let mut temp: f64;                      // Current temperature (K)

    let mut atom_mass: Vec<f64> = Vec::with_capacity(s.natom);
    let mut global_mass: f64 = 0.0;
    let mut global_vel: Array1<f64>;
    for i in 0..s.natom
    {
        atom_mass.push( s.atom_type.as_ref().expect(&error_none_value("s.atom_type"))[i].get_atomic_mass() );
        global_mass += atom_mass[i];
    }

    // Define the parameters for extended dynamics
    let k: f64 = (BOLTZMANN * para.md_para.thermostat.temp_bath * JOULE_TO_HARTREE) / (para.md_para.extended_md_para.sigma * ANGSTROM_TO_BOHR).powi(2);    // Elastic coefficient (A.U.)
    let lambda_mass: f64 = (para.md_para.extended_md_para.tau * FEMTOSECOND_TO_AU).powi(2) * k / (4.0 * PI * PI);                // Mass of the extended freedom of degree (A.U.)

    // Define the parameters for Langevin dynamics
    let atten_factor: f64 = (-para.md_para.thermostat.gamma * para.md_para.dt).exp();                // Attenuation factor of velocity
    let a: f64 = BOLTZMANN * para.md_para.thermostat.temp_bath * JOULE_TO_HARTREE * (1.0 - atten_factor * atten_factor);        // Hartree
    let mut noise_strength: Array2<f64> = Array2::zeros(s.coord.raw_dim());                // Noise Strength (A.U. of velocity)
    for i in 0..s.natom
    {
        noise_strength[[i, 0]] = (a / atom_mass[i]).sqrt();
        noise_strength[[i, 1]] = (a / atom_mass[i]).sqrt();
        noise_strength[[i, 2]] = (a / atom_mass[i]).sqrt();
    }
    let noise_strength_lambda: f64 = (a / lambda_mass).sqrt();                // Noise Strength for the extened freedom of degree (A.U. of velocity)
    let mut noice: Array2<f64>;                // Current Gaussian stochastic noice for each dimension
    let mut noice_lambda: f64;                // Current Gaussian stochastic noice for the extened freedom of degree

    // Define the structure for eABF-ATF
    let mut biasing_force_1d: BiasingForce1D = cv.initialize_biasing_force(&para);
    let mut mean_force_1d: MeanForce1D = MeanForce1D::from_biasing_force(&biasing_force_1d, para.md_para.thermostat.temp_bath);



    // Initialization
    // Define and initialize the reaction coordinate and extended freedom of degree
    let (mut xi, mut grads): (f64, Array2<f64>) = cv.get_cv_grads(&s);                // Reaction coordinate (bohr) and its gradients
    let mut lambda: f64 = xi;                // Extended freedom of degree (bohr)
    let mut grads_norm: f64 = (&grads * &grads).sum().sqrt();                // Norm of the gradients

    // Calculate the real potential energy and atomic forces for the initial structure
    let (mut pot_real, mut force_real): (f64, Array2<f64>) = real_pes.get_energy_force(&s);
    s.pot = pot_real;
    let mut f_real: f64 = (&force_real * &force_real).sum().sqrt();

    // Calculate the elastic forces in extended dynamics
    let mut force_elastic_lambda: f64 = k * (xi - lambda);                // Elastic force on the extended freedom of degree (A.U.)
    let mut force_elastic: Array2<f64> = -force_elastic_lambda * &grads;                // Elastic force on system (A.U.)
    let mut f_elastic: f64 = (&force_elastic * &force_elastic).sum().sqrt();                // Norm of elastic force on system (A.U.)

    // Define the biasing force in eABF-ATF
    let mut force_biasing_lambda: f64;                // Biasing force on the extended freedom of degree (A.U.)

    // Define the thermoleveling force in eABF-ATF
    let mut force_thermoleveling_lambda: f64 = 0.0;                // Thermoleveling force on the extended freedom of degree (A.U.)

    // The system is driven by the real forces and the elastic forces
    let mut force_total: Array2<f64> = &force_real + &force_elastic;
    let mut force_total_lambda: f64;

    // Obtain the initial acceleration
    let mut acc: Array2<f64> = Array2::zeros(s.coord.raw_dim());
    for i in 0..s.natom
    {
        acc[[i, 0]] = force_total[[i, 0]] / atom_mass[i];
        acc[[i, 1]] = force_total[[i, 1]] / atom_mass[i];
        acc[[i, 2]] = force_total[[i, 2]] / atom_mass[i];
    }
    let mut acc_lambda: f64 = force_elastic_lambda / lambda_mass;                // A.U.

    // Obtain the initial velocity according to the Maxwell-Boltzmann distribution
    let mut vel: Array2<f64> = Array2::zeros(s.coord.raw_dim());
    let mut vel_lambda: f64 = 0.0;

    // Calculate the initial kinetic energy and temperature
    kin = 0.0;
    for i in 0..s.natom
    {
        kin += atom_mass[i] * ( vel[[i, 0]].powi(2) + vel[[i, 1]].powi(2) + vel[[i, 2]].powi(2) );          // Hartree
    }
    temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * n_free as f64 );               // K
    kin *= 0.5;                // Hartree



    // Output the initial structure to PDB output file, and output the header to extended dynamics output file
    if rank == ROOT_RANK
    {
        s.write_pdb(&str_output_file, true, 0);

        let mut eabf_atf_output = File::create(&eabf_atf_output_file).expect(&error_file("creating", &eabf_atf_output_file));
        eabf_atf_output.write_all(b"    step            time            temp             kin        pot_real          f_real       f_elastic   f_thermolevel      grads_norm              xi          lambda\n").expect(&error_file("writing", &eabf_atf_output_file));
        eabf_atf_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", 0, t, temp, kin, pot_real, f_real, f_elastic, force_thermoleveling_lambda, grads_norm, xi*BOHR_TO_ANGSTROM, lambda*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &eabf_atf_output_file));

        let mut bias_pot_output = File::create(&bias_pot_output_file).expect(&error_file("creating", &bias_pot_output_file));
        for i in 0..biasing_force_1d.num_bin
        {
            bias_pot_output.write_all(format!("{:9.5} ", ( biasing_force_1d.cv_min + biasing_force_1d.cv_bin * (i as f64 + 0.5) ) * BOHR_TO_ANGSTROM ).as_bytes()).expect(&error_file("writing", &bias_pot_output_file));
        }
        bias_pot_output.write_all(format!("\n\n").as_bytes()).expect(&error_file("writing", &bias_pot_output_file));

        let mut free_energy_output = File::create(&free_energy_output_file).expect(&error_file("creating", &free_energy_output_file));
        for i in 0..mean_force_1d.num_bin
        {
            free_energy_output.write_all(format!("{:9.5} ", ( mean_force_1d.cv_min + mean_force_1d.cv_bin * (i as f64 + 0.5) ) * BOHR_TO_ANGSTROM ).as_bytes()).expect(&error_file("writing", &free_energy_output_file));
        }
        free_energy_output.write_all(format!("\n\n").as_bytes()).expect(&error_file("writing", &free_energy_output_file));
    }





    // Perform the initial Langevin NVT MD iteractively to equilibrium (without biasing force on the extended freedom of degree)
    for n in 1..(para.md_para.equi_step+1)
    {
        // Update the next 1/2 step velocity based on the BAOAB algorithm
        t += para.md_para.dt;                    // fs
        vel += &(dt_au_half * &acc);                // A.U.
        vel_lambda += dt_au_half * acc_lambda;                // A.U.

        // Update the next 1/2 step position based on the BAOAB algorithm
        s.coord += &(dt_au_half * &vel);                // A.U.
        lambda += dt_au_half * vel_lambda;                // A.U.

        // Calculate the impacts of frictional drag and stochastic collisions on the velocity
        noice = Array2::random(s.coord.raw_dim(), Normal::new(0.0, 1.0).expect(&error_none_value("noice")));                // Gaussian stochastic noice with mean value of 0 and variance of 1
        noice_lambda = Array1::random(1, Normal::new(0.0, 1.0).expect(&error_none_value("noice_lambda")))[0];
        let noice_p: &mut [f64] = noice.as_slice_mut().expect(&error_as_slice("noice"));
        root_process.broadcast_into(noice_p);                // Broadcast to all ranks
        root_process.broadcast_into(&mut noice_lambda);                // Broadcast to all ranks
        vel = atten_factor * &vel + &noise_strength * &noice;
        vel_lambda = atten_factor * vel_lambda + noise_strength_lambda * noice_lambda;

        // Update the next step position based on the BAOAB algorithm
        s.coord += &(dt_au_half * &vel);                // A.U.
        lambda += dt_au_half * vel_lambda;                // A.U.
        (xi, grads) = cv.get_cv_grads(&s);                // Reaction coordinate (bohr) and its gradients
        grads_norm = (&grads * &grads).sum().sqrt();

        // Update the real potential energy and atomic forces for the next structure
        (pot_real, force_real) = real_pes.get_energy_force(&s);
        s.pot = pot_real;
        f_real = (&force_real * &force_real).sum().sqrt();

        // Update the elastic forces in extended dynamics
        force_elastic_lambda = k * (xi - lambda);                // Elastic force on the extended freedom of degree (A.U.)
        force_elastic = -force_elastic_lambda * &grads;                // Elastic force on system (A.U.)
        f_elastic = (&force_elastic * &force_elastic).sum().sqrt();                // Norm of elastic force on system (A.U.)

        // The system is driven by the real forces and the elastic forces
        force_total = &force_real + &force_elastic;

        // Update the next acceleration
        for i in 0..s.natom
        {
            acc[[i, 0]] = force_total[[i, 0]] / atom_mass[i];
            acc[[i, 1]] = force_total[[i, 1]] / atom_mass[i];
            acc[[i, 2]] = force_total[[i, 2]] / atom_mass[i];
        }
        acc_lambda = force_elastic_lambda / lambda_mass;                // A.U.

        // Update the next step velocity based on the BAOAB algorithm
        vel += &(dt_au_half * &acc);                // A.U.
        vel_lambda += dt_au_half * acc_lambda;                // A.U.



        // Remove the overall translational speed of the mass center
        if (n % para.md_para.remove_tran_step) == 0
        {
            global_vel = Array1::zeros(3);
            for i in 0..s.natom
            {
                global_vel += &(atom_mass[i] * &vel.slice(s![i, ..]));
            }
            global_vel /= global_mass;             // Global velocity
            for i in 0..s.natom
            {
                vel[[i, 0]] -= global_vel[0];
                vel[[i, 1]] -= global_vel[1];
                vel[[i, 2]] -= global_vel[2];
            }
        }



        // Calculate the kinetic energy and temperature
        kin = 0.0;
        for i in 0..s.natom
        {
            kin += atom_mass[i] * ( vel[[i, 0]].powi(2) + vel[[i, 1]].powi(2) + vel[[i, 2]].powi(2) );          // Hartree
        }
        temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * n_free as f64 );               // K
        kin *= 0.5;                // Hartree



        // Output the information in this iterative step
        if rank == ROOT_RANK
        {
            if (n % para.md_para.print_step) == 0
            {
                s.write_pdb(&str_output_file, false, n);
            }
            let mut eabf_atf_output = File::options().append(true).open(&eabf_atf_output_file).expect(&error_file("opening", &eabf_atf_output_file));
            eabf_atf_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, f_real, f_elastic, force_thermoleveling_lambda, grads_norm, xi*BOHR_TO_ANGSTROM, lambda*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &eabf_atf_output_file));
        }
    }





    // Perform the Langevin NVT MD iteractively with accumulated biasing force on the extended freedom of degree
    t = 0.0;
    for n in 1..(para.md_para.max_step+1)
    {
        // Update the next 1/2 step velocity based on the BAOAB algorithm
        t += para.md_para.dt;                    // fs
        vel += &(dt_au_half * &acc);                // A.U.
        vel_lambda += dt_au_half * acc_lambda;                // A.U.

        // Update the next 1/2 step position based on the BAOAB algorithm
        s.coord += &(dt_au_half * &vel);                // A.U.
        lambda += dt_au_half * vel_lambda;                // A.U.

        // Calculate the impacts of frictional drag and stochastic collisions on the velocity
        noice = Array2::random(s.coord.raw_dim(), Normal::new(0.0, 1.0).expect(&error_none_value("noice")));                // Gaussian stochastic noice with mean value of 0 and variance of 1
        noice_lambda = Array1::random(1, Normal::new(0.0, 1.0).expect(&error_none_value("noice_lambda")))[0];
        let noice_p: &mut [f64] = noice.as_slice_mut().expect(&error_as_slice("noice"));
        root_process.broadcast_into(noice_p);                // Broadcast to all ranks
        root_process.broadcast_into(&mut noice_lambda);                // Broadcast to all ranks
        vel = atten_factor * &vel + &noise_strength * &noice;
        vel_lambda = atten_factor * vel_lambda + noise_strength_lambda * noice_lambda;

        // Update the next step position based on the BAOAB algorithm
        s.coord += &(dt_au_half * &vel);                // A.U.
        lambda += dt_au_half * vel_lambda;                // A.U.
        (xi, grads) = cv.get_cv_grads(&s);                // Reaction coordinate (bohr) and its gradients
        grads_norm = (&grads * &grads).sum().sqrt();

        // Update the real potential energy and atomic forces for the next structure
        (pot_real, force_real) = real_pes.get_energy_force(&s);
        s.pot = pot_real;
        f_real = (&force_real * &force_real).sum().sqrt();

        // Update the elastic forces in extended dynamics
        force_elastic_lambda = k * (xi - lambda);                // Elastic force on the extended freedom of degree (A.U.)
        force_elastic = -force_elastic_lambda * &grads;                // Elastic force on system (A.U.)
        f_elastic = (&force_elastic * &force_elastic).sum().sqrt();                // Norm of elastic force on system (A.U.)

        // Update the biasing force in eABF-ATF
        force_biasing_lambda = biasing_force_1d.get_biasing_force(lambda);                // Biasing force on the extended degree of freedom (A.U.)

        // Update the thermoleveling force in eABF-ATF
        force_thermoleveling_lambda = biasing_force_1d.get_thermoleveling_force(lambda);                // Thermoleveling force on the extended degree of freedom (A.U.)

        // The system is driven by the real forces and the elastic forces
        // The extended degree of freedom is driven by the elastic force, biasing force, and thermoleveling force
        force_total = &force_real + &force_elastic;
        force_total_lambda = force_elastic_lambda + force_biasing_lambda + force_thermoleveling_lambda;

        // Update the next acceleration
        for i in 0..s.natom
        {
            acc[[i, 0]] = force_total[[i, 0]] / atom_mass[i];
            acc[[i, 1]] = force_total[[i, 1]] / atom_mass[i];
            acc[[i, 2]] = force_total[[i, 2]] / atom_mass[i];
        }
        acc_lambda = force_total_lambda / lambda_mass;                // A.U.

        // Update the next step velocity based on the BAOAB algorithm
        vel += &(dt_au_half * &acc);                // A.U.
        vel_lambda += dt_au_half * acc_lambda;                // A.U.



        // Remove the overall translational speed of the mass center
        if (n % para.md_para.remove_tran_step) == 0
        {
            global_vel = Array1::zeros(3);
            for i in 0..s.natom
            {
                global_vel += &(atom_mass[i] * &vel.slice(s![i, ..]));
            }
            global_vel /= global_mass;             // Global velocity
            for i in 0..s.natom
            {
                vel[[i, 0]] -= global_vel[0];
                vel[[i, 1]] -= global_vel[1];
                vel[[i, 2]] -= global_vel[2];
            }
        }



        // Calculate the kinetic energy and temperature
        kin = 0.0;
        for i in 0..s.natom
        {
            kin += atom_mass[i] * ( vel[[i, 0]].powi(2) + vel[[i, 1]].powi(2) + vel[[i, 2]].powi(2) );          // Hartree
        }
        temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * n_free as f64 );               // K
        kin *= 0.5;                // Hartree



        // Accumulate the current biasing force
        biasing_force_1d.accumulate_biasing_force(lambda, -force_elastic_lambda);
        mean_force_1d.accumulate_sampling_force(xi, force_elastic_lambda);



        // Output the information in this iterative step
        if rank == ROOT_RANK
        {
            if (n % para.md_para.print_step) == 0
            {
                s.write_pdb(&str_output_file, false, n);

                let mut bias_pot_output = File::options().append(true).open(&bias_pot_output_file).expect(&error_file("opening", &bias_pot_output_file));
                for i in 0..biasing_force_1d.num_bin
                {
                    bias_pot_output.write_all(format!("{:9.5} ", biasing_force_1d.accum_num[i]).as_bytes()).expect(&error_file("writing", &bias_pot_output_file));
                }
                bias_pot_output.write_all(format!("\n").as_bytes()).expect(&error_file("writing", &bias_pot_output_file));
                let biasing_pot: Vec<f64> = biasing_force_1d.get_biasing_pot();
                for i in 0..biasing_pot.len()
                {
                    bias_pot_output.write_all(format!("{:9.5} ", biasing_pot[i]).as_bytes()).expect(&error_file("writing", &bias_pot_output_file));
                }
                bias_pot_output.write_all(format!("\n\n").as_bytes()).expect(&error_file("writing", &bias_pot_output_file));

                let mut free_energy_output = File::options().append(true).open(&free_energy_output_file).expect(&error_file("opening", &free_energy_output_file));
                for i in 0..mean_force_1d.num_bin
                {
                    free_energy_output.write_all(format!("{:9.5} ", mean_force_1d.accum_num[i]).as_bytes()).expect(&error_file("writing", &free_energy_output_file));
                }
                free_energy_output.write_all(format!("\n").as_bytes()).expect(&error_file("writing", &free_energy_output_file));
                let free_energy: Vec<f64> = mean_force_1d.get_free_energy();
                for i in 0..free_energy.len()
                {
                    free_energy_output.write_all(format!("{:9.5} ", free_energy[i]).as_bytes()).expect(&error_file("writing", &free_energy_output_file));
                }
                free_energy_output.write_all(format!("\n\n").as_bytes()).expect(&error_file("writing", &free_energy_output_file));
            }

            let mut eabf_atf_output = File::options().append(true).open(&eabf_atf_output_file).expect(&error_file("opening", &eabf_atf_output_file));
            eabf_atf_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, f_real, f_elastic, force_thermoleveling_lambda, grads_norm, xi*BOHR_TO_ANGSTROM, lambda*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &eabf_atf_output_file));
        }
    }
}










