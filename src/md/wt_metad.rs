use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::fmt::Debug;
use crate::common::constants::*;
use crate::common::error::*;
use crate::common::traits::*;
use crate::io::input::Para;
use crate::pes_exploration::system::System;
use mpi::traits::Communicator;
use mpi::collective::Root;
use mpi::collective::CommunicatorCollectives;
use ndarray::{Array1, Array2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;










/// 1D Gaussian functions for well-tempered metadynamics
///
/// # Fields
/// ```
/// height: heights of the Gaussians (Unit: Hartree)
/// visited_cv: visited CV values, as the centers of the Gaussians (Unit: bohr)
/// two_sigma_squared: 2 * sigma * sigma, with sigma being the standard deviation of the Gaussians (invariant, Unit: bohr)
/// comm: the communicator for MPI calculations
/// ```
pub struct Gaussians1D<'a, C: Communicator>
{
    pub height: Vec<f64>,
    pub visited_cv: Vec<f64>,
    pub two_sigma_squared: f64,
    pub comm: &'a C,
}










impl<'a, C: Communicator> Gaussians1D<'a, C>
{
    /// Construct an empty Gaussians1D to return
    ///
    /// # Parameters
    /// ```
    /// n_max: the inferred maximum number of Gaussian potentials
    /// two_sigma_squared: 2 * sigma * sigma, with sigma being the standard deviation of all the Gaussians (Unit: bohr)
    /// comm: the communicator for MPI calculations
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// let mut gaussians_1d = Gaussians1D::new(n_max, two_sigma_squared, &comm);
    /// ```
    fn new(n_max: usize, two_sigma_squared: f64, comm: &'a C) -> Self
    {
        Gaussians1D
        {
            height: Vec::with_capacity(n_max),
            visited_cv: Vec::with_capacity(n_max),
            two_sigma_squared,
            comm,
        }
    }





    /// Write out all the Gaussians
    ///
    /// # Parameters
    /// ```
    /// filename: the output filename
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// gaussians_1d.write("Gaussians.out");
    /// ```
    pub fn write<P: AsRef<Path> + Debug>(&self, filename: P)
    {
        let sigma: f64 = (self.two_sigma_squared / 2.0).sqrt() * BOHR_TO_ANGSTROM;                // standard deviation of all the Gaussians (Unit: Å)
        let mut output = File::create(&filename).expect(&error_file("creating", &filename));
        output.write_all(b"         height      visited_cv           sigma\n").expect(&error_file("writing", &filename));
        for i in 0..self.visited_cv.len()
        {
            output.write_all(format!("{:15.8} {:15.8} {:15.8}\n", self.height[i], self.visited_cv[i]*BOHR_TO_ANGSTROM, sigma).as_bytes()).expect(&error_file("writing", &filename));
        }
    }





    /// Add a new Gaussian potential to Gaussians1D
    ///
    /// # Parameters
    /// ```
    /// height: height of the new Gaussian potential (Unit: Hartree)
    /// cv: center of the new Gaussian potential (Unit: bohr)
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// gaussians_1d.add_pot(height, cv);
    /// ```
    fn add_pot(&mut self, height: f64, cv: f64)
    {
        self.height.push(height);
        self.visited_cv.push(cv);
    }





    /// Input a value of reaction coordinate, output its Gaussian potential based on the visited list
    ///
    /// # Parameters
    /// ```
    /// xi: the input value of reaction coordinate (Unit: bohr)
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// let pot: f64 = gaussians_1d.get_pot(xi);
    /// ```
    fn get_pot(&self, xi: f64) -> f64
    {
        // Parallel processing
        let size: usize = self.comm.size().try_into().expect(&error_type_transformation("size", "i32", "usize"));
        let rank: usize = self.comm.rank().try_into().expect(&error_type_transformation("rank", "i32", "usize"));

        // Distribute the visited CVs on each process, and calculate the Gaussian potentials respectively
        let mut pot_rank: f64 = 0.0;
        for i in 0..self.visited_cv.len()
        {
            if i % size == rank
            {
                pot_rank += self.height[i] * ( -(xi - self.visited_cv[i]).powi(2) / self.two_sigma_squared ).exp();
            }
        }

        // Gather the Gaussian potentials on all the ranks
        let mut pot_gather: Vec<f64> = vec![0.0; size];
        self.comm.all_gather_into(&pot_rank, &mut pot_gather[..]);

        pot_gather.into_iter().sum()
    }





    /// Input a value of reaction coordinate, output its derivative of Gaussian potential based on the visited list
    ///
    /// # Parameters
    /// ```
    /// xi: the input value of reaction coordinate (Unit: bohr)
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// let deriv: f64 = gaussians_1d.get_deriv(xi);
    /// ```
    fn get_deriv(&self, xi: f64) -> f64
    {
        // Parallel processing
        let size: usize = self.comm.size().try_into().expect(&error_type_transformation("size", "i32", "usize"));
        let rank: usize = self.comm.rank().try_into().expect(&error_type_transformation("rank", "i32", "usize"));

        // Distribute the visited CVs on each process, and calculate the derivatives of Gaussian potentials respectively
        let mut dist: f64;
        let mut deriv_rank: f64 = 0.0;
        for i in 0..self.visited_cv.len()
        {
            if i % size == rank
            {
                dist = xi - self.visited_cv[i];
                deriv_rank += (-2.0 * self.height[i] * dist / self.two_sigma_squared) * (-dist * dist / self.two_sigma_squared).exp();
            }
        }

        // Gather the derivatives of Gaussian potentials on all the ranks
        let mut deriv_gather: Vec<f64> = vec![0.0; size];
        self.comm.all_gather_into(&deriv_rank, &mut deriv_gather[..]);

        deriv_gather.into_iter().sum()
    }
}










pub fn wt_metad_1d<C: Communicator, P: PES, CV: ColVar>(comm: &C, real_pes: &P, cv: &CV, para: &Para, output_path: &PathBuf)
{
    // Specify the output files
    let str_output_file: PathBuf = output_path.join("wtm.pdb");
    let wtm_output_file: PathBuf = output_path.join("wtm.out");
    let gau_output_file: PathBuf = output_path.join("Gaussians.out");

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
    let mut noice: Array2<f64>;                // Current Gaussian stochastic noice for each dimension

    // Define the structure for well-tempered metadynamics
    let n_max: usize = para.md_para.max_step / para.md_para.wtm_para.pace_step + 1;
    let two_sigma_squared: f64 = 2.0 * (para.md_para.wtm_para.sigma * ANGSTROM_TO_BOHR).powi(2);                // (Unit: bohr^2)
    let mut gaussians_1d = Gaussians1D::new(n_max, two_sigma_squared, comm);
    let delta_energy: f64 = BOLTZMANN * (para.md_para.wtm_para.bias_factor - 1.0) * para.md_para.thermostat.temp_bath * JOULE_TO_HARTREE;                // kB * delta_T (Unit: Hartree)
    let mut height: f64;



    // Initialization
    // Define and initialize the reaction coordinate
    let (mut xi, mut grads): (f64, Array2<f64>) = cv.get_cv_grads(&s);                // Reaction coordinate (bohr) and its gradients
    let mut grads_norm: f64 = (&grads * &grads).sum().sqrt();                // Norm of the gradients

    // Calculate the real potential energy and atomic forces for the initial structure
    let (mut pot_real, mut force_real): (f64, Array2<f64>) = real_pes.get_energy_force(&s);
    s.pot = pot_real;
    let mut f_real: f64 = (&force_real * &force_real).sum().sqrt();

    // Define the Gaussian forces in WTM
    let mut pot_gaussian: f64;                // Gaussian potential (Hartree)
    let mut force_gaussian: Array2<f64>;                // Gaussian forces on system (A.U.)
    let mut f_gaussian: f64 = 0.0;                // Norm of Gaussian forces on system (A.U.)

    // Define the total forces
    let mut force_total: Array2<f64>;

    // Obtain the initial acceleration
    let mut acc: Array2<f64> = Array2::zeros(s.coord.raw_dim());
    for i in 0..s.natom
    {
        acc[[i, 0]] = force_real[[i, 0]] / atom_mass[i];
        acc[[i, 1]] = force_real[[i, 1]] / atom_mass[i];
        acc[[i, 2]] = force_real[[i, 2]] / atom_mass[i];
    }

    // Obtain the initial velocity according to the Maxwell-Boltzmann distribution
    let mut vel: Array2<f64> = Array2::zeros(s.coord.raw_dim());

    // Calculate the initial kinetic energy and temperature
    kin = 0.0;
    for i in 0..s.natom
    {
        kin += atom_mass[i] * ( vel[[i, 0]].powi(2) + vel[[i, 1]].powi(2) + vel[[i, 2]].powi(2) );          // Hartree
    }
    temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * n_free as f64 );               // K
    kin *= 0.5;                // Hartree



    // Output the initial structure to PDB output file, and output the header to WTM output file
    if rank == ROOT_RANK
    {
        s.write_pdb(&str_output_file, true, 0);
        let mut wtm_output = File::create(&wtm_output_file).expect(&error_file("creating", &wtm_output_file));
        wtm_output.write_all(b"    step            time            temp             kin        pot_real          f_real      f_gaussian      grads_norm              xi\n").expect(&error_file("writing", &wtm_output_file));
        wtm_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", 0, t, temp, kin, pot_real, f_real, f_gaussian, grads_norm, xi*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &wtm_output_file));
        let mut gau_output = File::create(&gau_output_file).expect(&error_file("creating", &gau_output_file));
        gau_output.write_all(b"         height      visited_cv           sigma\n").expect(&error_file("writing", &gau_output_file));
    }





    // Perform the initial Langevin NVT MD iteractively to equilibrium (without biasing Gaussians)
    for n in 1..(para.md_para.equi_step+1)
    {
        // Update the next 1/2 step velocity based on the BAOAB algorithm
        t += para.md_para.dt;                    // fs
        vel += &(dt_au_half * &acc);                // A.U.

        // Update the next 1/2 step position based on the BAOAB algorithm
        s.coord += &(dt_au_half * &vel);                // A.U.

        // Calculate the impacts of frictional drag and stochastic collisions on the velocity
        noice = Array2::random(s.coord.raw_dim(), Normal::new(0.0, 1.0).expect(&error_none_value("noice")));                // Gaussian stochastic noice with mean value of 0 and variance of 1
        let noice_p: &mut [f64] = noice.as_slice_mut().expect(&error_as_slice("noice"));
        root_process.broadcast_into(noice_p);                // Broadcast to all ranks
        vel = atten_factor * &vel + &noise_strength * &noice;

        // Update the next step position based on the BAOAB algorithm
        s.coord += &(dt_au_half * &vel);                // A.U.
        (xi, grads) = cv.get_cv_grads(&s);                // Reaction coordinate (bohr) and its gradients
        grads_norm = (&grads * &grads).sum().sqrt();

        // Update the real potential energy and atomic forces for the next structure
        (pot_real, force_real) = real_pes.get_energy_force(&s);
        s.pot = pot_real;
        f_real = (&force_real * &force_real).sum().sqrt();

        // Update the next acceleration
        for i in 0..s.natom
        {
            acc[[i, 0]] = force_real[[i, 0]] / atom_mass[i];
            acc[[i, 1]] = force_real[[i, 1]] / atom_mass[i];
            acc[[i, 2]] = force_real[[i, 2]] / atom_mass[i];
        }

        // Update the next step velocity based on the BAOAB algorithm
        vel += &(dt_au_half * &acc);                // A.U.



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
            let mut wtm_output = File::options().append(true).open(&wtm_output_file).expect(&error_file("opening", &wtm_output_file));
            wtm_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, f_real, f_gaussian, grads_norm, xi*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &wtm_output_file));
        }
    }





    // Perform the Langevin NVT MD iteractively with accumulated biasing Gaussians
    t = 0.0;
    for n in 1..(para.md_para.max_step+1)
    {
        // Update the next 1/2 step velocity based on the BAOAB algorithm
        t += para.md_para.dt;                    // fs
        vel += &(dt_au_half * &acc);                // A.U.

        // Update the next 1/2 step position based on the BAOAB algorithm
        s.coord += &(dt_au_half * &vel);                // A.U.

        // Calculate the impacts of frictional drag and stochastic collisions on the velocity
        noice = Array2::random(s.coord.raw_dim(), Normal::new(0.0, 1.0).expect(&error_none_value("noice")));                // Gaussian stochastic noice with mean value of 0 and variance of 1
        let noice_p: &mut [f64] = noice.as_slice_mut().expect(&error_as_slice("noice"));
        root_process.broadcast_into(noice_p);                // Broadcast to all ranks
        vel = atten_factor * &vel + &noise_strength * &noice;

        // Update the next step position based on the BAOAB algorithm
        s.coord += &(dt_au_half * &vel);                // A.U.
        (xi, grads) = cv.get_cv_grads(&s);                // Reaction coordinate (bohr) and its gradients
        grads_norm = (&grads * &grads).sum().sqrt();

        // Update the real potential energy and atomic forces for the next structure
        (pot_real, force_real) = real_pes.get_energy_force(&s);
        s.pot = pot_real;
        f_real = (&force_real * &force_real).sum().sqrt();

        // Update the Gaussian forces in WTM
        force_gaussian = -gaussians_1d.get_deriv(xi) * &grads;                // Gaussian forces on system (A.U.)
        f_gaussian = (&force_gaussian * &force_gaussian).sum().sqrt();                // Norm of Gaussian forces on system (A.U.)

        // The system is driven by the real forces and the Gaussian forces
        force_total = &force_real + &force_gaussian;

        // Update the next acceleration
        for i in 0..s.natom
        {
            acc[[i, 0]] = force_total[[i, 0]] / atom_mass[i];
            acc[[i, 1]] = force_total[[i, 1]] / atom_mass[i];
            acc[[i, 2]] = force_total[[i, 2]] / atom_mass[i];
        }

        // Update the next step velocity based on the BAOAB algorithm
        vel += &(dt_au_half * &acc);                // A.U.



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



        // Every pace step, add a Gaussian to the list
        if (n % para.md_para.wtm_para.pace_step) == 0
        {
            pot_gaussian = gaussians_1d.get_pot(xi);
            height = (-pot_gaussian / delta_energy).exp() * para.md_para.wtm_para.height;
            gaussians_1d.add_pot(height, xi);
            if rank == ROOT_RANK
            {
                let mut gau_output = File::options().append(true).open(&gau_output_file).expect(&error_file("opening", &gau_output_file));
                gau_output.write_all(format!("{:15.8} {:15.8} {:15.8}\n", height, xi * BOHR_TO_ANGSTROM, para.md_para.wtm_para.sigma).as_bytes()).expect(&error_file("writing", &gau_output_file));
            }
        }



        // Output the information in this iterative step
        if rank == ROOT_RANK
        {
            if (n % para.md_para.print_step) == 0
            {
                s.write_pdb(&str_output_file, false, n);
            }
            let mut wtm_output = File::options().append(true).open(&wtm_output_file).expect(&error_file("opening", &wtm_output_file));
            wtm_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, f_real, f_gaussian, grads_norm, xi*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &wtm_output_file));
        }
    }
}










pub fn extended_wt_metad_1d<C: Communicator, P: PES, CV: ColVar>(comm: &C, real_pes: &P, cv: &CV, para: &Para, output_path: &PathBuf)
{
    // Specify the output files
    let str_output_file: PathBuf = output_path.join("ewtm.pdb");
    let ewtm_output_file: PathBuf = output_path.join("ewtm.out");
    let gau_output_file: PathBuf = output_path.join("Gaussians.out");

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

    // Define the structure for well-tempered metadynamics
    let n_max: usize = para.md_para.max_step / para.md_para.wtm_para.pace_step + 1;
    let two_sigma_squared: f64 = 2.0 * (para.md_para.wtm_para.sigma * ANGSTROM_TO_BOHR).powi(2);                // (Unit: bohr^2)
    let mut gaussians_1d = Gaussians1D::new(n_max, two_sigma_squared, comm);
    let delta_energy: f64 = BOLTZMANN * (para.md_para.wtm_para.bias_factor - 1.0) * para.md_para.thermostat.temp_bath * JOULE_TO_HARTREE;                // kB * delta_T (Unit: Hartree)
    let mut height: f64;



    // Initialization
    // Define and initialize the reaction coordinate and extended freedom of degree
    let (mut xi, mut grads): (f64, Array2<f64>) = cv.get_cv_grads(&s);                // Reaction coordinate (bohr) and its gradients
    let mut lambda: f64 = xi;                // Extended freedom of degree (bohr)
    let mut grads_norm: f64 = (&grads * &grads).sum().sqrt();                // Norm of the gradients

    // Calculate the real potential energy and atomic forces for the initial structure
    let (mut pot_real, mut force_real): (f64, Array2<f64>) = real_pes.get_energy_force(&s);
    s.pot = pot_real;
    let mut f_real: f64 = (&force_real * &force_real).sum().sqrt();

    // Calculate the elastic forces in extened dynamics
    let mut force_elastic_lambda: f64 = k * (xi - lambda);                // Elastic force on the extended freedom of degree (A.U.)
    let mut force_elastic: Array2<f64> = -force_elastic_lambda * &grads;                // Elastic force on system (A.U.)
    let mut f_elastic: f64 = (&force_elastic * &force_elastic).sum().sqrt();                // Norm of elastic force on system (A.U.)

    // Define the Gaussian forces in WTM
    let mut pot_gaussian_lambda: f64;                // Gaussian potential (Hartree)
    let mut force_gaussian_lambda: f64;                // Gaussian forces on the extended freedom of degree (A.U.)

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



    // Output the initial structure to PDB output file, and output the header to eWTM output file
    if rank == ROOT_RANK
    {
        s.write_pdb(&str_output_file, true, 0);
        let mut ewtm_output = File::create(&ewtm_output_file).expect(&error_file("creating", &ewtm_output_file));
        ewtm_output.write_all(b"    step            time            temp             kin        pot_real          f_real       f_elastic      grads_norm              xi          lambda\n").expect(&error_file("writing", &ewtm_output_file));
        ewtm_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", 0, t, temp, kin, pot_real, f_real, f_elastic, grads_norm, xi*BOHR_TO_ANGSTROM, lambda*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &ewtm_output_file));
        let mut gau_output = File::create(&gau_output_file).expect(&error_file("creating", &gau_output_file));
        gau_output.write_all(b"         height      visited_cv           sigma\n").expect(&error_file("writing", &gau_output_file));
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
            let mut ewtm_output = File::options().append(true).open(&ewtm_output_file).expect(&error_file("opening", &ewtm_output_file));
            ewtm_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, f_real, f_elastic, grads_norm, xi*BOHR_TO_ANGSTROM, lambda*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &ewtm_output_file));
        }
    }





    // Perform the Langevin NVT MD iteractively with accumulated biasing Gaussians on the extended freedom of degree
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

        // Update the Gaussian forces in eWTM
        force_gaussian_lambda = -gaussians_1d.get_deriv(lambda);                // Gaussian forces on the extended degree of freedom (A.U.)

        // The system is driven by the real forces and the elastic forces
        // The extended degree of freedom is driven by the elastic force and Gaussian force
        force_total = &force_real + &force_elastic;
        force_total_lambda = force_elastic_lambda + force_gaussian_lambda;

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



        // Every pace step, add a Gaussian to the list
        if (n % para.md_para.wtm_para.pace_step) == 0
        {
            pot_gaussian_lambda = gaussians_1d.get_pot(lambda);
            height = (-pot_gaussian_lambda / delta_energy).exp() * para.md_para.wtm_para.height;
            gaussians_1d.add_pot(height, lambda);
            if rank == ROOT_RANK
            {
                let mut gau_output = File::options().append(true).open(&gau_output_file).expect(&error_file("opening", &gau_output_file));
                gau_output.write_all(format!("{:15.8} {:15.8} {:15.8}\n", height, lambda * BOHR_TO_ANGSTROM, para.md_para.wtm_para.sigma).as_bytes()).expect(&error_file("writing", &gau_output_file));
            }
        }



        // Output the information in this iterative step
        if rank == ROOT_RANK
        {
            if (n % para.md_para.print_step) == 0
            {
                s.write_pdb(&str_output_file, false, n);
            }
            let mut ewtm_output = File::options().append(true).open(&ewtm_output_file).expect(&error_file("opening", &ewtm_output_file));
            ewtm_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, f_real, f_elastic, grads_norm, xi*BOHR_TO_ANGSTROM, lambda*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &ewtm_output_file));
        }
    }
}










