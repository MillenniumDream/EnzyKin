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










pub fn extended_nvt_md_1d<C: Communicator, P: PES, CV: ColVar>(comm: &C, real_pes: &P, cv: &CV, para: &Para, output_path: &PathBuf)
{
    // Specify the output files
    let str_output_file: PathBuf = output_path.join("eabf.pdb");
    let eabf_output_file: PathBuf = output_path.join("eabf.out");

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

    // The system is driven by the real forces and the elastic forces
    let mut force_total: Array2<f64> = &force_real + &force_elastic;

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
        let mut eabf_output = File::create(&eabf_output_file).expect(&error_file("creating", &eabf_output_file));
        eabf_output.write_all(b"    step            time            temp             kin        pot_real          f_real       f_elastic      grads_norm              xi          lambda\n").expect(&error_file("writing", &eabf_output_file));
        eabf_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", 0, t, temp, kin, pot_real, f_real, f_elastic, grads_norm, xi*BOHR_TO_ANGSTROM, lambda*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &eabf_output_file));
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
            let mut eabf_output = File::options().append(true).open(&eabf_output_file).expect(&error_file("opening", &eabf_output_file));
            eabf_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, f_real, f_elastic, grads_norm, xi*BOHR_TO_ANGSTROM, lambda*BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", &eabf_output_file));
        }
    }
}










