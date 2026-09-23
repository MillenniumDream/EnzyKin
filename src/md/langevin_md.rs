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










impl LangevinNVEMD for System
{
    fn langevin_nve_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf)
    {
        // Specify the output files
        let str_output_file: PathBuf = output_path.join("md.pdb");
        let md_output_file: PathBuf = output_path.join("md.out");

        // Parallel processing
        let rank = comm.rank();



        // Define the variables for NVE MD
        let mut s: System = self.clone();

        let dt_au_half: f64 = 0.5 * para.md_para.dt * FEMTOSECOND_TO_AU;                // A.U.
        let mut t: f64 = 0.0;                // fs

        let n_free: usize = match s.cell
        {
            Some(_) => 3 * s.natom - 3,                // For NVE periodic system, non-conservation of angular momentum
            None => 3 * s.natom - 3,                // For NVE non-periodic system, doesn't remove angular momentum drift explicitly
        };

        let mut kin: f64;                // Current kinetic energy (Hartree)
        let mut temp: f64;                      // Current temperature (K)

        let mut atom_mass: Vec<f64> = Vec::with_capacity(s.natom);
        for i in 0..s.natom
        {
            atom_mass.push( s.atom_type.as_ref().expect(&error_none_value("s.atom_type"))[i].get_atomic_mass() );
        }



        // Initialization
        // Calculate the real potential energy and atomic forces for the initial structure
        let (mut pot_real, mut force_real): (f64, Array2<f64>) = real_pes.get_energy_force(&s);
        s.pot = pot_real;
        let mut f_real: f64 = (&force_real * &force_real).sum().sqrt();

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



        // Output the input structure to PDB output file, and output the header to MD output file
        if rank == ROOT_RANK
        {
            s.write_pdb(&str_output_file, true, 0);
            let mut md_output = File::create(&md_output_file).expect(&error_file("creating", &md_output_file));
            md_output.write_all(b"    step            time            temp             kin             pot               f\n").expect(&error_file("writing", &md_output_file));
            md_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", 0, t, temp, kin, pot_real, f_real).as_bytes()).expect(&error_file("writing", &md_output_file));
        }





        // Perform the NVE MD iteractively
        for n in 1..(para.md_para.max_step+1)
        {
            // Update the next 1/2 step velocity based on the BAOAB algorithm
            t += para.md_para.dt;                    // fs
            vel += &(dt_au_half * &acc);                // A.U.

            // Update the next 1/2 step position based on the BAOAB algorithm
            s.coord += &(dt_au_half * &vel);                // A.U.

            // Update the next step position based on the BAOAB algorithm
            s.coord += &(dt_au_half * &vel);                // A.U.

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
                let mut md_output = File::options().append(true).open(&md_output_file).expect(&error_file("opening", &md_output_file));
                md_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, f_real).as_bytes()).expect(&error_file("writing", &md_output_file));
            }
        }
    }
}










impl LangevinNVTMD for System
{
    fn langevin_nvt_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf)
    {
        // Specify the output files
        let str_output_file: PathBuf = output_path.join("md.pdb");
        let md_output_file: PathBuf = output_path.join("md.out");

        // Parallel processing
        let rank = comm.rank();
        let root_process = comm.process_at_rank(ROOT_RANK);



        // Define the variables for NVT MD
        let mut s: System = self.clone();

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



        // Initialization
        // Calculate the real potential energy and atomic forces for the initial structure
        let (mut pot_real, mut force_real): (f64, Array2<f64>) = real_pes.get_energy_force(&s);
        s.pot = pot_real;
        let mut f_real: f64 = (&force_real * &force_real).sum().sqrt();

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



        // Output the input structure to PDB output file, and output the header to MD output file
        if rank == ROOT_RANK
        {
            s.write_pdb(&str_output_file, true, 0);
            let mut md_output = File::create(&md_output_file).expect(&error_file("creating", &md_output_file));
            md_output.write_all(b"    step            time            temp             kin             pot               f\n").expect(&error_file("writing", &md_output_file));
            md_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", 0, t, temp, kin, pot_real, f_real).as_bytes()).expect(&error_file("writing", &md_output_file));
        }





        // Perform the NVT MD iteractively
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
                let mut md_output = File::options().append(true).open(&md_output_file).expect(&error_file("opening", &md_output_file));
                md_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, f_real).as_bytes()).expect(&error_file("writing", &md_output_file));
            }
        }
    }
}










