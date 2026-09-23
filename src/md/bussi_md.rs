use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use crate::common::constants::*;
use crate::common::error::*;
use crate::common::traits::*;
use crate::io::input::Para;
use crate::pes_exploration::system::System;
use crate::pes_exploration::potential::*;
use crate::pes_exploration::rtip::*;
use crate::nn::training_data::DataSaved;
use mpi::traits::Communicator;
use mpi::collective::Root;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use savefile::save_file;










/// A structure containing two adjacent accelerations for Velocity Verlet integrator
///
/// # Fields
/// ```
/// index: index of the current acceleration
/// acc1: one of the adjacent accelerations
/// acc2: the other acceleration
/// ```
struct Acc
{
    index: usize,
    acc1: Array2<f64>,
    acc2: Array2<f64>,
}










impl BussiNVEMD for System
{
    fn bussi_nve_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf)
    {
        // Specify the output files
        let str_output_file: PathBuf = output_path.join("md.pdb");
        let md_output_file: PathBuf = output_path.join("md.out");

        // Parallel processing
        let rank = comm.rank();



        // Define the variables for NVE MD
        let mut s: System = self.clone();

        let dt_au: f64 = para.md_para.dt * FEMTOSECOND_TO_AU;                // A.U.
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
        let mut acc: Acc = Acc
        {
            index: 1,
            acc1: Array2::zeros(s.coord.raw_dim()),
            acc2: Array2::zeros(s.coord.raw_dim()),
        };
        for i in 0..s.natom
        {
            acc.acc1[[i, 0]] = force_real[[i, 0]] / atom_mass[i];
            acc.acc1[[i, 1]] = force_real[[i, 1]] / atom_mass[i];
            acc.acc1[[i, 2]] = force_real[[i, 2]] / atom_mass[i];
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
            // Update the next atomic coordinates based on the Velocity Verlet algorithm
            t += para.md_para.dt;                    // fs
            if acc.index == 1
            {
                s.coord += &(dt_au * &vel + 0.5 * dt_au * dt_au * &acc.acc1);                  // A.U.
            }
            else
            {
                s.coord += &(dt_au * &vel + 0.5 * dt_au * dt_au * &acc.acc2);                  // A.U.
            }

            // Update the real potential energy and atomic forces for the next structure
            (pot_real, force_real) = real_pes.get_energy_force(&s);
            s.pot = pot_real;
            f_real = (&force_real * &force_real).sum().sqrt();

            // Update the next acceleration
            if acc.index == 1
            {
                acc.index = 2;
                for i in 0..s.natom
                {
                    acc.acc2[[i, 0]] = force_real[[i, 0]] / atom_mass[i];
                    acc.acc2[[i, 1]] = force_real[[i, 1]] / atom_mass[i];
                    acc.acc2[[i, 2]] = force_real[[i, 2]] / atom_mass[i];
                }
            }
            else
            {
                acc.index = 1;
                for i in 0..s.natom
                {
                    acc.acc1[[i, 0]] = force_real[[i, 0]] / atom_mass[i];
                    acc.acc1[[i, 1]] = force_real[[i, 1]] / atom_mass[i];
                    acc.acc1[[i, 2]] = force_real[[i, 2]] / atom_mass[i];
                }
            }

            // Update the next velocity based on the Velocity Verlet algorithm
            vel += &(0.5 * dt_au * (&acc.acc1 + &acc.acc2));                // A.U.



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










impl BussiNVTMD for System
{
    fn bussi_nvt_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf)
    {
        // Specify the output files
        let str_output_file: PathBuf = output_path.join("md.pdb");
        let md_output_file: PathBuf = output_path.join("md.out");

        // Parallel processing
        let rank = comm.rank();
        let root_process = comm.process_at_rank(ROOT_RANK);



        // Define the variables for NVT MD
        let mut s: System = self.clone();

        let dt_au: f64 = para.md_para.dt * FEMTOSECOND_TO_AU;                // A.U.
        let mut t: f64 = 0.0;                // fs

        let n_free: usize = match s.cell
        {
            Some(_) => 3 * s.natom - 3,                // For NVT periodic system, non-conservation of angular momentum
            None => 3 * s.natom - 3,                // For NVT non-periodic system, doesn't remove angular momentum drift explicitly
        };

        let kin_avg: f64 = 0.5 * BOLTZMANN * para.md_para.thermostat.temp_bath * JOULE_TO_HARTREE * (n_free as f64);       // Target average kinetic energy (Hartree)
        let mut kin: f64;                // Current kinetic energy (Hartree)
        let mut d_kin: f64;                // Current kinetic energy increment (Hartree)
        let mut dw: f64;                // Current Gaussian stochastic noice with mean value of 0 and variance of dt
        let mut lambda: f64;                // Velocity rescaling factor
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
        let mut acc: Acc = Acc
        {
            index: 1,
            acc1: Array2::zeros(s.coord.raw_dim()),
            acc2: Array2::zeros(s.coord.raw_dim()),
        };
        for i in 0..s.natom
        {
            acc.acc1[[i, 0]] = force_real[[i, 0]] / atom_mass[i];
            acc.acc1[[i, 1]] = force_real[[i, 1]] / atom_mass[i];
            acc.acc1[[i, 2]] = force_real[[i, 2]] / atom_mass[i];
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
            // Update the next atomic coordinates based on the Velocity Verlet algorithm
            t += para.md_para.dt;                    // fs
            if acc.index == 1
            {
                s.coord += &(dt_au * &vel + 0.5 * dt_au * dt_au * &acc.acc1);                  // A.U.
            }
            else
            {
                s.coord += &(dt_au * &vel + 0.5 * dt_au * dt_au * &acc.acc2);                  // A.U.
            }

            // Update the real potential energy and atomic forces for the next structure
            (pot_real, force_real) = real_pes.get_energy_force(&s);
            s.pot = pot_real;
            f_real = (&force_real * &force_real).sum().sqrt();

            // Update the next acceleration
            if acc.index == 1
            {
                acc.index = 2;
                for i in 0..s.natom
                {
                    acc.acc2[[i, 0]] = force_real[[i, 0]] / atom_mass[i];
                    acc.acc2[[i, 1]] = force_real[[i, 1]] / atom_mass[i];
                    acc.acc2[[i, 2]] = force_real[[i, 2]] / atom_mass[i];
                }
            }
            else
            {
                acc.index = 1;
                for i in 0..s.natom
                {
                    acc.acc1[[i, 0]] = force_real[[i, 0]] / atom_mass[i];
                    acc.acc1[[i, 1]] = force_real[[i, 1]] / atom_mass[i];
                    acc.acc1[[i, 2]] = force_real[[i, 2]] / atom_mass[i];
                }
            }

            // Update the next velocity based on the Velocity Verlet algorithm
            vel += &(0.5 * dt_au * (&acc.acc1 + &acc.acc2));                // A.U.



            // Calculate the kinetic energy
            kin = 0.0;
            for i in 0..s.natom
            {
                kin += atom_mass[i] * ( vel[[i, 0]].powi(2) + vel[[i, 1]].powi(2) + vel[[i, 2]].powi(2) );          // Hartree
            }
            kin *= 0.5;                // Hartree

            // Update the kinetic energy and atomic velocities according to Bussi thermostat, and calculate the temperature
            dw = Array1::random(1, Normal::new(0.0, para.md_para.dt.sqrt()).expect(&error_none_value("dw")))[0];                // Gaussian stochastic noice with mean value of 0 and variance of dt
            root_process.broadcast_into(&mut dw);                // Broadcast to all ranks
            d_kin = (kin_avg - kin) * (para.md_para.dt / para.md_para.thermostat.tau) + 2.0 * (kin * kin_avg / n_free as f64).sqrt() * (dw / para.md_para.thermostat.tau.sqrt());
            lambda = (1.0 + d_kin / kin).sqrt();
            vel *= lambda;
            kin += d_kin;
            temp = 2.0 * kin * HARTREE_TO_JOULE / ( BOLTZMANN * n_free as f64 );               // K



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










impl BussiNVTMD for DistributedPot
{
    fn bussi_nvt_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf)
    {
        // Specify the output files
        let str_output_file: PathBuf = output_path.join("rtip.pdb");
        let rtip_output_file: PathBuf = output_path.join("rtip.out");
        let data_bin_file: PathBuf = output_path.join("data.bin");

        // Parallel processing
        let rank = comm.rank();
        let root_process = comm.process_at_rank(ROOT_RANK);



        // Take self.visited_states[end] as the initial structure
        let mut s: System = self.visited_states[self.visited_states.len()-1].clone();

        // Define the structure for data saving
        let nstruct: usize = para.md_para.max_step / para.md_para.print_step + 1;
        let mut data: DataSaved = match &s.cell
        {
            Some(cell) =>
            {
                DataSaved
                {
                    nstruct: 0,
                    cell: Some(cell.clone().into_raw_vec()),
                    natom: s.natom,
                    atom_type: s.atom_type.as_ref().expect(&error_none_value("s.atom_type")).clone(),
                    coord: Vec::with_capacity(nstruct * s.natom * 3),
                    pot: Vec::with_capacity(nstruct),
                    force: Vec::with_capacity(nstruct * s.natom * 3),
                }
            },

            None =>
            {
                DataSaved
                {
                    nstruct: 0,
                    cell: None,
                    natom: s.natom,
                    atom_type: s.atom_type.as_ref().expect(&error_none_value("s.atom_type")).clone(),
                    coord: Vec::with_capacity(nstruct * s.natom * 3),
                    pot: Vec::with_capacity(nstruct),
                    force: Vec::with_capacity(nstruct * s.natom * 3),
                }
            },
        };

        // Define the variables for RTIP
        let mut visited_states: Vec<System> = Vec::with_capacity(self.visited_states.len() + nstruct);
        let mut a: Vec<f64> = Vec::with_capacity(self.visited_states.len() + nstruct);
        let mut sigma: Vec<f64> = Vec::with_capacity(self.visited_states.len() + nstruct);
        for i in 0..self.visited_states.len()
        {
            visited_states.push(self.visited_states[i].clone());
            a.push(self.a[i]);
            sigma.push(self.sigma[i]);
        }



        // Define the variables for NVT MD
        let dt_au: f64 = para.md_para.dt * FEMTOSECOND_TO_AU;                // A.U.
        let mut t: f64 = 0.0;                // fs

        let n_free: usize = match s.cell
        {
            Some(_) => 3 * s.natom - 3,                // For NVT periodic system, non-conservation of angular momentum
            None => 3 * s.natom - 3,                // For NVT non-periodic system, doesn't remove angular momentum drift explicitly
        };

        let kin_avg: f64 = 0.5 * BOLTZMANN * para.md_para.thermostat.temp_bath * JOULE_TO_HARTREE * (n_free as f64);       // Target average kinetic energy (Hartree)
        let mut kin: f64;                // Current kinetic energy (Hartree)
        let mut d_kin: f64;                // Current kinetic energy increment (Hartree)
        let mut dw: f64;                // Current Gaussian stochastic noice with mean value of 0 and variance of dt
        let mut lambda: f64;                // Velocity rescaling factor
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

        // Define the variables for RTIP
        let mut pot_rtip: f64;
        let mut force_rtip: Array2<f64>;
        let mut f_rtip: f64;
        let mut force_total: Array2<f64>;

        // Obtain the initial acceleration
        let mut acc: Acc = Acc
        {
            index: 1,
            acc1: Array2::zeros(s.coord.raw_dim()),
            acc2: Array2::zeros(s.coord.raw_dim()),
        };
        for i in 0..s.natom
        {
            acc.acc1[[i, 0]] = force_real[[i, 0]] / atom_mass[i];
            acc.acc1[[i, 1]] = force_real[[i, 1]] / atom_mass[i];
            acc.acc1[[i, 2]] = force_real[[i, 2]] / atom_mass[i];
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



        // Output the visited structure to PDB output file, and output the header to RTIP output file
        if rank == ROOT_RANK
        {
            for i in 0..self.visited_states.len()
            {
                if i == 0
                {
                    self.visited_states[i].write_pdb(&str_output_file, true, 0);
                }
                else
                {
                    self.visited_states[i].write_pdb(&str_output_file, false, 0);
                }
            }
            let mut rtip_output = File::create(&rtip_output_file).expect(&error_file("creating", &rtip_output_file));
            rtip_output.write_all(b"    step            time            temp             kin        pot_real        pot_rtip          f_real          f_rtip\n").expect(&error_file("writing", &rtip_output_file));
            rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", 0, t, temp, kin, pot_real, 0.0, f_real, 0.0).as_bytes()).expect(&error_file("writing", &rtip_output_file));
        }





        // Calculate the input RTIP energy and atomic forces
        let rtip_pes = RtipPES
        {
            visited_states: &visited_states,
            a: &a,
            sigma: &sigma,
            comm,
        };
        (pot_rtip, force_rtip) = rtip_pes.get_energy_force(&s);
        f_rtip = (&force_rtip * &force_rtip).sum().sqrt();


        // Perform the initial NVT MD with the input RTIP (Unchanged) iteractively
        for n in 1..(para.md_para.equi_step+1)
        {
            // Update the next atomic coordinates based on the Velocity Verlet algorithm
            t += para.md_para.dt;                    // fs
            if acc.index == 1
            {
                s.coord += &(dt_au * &vel + 0.5 * dt_au * dt_au * &acc.acc1);                  // A.U.
            }
            else
            {
                s.coord += &(dt_au * &vel + 0.5 * dt_au * dt_au * &acc.acc2);                  // A.U.
            }

            // Update the real potential energy and atomic forces for the next structure
            (pot_real, force_real) = real_pes.get_energy_force(&s);
            s.pot = pot_real;
            f_real = (&force_real * &force_real).sum().sqrt();

            // Update the next acceleration
            force_total = &force_real + &force_rtip;
            if acc.index == 1
            {
                acc.index = 2;
                for i in 0..s.natom
                {
                    acc.acc2[[i, 0]] = force_total[[i, 0]] / atom_mass[i];
                    acc.acc2[[i, 1]] = force_total[[i, 1]] / atom_mass[i];
                    acc.acc2[[i, 2]] = force_total[[i, 2]] / atom_mass[i];
                }
            }
            else
            {
                acc.index = 1;
                for i in 0..s.natom
                {
                    acc.acc1[[i, 0]] = force_total[[i, 0]] / atom_mass[i];
                    acc.acc1[[i, 1]] = force_total[[i, 1]] / atom_mass[i];
                    acc.acc1[[i, 2]] = force_total[[i, 2]] / atom_mass[i];
                }
            }

            // Update the next velocity based on the Velocity Verlet algorithm
            vel += &(0.5 * dt_au * (&acc.acc1 + &acc.acc2));                // A.U.



            // Calculate the kinetic energy
            kin = 0.0;
            for i in 0..s.natom
            {
                kin += atom_mass[i] * ( vel[[i, 0]].powi(2) + vel[[i, 1]].powi(2) + vel[[i, 2]].powi(2) );          // Hartree
            }
            kin *= 0.5;                // Hartree

            // Update the kinetic energy and atomic velocities according to Bussi thermostat, and calculate the temperature
            dw = Array1::random(1, Normal::new(0.0, para.md_para.dt.sqrt()).expect(&error_none_value("dw")))[0];                // Gaussian stochastic noice with mean value of 0 and variance of dt
            root_process.broadcast_into(&mut dw);                // Broadcast to all ranks
            d_kin = (kin_avg - kin) * (para.md_para.dt / para.md_para.thermostat.tau) + 2.0 * (kin * kin_avg / n_free as f64).sqrt() * (dw / para.md_para.thermostat.tau.sqrt());
            lambda = (1.0 + d_kin / kin).sqrt();
            vel *= lambda;
            kin += d_kin;
            temp = 2.0 * kin * HARTREE_TO_JOULE / ( BOLTZMANN * n_free as f64 );               // K



            // Output the information in this iterative step
            if rank == ROOT_RANK
            {
                let mut rtip_output = File::options().append(true).open(&rtip_output_file).expect(&error_file("opening", &rtip_output_file));
                rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, pot_rtip, f_real, f_rtip).as_bytes()).expect(&error_file("writing", &rtip_output_file));
            }
        }





        // Perform the NVT RTIP-MD with the updating RTIP iteractively
        t = 0.0;
        for n in 1..(para.md_para.max_step+1)
        {
            // Update the next atomic coordinates based on the Velocity Verlet algorithm
            t += para.md_para.dt;                    // fs
            if acc.index == 1
            {
                s.coord += &(dt_au * &vel + 0.5 * dt_au * dt_au * &acc.acc1);                  // A.U.
            }
            else
            {
                s.coord += &(dt_au * &vel + 0.5 * dt_au * dt_au * &acc.acc2);                  // A.U.
            }

            // Update the real potential energy and atomic forces for the next structure
            (pot_real, force_real) = real_pes.get_energy_force(&s);
            s.pot = pot_real;
            f_real = (&force_real * &force_real).sum().sqrt();

            // Update the RTIP potential energy and atomic forces for the next structure
            let rtip_pes = RtipPES
            {
                visited_states: &visited_states,
                a: &a,
                sigma: &sigma,
                comm,
            };
            (pot_rtip, force_rtip) = rtip_pes.get_energy_force(&s);
            f_rtip = (&force_rtip * &force_rtip).sum().sqrt();

            // Update the next acceleration
            force_total = &force_real + &force_rtip;
            if acc.index == 1
            {
                acc.index = 2;
                for i in 0..s.natom
                {
                    acc.acc2[[i, 0]] = force_total[[i, 0]] / atom_mass[i];
                    acc.acc2[[i, 1]] = force_total[[i, 1]] / atom_mass[i];
                    acc.acc2[[i, 2]] = force_total[[i, 2]] / atom_mass[i];
                }
            }
            else
            {
                acc.index = 1;
                for i in 0..s.natom
                {
                    acc.acc1[[i, 0]] = force_total[[i, 0]] / atom_mass[i];
                    acc.acc1[[i, 1]] = force_total[[i, 1]] / atom_mass[i];
                    acc.acc1[[i, 2]] = force_total[[i, 2]] / atom_mass[i];
                }
            }

            // Update the next velocity based on the Velocity Verlet algorithm
            vel += &(0.5 * dt_au * (&acc.acc1 + &acc.acc2));                // A.U.



            // Calculate the kinetic energy
            kin = 0.0;
            for i in 0..s.natom
            {
                kin += atom_mass[i] * ( vel[[i, 0]].powi(2) + vel[[i, 1]].powi(2) + vel[[i, 2]].powi(2) );          // Hartree
            }
            kin *= 0.5;                // Hartree

            // Update the kinetic energy and atomic velocities according to Bussi thermostat, and calculate the temperature
            dw = Array1::random(1, Normal::new(0.0, para.md_para.dt.sqrt()).expect(&error_none_value("dw")))[0];                // Gaussian stochastic noice with mean value of 0 and variance of dt
            root_process.broadcast_into(&mut dw);                // Broadcast to all ranks
            d_kin = (kin_avg - kin) * (para.md_para.dt / para.md_para.thermostat.tau) + 2.0 * (kin * kin_avg / n_free as f64).sqrt() * (dw / para.md_para.thermostat.tau.sqrt());
            lambda = (1.0 + d_kin / kin).sqrt();
            vel *= lambda;
            kin += d_kin;
            temp = 2.0 * kin * HARTREE_TO_JOULE / ( BOLTZMANN * n_free as f64 );               // K



            // Output the information in this iterative step
            if rank == ROOT_RANK
            {
                let mut rtip_output = File::options().append(true).open(&rtip_output_file).expect(&error_file("opening", &rtip_output_file));
                rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", n, t, temp, kin, pot_real, pot_rtip, f_real, f_rtip).as_bytes()).expect(&error_file("writing", &rtip_output_file));
            }

            // Update the visited states, and reserve the structure, real potential, and real force
            if (n % para.md_para.print_step) == 0
            {
                visited_states.push(s.clone());
                if para.rtip_para.a0 * (n as f64) < para.rtip_para.a_max
                {
                    a.push(para.rtip_para.a0 * (n as f64));
                }
                else
                {
                    a.push(para.rtip_para.a_max);
                }
                sigma.push(para.rtip_para.sigma);

                data.nstruct += 1;
                data.coord.append(&mut s.coord.clone().into_raw_vec());
                data.pot.push(pot_real);
                data.force.append(&mut force_real.into_raw_vec());

                if rank == ROOT_RANK
                {
                    s.write_pdb(&str_output_file, false, n);
                }
            }
        }





        // Save the DFT data (including structures, potentials, and forces)
        if rank == ROOT_RANK
        {
            save_file(&data_bin_file, 0, &data).expect(&error_file("creating", &data_bin_file));
        }
    }
}










