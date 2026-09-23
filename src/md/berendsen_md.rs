use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use crate::common::constants::*;
use crate::common::error::*;
use crate::common::traits::*;
use crate::io::input::Para;
use crate::pes_exploration::potential::*;
use crate::pes_exploration::rtip::*;
use crate::pes_exploration::system::System;
use crate::nn::training_data::DataSaved;
use mpi::traits::Communicator;
use mpi::collective::Root;
use ndarray::{Array1, Array2, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use savefile::save_file;










#[derive(PartialEq, Debug)]
enum RtipStatus
{
    Increasing,
    Decreasing,
    Falling,
    Quiescent,
}










impl BerendsenNVTMD for RepulsivePot
{
    fn berendsen_nvt_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf)
    {
        // Specify the output files
        let str_output_file: PathBuf = output_path.join("rtip.pdb");
        let rtip_output_file: PathBuf = output_path.join("rtip.out");

        // Parallel processing
        let rank = comm.rank();
        let root_process = comm.process_at_rank(ROOT_RANK);

        // Output the local minimum structure to PDB output file, and output the header to RTIP output file
        if rank == ROOT_RANK
        {
            self.local_min.write_pdb(&str_output_file, true, 0);
            let mut rtip_output = File::create(&rtip_output_file).expect(&error_file("creating", &rtip_output_file));
            rtip_output.write_all(b"    step            time        rti_dist            temp             kin        pot_real        pot_rtip          f_real          f_rtip\n").expect(&error_file("writing", &rtip_output_file));
        }



        // Randomly obtain a nearby structure around self.local_min
        let mut s: System = self.local_min.clone();
        match &self.local_min.atom_add_pot
        {
            // If self.local_min.atom_add_pot has none value, disturb all the atoms
            None =>
            {
                let mut dcoord: Array2<f64> = Array2::random(s.coord.raw_dim(), Uniform::new(0.0, 1.0));
                let dcoord_p: &mut [f64] = dcoord.as_slice_mut().expect(&error_as_slice("dcoord"));
                root_process.broadcast_into(dcoord_p);
                dcoord = &dcoord - dcoord.mean_axis(Axis(0)).expect(&error_none_value("dcoord"));
                let dcoord_norm: f64 = (&dcoord * &dcoord).sum().sqrt();
                s.coord += &(dcoord * (0.1/dcoord_norm));
            },

            // If self.local_min.atom_add_pot specifies the atoms to be added RTI potential, only disturb these atoms
            Some(atom_add_pot) =>
            {
                let mut dcoord: Array2<f64> = Array2::random((atom_add_pot.len(), 3), Uniform::new(0.0, 1.0));
                let dcoord_p: &mut [f64] = dcoord.as_slice_mut().expect(&error_as_slice("dcoord"));
                root_process.broadcast_into(dcoord_p);
                dcoord = &dcoord - dcoord.mean_axis(Axis(0)).expect(&error_none_value("dcoord"));
                let dcoord_norm: f64 = (&dcoord * &dcoord).sum().sqrt();
                for i in 0..atom_add_pot.len()
                {
                    s.coord[[atom_add_pot[i], 0]] += dcoord[[i, 0]] * (0.1/dcoord_norm);
                    s.coord[[atom_add_pot[i], 1]] += dcoord[[i, 1]] * (0.1/dcoord_norm);
                    s.coord[[atom_add_pot[i], 2]] += dcoord[[i, 2]] * (0.1/dcoord_norm);
                }
            },
        }



        // Define the variables for NVT MD
        let mut pot_real: f64;
        let mut force_real: Array2<f64>;
        let mut f_real: f64;

        let mut local_min: System;                  // A fragment of the original local minimum containing the atoms to be added RTI potential
        let mut nearby_ts: Vec<System>;                 // A fragment of the original nearby TS containing the atoms to be added RTI potential
        let mut s_fragment: System;                 // A fragment of the current structure containing the atoms to be added RTI potential
        let mut rtip0_pes = match &self.local_min.atom_add_pot
        {
            // If self.local_min.atom_add_pot has none value, create the RTIP PES from the original local minimum and nearby TS
            None =>
            {
                // Initialize the fragments
                local_min = System
                {
                    natom: 0,
                    coord: Array2::zeros((0, 3)),
                    cell: None,
                    atom_type: None,
                    atom_add_pot: None,
                    mutable: None,
                    pot: 0.0,
                };
                nearby_ts = vec![local_min.clone(); self.nearby_ts.len()];
                s_fragment = local_min.clone();

                // Create the RTI potential from the original local minimum and nearby TS
                Rtip0PES
                {
                    local_min: &self.local_min,
                    nearby_ts: &self.nearby_ts,
                    a_min: 0.0,
                    a_ts: 0.0,
                    sigma_min: 0.0,
                    sigma_ts: vec![10.0; self.nearby_ts.len()],
                }
            },

            // If self.local_min.atom_add_pot specifies the atoms to be added RTI potential, create fragments for the local minimum and nearby TS
            Some(atom_add_pot) =>
            {
                // Initialize the fragments
                local_min = System
                {
                    natom: atom_add_pot.len(),
                    coord: Array2::zeros((atom_add_pot.len(), 3)),
                    cell: None,
                    atom_type: None,
                    atom_add_pot: None,
                    mutable: None,
                    pot: 0.0,
                };
                nearby_ts = vec![local_min.clone(); self.nearby_ts.len()];
                s_fragment = local_min.clone();

                // Copy the atomic coordinates from the original local minimum and nearby TS to the corresponding fragments
                for i in 0..atom_add_pot.len()
                {
                    local_min.coord[[i, 0]] = self.local_min.coord[[atom_add_pot[i], 0]];
                    local_min.coord[[i, 1]] = self.local_min.coord[[atom_add_pot[i], 1]];
                    local_min.coord[[i, 2]] = self.local_min.coord[[atom_add_pot[i], 2]];
                }
                for j in 0..self.nearby_ts.len()
                {
                    for i in 0..atom_add_pot.len()
                    {
                        nearby_ts[j].coord[[i, 0]] = self.nearby_ts[j].coord[[atom_add_pot[i], 0]];
                        nearby_ts[j].coord[[i, 1]] = self.nearby_ts[j].coord[[atom_add_pot[i], 1]];
                        nearby_ts[j].coord[[i, 2]] = self.nearby_ts[j].coord[[atom_add_pot[i], 2]];
                    }
                }

                // Create the RTIP PES from the fragments
                Rtip0PES
                {
                    local_min: &local_min,
                    nearby_ts: &nearby_ts,
                    a_min: 0.0,
                    a_ts: 0.0,
                    sigma_min: 0.0,
                    sigma_ts: vec![10.0; self.nearby_ts.len()],
                }
            },
        };

        let mut pot_rtip: f64 = 0.0;
        let mut force_rtip: Array2<f64> = Array2::zeros(s.coord.raw_dim());
        let mut force_rtip_fragment: Array2<f64>;
        let mut f_rtip: f64 = 0.0;

        let mut force_total: Array2<f64>;
        let dt: f64 = para.md_para.dt;
        let mut t: f64 = 0.0;
        let mut kin: f64;
        let mut temp: f64;                      // K
        let mut lambda: f64;
        let mut atom_mass: Vec<f64> = Vec::with_capacity(s.natom);
        for i in 0..s.natom
        {
            atom_mass.push( s.atom_type.as_ref().expect(&error_none_value("s.atom_type"))[i].get_atomic_mass() );
        }
        let mut vel: Array2<f64> = Array2::zeros(s.coord.raw_dim());
        let mut acc: Array2<f64> = Array2::zeros(s.coord.raw_dim());



        // Perform the traditional NVT MD iteractively
        for i in 1..(para.md_para.equi_step+1)
        {
            // First step of leapfrog method
            t += dt;                    // fs
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);                  // A.U.
            s.coord += &(dt * FEMTOSECOND_TO_AU * &vel);                  // A.U.

            // Berendsen thermostat
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );          // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            if temp < 1.0
            {
                temp = 1.0;
            }
            lambda = ( 1.0 + (dt / para.md_para.thermostat.tau) * (para.md_para.thermostat.temp_bath / temp - 1.0) ).sqrt();

            // Calculate the real potential energy and atomic forces
            (pot_real, force_real) = real_pes.get_energy_force(&s);
            s.pot = pot_real;
            f_real = (&force_real * &force_real).sum().sqrt();

            // Calculate the atomic acceleration from the real force
            for j in 0..s.natom
            {
                acc[[j, 0]] = force_real[[j, 0]] / atom_mass[j];
                acc[[j, 1]] = force_real[[j, 1]] / atom_mass[j];
                acc[[j, 2]] = force_real[[j, 2]] / atom_mass[j];
            }

            // Second step of leapfrog method
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);              // A.U.
            vel *= lambda;

            // Calculate kinetic energy
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );              // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            kin *= 0.5;                 // Hartree

            // Output the information in this iterative step
            if rank == ROOT_RANK
            {
                let mut rtip_output = File::options().append(true).open(&rtip_output_file).expect(&error_file("opening", &rtip_output_file));
                rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, t, 0.0, temp, kin, pot_real, pot_rtip, f_real, f_rtip).as_bytes()).expect(&error_file("writing", &rtip_output_file));
            }
        }



        // Perform the NVT RTIP-MD iteractively
        t = 0.0;
        for i in 1..(para.md_para.max_step+1)
        {
            // First step of leapfrog method
            t += dt;                    // fs
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);                  // A.U.
            s.coord += &(dt * FEMTOSECOND_TO_AU * &vel);                  // A.U.

            // Berendsen thermostat
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );          // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            if temp < 1.0
            {
                temp = 1.0;
            }
            lambda = ( 1.0 + (dt / para.md_para.thermostat.tau) * (para.md_para.thermostat.temp_bath / temp - 1.0) ).sqrt();

            // Calculate the real potential energy and atomic forces
            (pot_real, force_real) = real_pes.get_energy_force(&s);
            s.pot = pot_real;
            f_real = (&force_real * &force_real).sum().sqrt();



            // Calculate the RTIP potential energy and atomic forces
            match &self.local_min.atom_add_pot
            {
                // If self.local_min.atom_add_pot has none value, add the RTI potential on all the atoms
                None =>
                {
                    rtip0_pes.sigma_min = rti_dist(&self.local_min.coord, &s.coord);
                    match para.rtip_para.scale_ts_sigma
                    {
                        None =>
                        {
                            for j in 0..self.nearby_ts.len()
                            {
                                rtip0_pes.sigma_ts[j] = rti_dist(&self.nearby_ts[j].coord, &s.coord);
                            }
                        },
                        Some(scale_ts_sigma) =>
                        {
                            for j in 0..self.nearby_ts.len()
                            {
                                rtip0_pes.sigma_ts[j] = (0.5 * scale_ts_sigma) * rti_dist(&self.nearby_ts[j].coord, &self.local_min.coord);
                            }
                        },
                    }
                    rtip0_pes.a_min = para.rtip_para.a0 * (i as f64);
                    rtip0_pes.a_ts = rtip0_pes.a_min * para.rtip_para.scale_ts_a0;
                    (pot_rtip, force_rtip) = rtip0_pes.get_energy_force(&s);
                    f_rtip = (&force_rtip * &force_rtip).sum().sqrt();
                },

                // If self.local_min.atom_add_pot specifies the atoms to be add RTI potential, add the RTI potential on the fragments
                Some(atom_add_pot) =>
                {
                    for j in 0..atom_add_pot.len()
                    {
                        s_fragment.coord[[j, 0]] = s.coord[[atom_add_pot[j], 0]];
                        s_fragment.coord[[j, 1]] = s.coord[[atom_add_pot[j], 1]];
                        s_fragment.coord[[j, 2]] = s.coord[[atom_add_pot[j], 2]];
                    }
                    rtip0_pes.sigma_min = rti_dist(&local_min.coord, &s_fragment.coord);
                    match para.rtip_para.scale_ts_sigma
                    {
                        None =>
                        {
                            for j in 0..self.nearby_ts.len()
                            {
                                rtip0_pes.sigma_ts[j] = rti_dist(&nearby_ts[j].coord, &s_fragment.coord);
                            }
                        },
                        Some(scale_ts_sigma) =>
                        {
                            for j in 0..self.nearby_ts.len()
                            {
                                rtip0_pes.sigma_ts[j] = (0.5 * scale_ts_sigma) * rti_dist(&nearby_ts[j].coord, &local_min.coord);
                            }
                        },
                    }
                    rtip0_pes.a_min = para.rtip_para.a0 * (i as f64);
                    rtip0_pes.a_ts = rtip0_pes.a_min * para.rtip_para.scale_ts_a0;
                    (pot_rtip, force_rtip_fragment) = rtip0_pes.get_energy_force(&s_fragment);
                    f_rtip = (&force_rtip_fragment * &force_rtip_fragment).sum().sqrt();
                    for j in 0..atom_add_pot.len()
                    {
                        force_rtip[[atom_add_pot[j], 0]] = force_rtip_fragment[[j, 0]];
                        force_rtip[[atom_add_pot[j], 1]] = force_rtip_fragment[[j, 1]];
                        force_rtip[[atom_add_pot[j], 2]] = force_rtip_fragment[[j, 2]];
                    }
                },
            }



            // Calculate the atomic acceleration from the total force
            force_total = &force_real + &force_rtip;
            for j in 0..s.natom
            {
                acc[[j, 0]] = force_total[[j, 0]] / atom_mass[j];
                acc[[j, 1]] = force_total[[j, 1]] / atom_mass[j];
                acc[[j, 2]] = force_total[[j, 2]] / atom_mass[j];
            }

            // Second step of leapfrog method
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);              // A.U.
            vel *= lambda;

            // Calculate kinetic energy
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );              // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            kin *= 0.5;                 // Hartree

            // Output the information in this iterative step
            if rank == ROOT_RANK
            {
                if (i % para.md_para.print_step) == 0
                {
                    s.write_pdb(&str_output_file, false, i);
                }
                let mut rtip_output = File::options().append(true).open(&rtip_output_file).expect(&error_file("opening", &rtip_output_file));
                rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, t, rtip0_pes.sigma_min, temp, kin, pot_real, pot_rtip, f_real, f_rtip).as_bytes()).expect(&error_file("writing", &rtip_output_file));
            }
        }
    }
}










impl BerendsenNVTMD for EvolutionPot
{
    fn berendsen_nvt_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf)
    {
        // Specify the output files
        let str_output_file: PathBuf = output_path.join("rtip.pdb");
        let rtip_output_file: PathBuf = output_path.join("rtip.out");
        let dec_output_file: PathBuf = output_path.join("rtip_decreasing_steps");

        // Parallel processing
        let rank = comm.rank();

        // Output the initial state structure to PDB output file, and output the header to RTIP output file
        if rank == ROOT_RANK
        {
            self.initial_state.write_pdb(&str_output_file, true, 0);
            let mut rtip_output = File::create(&rtip_output_file).expect(&error_file("creating", &rtip_output_file));
            rtip_output.write_all(b"    step            time        rti_dist            temp             kin        pot_real        pot_rtip          f_real          f_rtip     rtip_status\n").expect(&error_file("writing", &rtip_output_file));
            let mut dec_output = File::create(&dec_output_file).expect(&error_file("creating", &dec_output_file));
            dec_output.write_all(b"      begin_step        end_step\n").expect(&error_file("writing", &dec_output_file));
        }

        // Begin from the initial state structure, and initialize the adjacency matrix of the system
        let mut s: System = self.initial_state.clone();
        let atomic_type_ref: &Vec<Element>;                // Initialize once
        let atomic_radii: Vec<f64>;                // Initialize once
        let mut dist_mat: Array2<f64>;                // Update in every step
        let mut adj_mat: Array2<i8>;                // Only update in the beginning of each search
        let mut mol_index: Vec<Vec<usize>>;                // Only update in the beginning of each search



        // Define the variables for NVT MD
        let mut pot_real: f64;
        let mut force_real: Array2<f64>;
        let mut f_real: f64;

        let mut o_all: Array1<f64>;                 // To reserve the geometric center of all the molecules for evolution
        let mut o_i: Array1<f64>;               // To reserve the geometric center of a special molecule for evolution

        // Initialization of the final state of the molecules for evolution, where their geometric centers coincide
        let mut final_state: System;
        let _nearby_ts: Vec<System>;
        let mut s_fragment: System;
        match &self.initial_state.atom_add_pot
        {
            None =>
            {
                final_state = System
                {
                    natom: s.natom,
                    coord: Array2::zeros(s.coord.raw_dim()),
                    cell: None,
                    atom_type: None,
                    atom_add_pot: None,
                    mutable: None,
                    pot: 0.0,
                };
                _nearby_ts = vec![final_state.clone(); 0];
                s_fragment = System
                {
                    natom: 0,
                    coord: Array2::zeros((0, 3)),
                    cell: None,
                    atom_type: None,
                    atom_add_pot: None,
                    mutable: None,
                    pot: 0.0,
                };
                atomic_type_ref = s.atom_type.as_ref().expect(&error_none_value("s.atom_type"));
                atomic_radii = s.get_atomic_radii();
                dist_mat = s.get_dist_mat();                // Update in every step
                adj_mat = System::get_adj_mat(atomic_type_ref, &atomic_radii, &dist_mat, 1.25, &para.md_para.evol_md_para.ignored_pair);                // Only update in the beginning of each search
                mol_index = System::split_into_mol(&atomic_radii, &dist_mat, 1.25);                // Initialization
            },

            Some(atom_add_pot) =>
            {
                final_state = System
                {
                    natom: atom_add_pot.len(),
                    coord: Array2::zeros((atom_add_pot.len(), 3)),
                    cell: None,
                    atom_type: None,
                    atom_add_pot: None,
                    mutable: None,
                    pot: 0.0,
                };
                _nearby_ts = vec![final_state.clone(); 0];
                s_fragment = final_state.clone();
                for i in 0..atom_add_pot.len()
                {
                    s_fragment.coord[[i, 0]] = s.coord[[atom_add_pot[i], 0]];
                    s_fragment.coord[[i, 1]] = s.coord[[atom_add_pot[i], 1]];
                    s_fragment.coord[[i, 2]] = s.coord[[atom_add_pot[i], 2]];
                }
                match &s.atom_type
                {
                    None => (),
                    Some(s_atom_type) =>
                    {
                        let mut atom_type: Vec<Element> = Vec::with_capacity(atom_add_pot.len());
                        for i in 0..atom_add_pot.len()
                        {
                            atom_type.push(s_atom_type[atom_add_pot[i]].clone());
                        }
                        s_fragment.atom_type = Some(atom_type);
                    },
                }
                atomic_type_ref = s_fragment.atom_type.as_ref().expect(&error_none_value("s_fragment.atom_type"));
                atomic_radii = s_fragment.get_atomic_radii();
                dist_mat = s_fragment.get_dist_mat();                // Update in every step
                adj_mat = System::get_adj_mat(atomic_type_ref, &atomic_radii, &dist_mat, 1.25, &para.md_para.evol_md_para.ignored_pair);                // Only update in the beginning of each search
                mol_index = System::split_into_mol(&atomic_radii, &dist_mat, 1.25);                // Initialization
            },
        }

        let mut pot_rtip: f64 = 0.0;
        let mut force_rtip: Array2<f64> = Array2::zeros(s.coord.raw_dim());
        let mut force_rtip_fragment: Array2<f64>;
        let mut f_rtip: f64 = 0.0;
        let mut a_min: f64 = 0.0;
        let mut a_min_threshold: f64 = 0.0;
        let mut rtip_dist: f64 = 0.0;

        let mut force_total: Array2<f64>;
        let mut rtip_status: RtipStatus = RtipStatus::Quiescent;
        let dt: f64 = para.md_para.dt;
        let mut t: f64 = 0.0;
        let mut kin: f64;
        let mut temp: f64;                      // K
        let mut lambda: f64;
        let mut atom_mass: Vec<f64> = Vec::with_capacity(s.natom);
        for i in 0..s.natom
        {
            atom_mass.push( s.atom_type.as_ref().expect(&error_none_value("s.atom_type"))[i].get_atomic_mass() );
        }
        let mut vel: Array2<f64> = match &self.initial_velocity
        {
            Some(initial_velocity) => initial_velocity.clone(),
            None => Array2::zeros(s.coord.raw_dim()),
        };
        let mut acc: Array2<f64> = Array2::zeros(s.coord.raw_dim());



        // Perform the traditional NVT MD iteractively
        for i in 1..(para.md_para.equi_step+1)
        {
            // First step of leapfrog method
            t += dt;                    // fs
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);                  // A.U.
            s.coord += &(dt * FEMTOSECOND_TO_AU * &vel);                  // A.U.

            // Berendsen thermostat
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );          // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            if temp < 1.0
            {
                temp = 1.0;
            }
            lambda = ( 1.0 + (dt / para.md_para.thermostat.tau) * (para.md_para.thermostat.temp_bath / temp - 1.0) ).sqrt();

            // Calculate the real potential energy and atomic forces
            (pot_real, force_real) = real_pes.get_energy_force(&s);
            s.pot = pot_real;
            f_real = (&force_real * &force_real).sum().sqrt();

            // Calculate the atomic acceleration from the real force
            for j in 0..s.natom
            {
                acc[[j, 0]] = force_real[[j, 0]] / atom_mass[j];
                acc[[j, 1]] = force_real[[j, 1]] / atom_mass[j];
                acc[[j, 2]] = force_real[[j, 2]] / atom_mass[j];
            }

            // Second step of leapfrog method
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);              // A.U.
            vel *= lambda;

            // Calculate kinetic energy
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );              // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            kin *= 0.5;                 // Hartree

            // Output the information in this iterative step
            if rank == ROOT_RANK
            {
                let mut rtip_output = File::options().append(true).open(&rtip_output_file).expect(&error_file("opening", &rtip_output_file));
                rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}      {:?}\n", i, t, rtip_dist, temp, kin, pot_real, pot_rtip, f_real, f_rtip, rtip_status).as_bytes()).expect(&error_file("writing", &rtip_output_file));
            }
        }



        // Perform the NVT RTIP-MD iteractively
        t = 0.0;
        rtip_status = RtipStatus::Increasing;
        for i in 1..(para.md_para.max_step+1)
        {
            // Control the variation of the RTIP
            if (rtip_status == RtipStatus::Increasing || rtip_status == RtipStatus::Falling) && System::judge_variation_of_bonding(&atomic_radii, &dist_mat, &adj_mat, 1.0, 1.6)
            {
                a_min_threshold = a_min * para.md_para.evol_md_para.decreasing_bound;
//                mol_index = System::split_into_mol(&atomic_radii, &dist_mat, 1.25);                // Update when the bonding changes
                rtip_status = RtipStatus::Decreasing;
                if rank == ROOT_RANK
                {
                    let mut dec_output = File::options().append(true).open(&dec_output_file).expect(&error_file("opening", &dec_output_file));
                    dec_output.write_all(format!("{:16}", i).as_bytes()).expect(&error_file("writing", &dec_output_file));
                }
            }
            if (rtip_status == RtipStatus::Decreasing) && (a_min > a_min_threshold)
            {
                // If circulate the simulation, reset RtipStatus to be Increasing
                if para.md_para.evol_md_para.circulate
                {
                    // Only update in the beginning of each search
                    adj_mat = System::get_adj_mat(atomic_type_ref, &atomic_radii, &dist_mat, 1.25, &para.md_para.evol_md_para.ignored_pair);
                    mol_index = System::split_into_mol(&atomic_radii, &dist_mat, 1.25);                // Update at the beginning of each search
                    rtip_status = RtipStatus::Increasing;
                }
                // Else, set RtipStatus to be Quiescent
                else
                {
                    rtip_status = RtipStatus::Quiescent;
                }
                if rank == ROOT_RANK
                {
                    let mut dec_output = File::options().append(true).open(&dec_output_file).expect(&error_file("opening", &dec_output_file));
                    dec_output.write_all(format!("{:16}\n", i-1).as_bytes()).expect(&error_file("writing", &dec_output_file));
                }
            }
            match para.md_para.evol_md_para.split_step
            {
                Some(split_step) =>
                {
                    if (i % split_step) == 0
                    {
                        mol_index = System::split_into_mol(&atomic_radii, &dist_mat, 1.25);                // Split the molecules every split_step
                    };
                },
                None => (),
            }
            match rtip_status
            {
                RtipStatus::Increasing => a_min -= para.rtip_para.a0,
                RtipStatus::Decreasing => a_min += para.rtip_para.a0 * para.md_para.evol_md_para.decreasing_multiple,
                RtipStatus::Falling => a_min += para.rtip_para.a0 * para.md_para.evol_md_para.falling_multiple,
                RtipStatus::Quiescent =>
                {
                    if a_min < 0.0
                    {
                        a_min += para.rtip_para.a0 * para.md_para.evol_md_para.decreasing_multiple;
                    }
                    else
                    {
                        a_min = 0.0;
                    }
                },
            }



            // First step of leapfrog method
            t += dt;                    // fs
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);                  // A.U.
            s.coord += &(dt * FEMTOSECOND_TO_AU * &vel);                  // A.U.

            // Berendsen thermostat
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );          // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            if temp < 1.0
            {
                temp = 1.0;
            }
            lambda = ( 1.0 + (dt / para.md_para.thermostat.tau) * (para.md_para.thermostat.temp_bath / temp - 1.0) ).sqrt();

            // Calculate the real potential energy and atomic forces
            (pot_real, force_real) = real_pes.get_energy_force(&s);
            s.pot = pot_real;
            f_real = (&force_real * &force_real).sum().sqrt();



            // Calculate the RTIP potential energy and atomic forces
            match &self.initial_state.atom_add_pot
            {
                None =>
                {
                    // Update distance matrix and the status of RTIP
                    dist_mat = s.get_dist_mat();                // Update in every step
                    if rtip_status == RtipStatus::Increasing || rtip_status == RtipStatus::Falling
                    {
                        // If there are two molecules too close, remain the RTIP unchanged in the current step
                        if System::judge_adj_of_mol(&mol_index, &atomic_radii, &adj_mat, &dist_mat, 1.2)
                        {
                            rtip_status = RtipStatus::Falling;
                        }
                        // Otherwise, increase the RTIP in the current step
                        else
                        {
                            rtip_status = RtipStatus::Increasing;
                        }
                    }

                    // Define the final state of the molecules for evolution, where their geometric centers coincide
                    o_all = s.coord.mean_axis(Axis(0)).expect(&error_none_value("s.coord"));
                    for j in 0..mol_index.len()                // For each molecule in evolution
                    {
                        // Find the geometric center of Molecule j
                        o_i = Array1::zeros(3);
                        for k in 0..mol_index[j].len()                 // for each atom in Molecule j
                        {
                            o_i += &s.coord.slice(s![mol_index[j][k], ..]);
                        }
                        o_i /= mol_index[j].len() as f64;
                        // Move Molecule j for the final state
                        for k in 0..mol_index[j].len()
                        {
                            final_state.coord[[mol_index[j][k], 0]] = s.coord[[mol_index[j][k], 0]] + 0.01 * (o_all[0] - o_i[0]);
                            final_state.coord[[mol_index[j][k], 1]] = s.coord[[mol_index[j][k], 1]] + 0.01 * (o_all[1] - o_i[1]);
                            final_state.coord[[mol_index[j][k], 2]] = s.coord[[mol_index[j][k], 2]] + 0.01 * (o_all[2] - o_i[2]);
                        }
                    }

                    // Calculate the RTIP potential energy and atomic forces
                    rtip_dist = rti_dist(&final_state.coord, &s.coord);
                    let rtip0_pes = Rtip0PES
                    {
                        local_min: &final_state,
                        nearby_ts: &_nearby_ts,
                        a_min,
                        a_ts: 0.0,
                        sigma_min: rtip_dist,
                        sigma_ts: vec![10.0; 0],
                    };
                    (pot_rtip, force_rtip) = rtip0_pes.get_energy_force(&s);
                    f_rtip = (&force_rtip * &force_rtip).sum().sqrt();
                },

                Some(atom_add_pot) =>
                {
                    // Extract the atomic coordinates of the molecules for evolution
                    for j in 0..atom_add_pot.len()
                    {
                        s_fragment.coord[[j, 0]] = s.coord[[atom_add_pot[j], 0]];
                        s_fragment.coord[[j, 1]] = s.coord[[atom_add_pot[j], 1]];
                        s_fragment.coord[[j, 2]] = s.coord[[atom_add_pot[j], 2]];
                    }

                    // Update distance matrix and the status of RTIP
                    dist_mat = s_fragment.get_dist_mat();                // Update in every step
                    if rtip_status == RtipStatus::Increasing || rtip_status == RtipStatus::Falling
                    {
                        // If there are two molecules too close, remain the RTIP unchanged in the current step
                        if System::judge_adj_of_mol(&mol_index, &atomic_radii, &adj_mat, &dist_mat, 1.2)
                        {
                            rtip_status = RtipStatus::Falling;
                        }
                        // Otherwise, increase the RTIP in the current step
                        else
                        {
                            rtip_status = RtipStatus::Increasing;
                        }
                    }

                    // Define the final state of the molecules for evolution, where their geometric centers coincide
                    o_all = s_fragment.coord.mean_axis(Axis(0)).expect(&error_none_value("s_fragment.coord"));
                    for j in 0..mol_index.len()                // For each molecule in evolution
                    {
                        // Find the geometric center of Molecule j
                        o_i = Array1::zeros(3);
                        for k in 0..mol_index[j].len()                 // for each atom in Molecule j
                        {
                            o_i += &s_fragment.coord.slice(s![mol_index[j][k], ..]);
                        }
                        o_i /= mol_index[j].len() as f64;
                        // Move Molecule j for the final state
                        for k in 0..mol_index[j].len()
                        {
                            final_state.coord[[mol_index[j][k], 0]] = s_fragment.coord[[mol_index[j][k], 0]] + 0.01 * (o_all[0] - o_i[0]);
                            final_state.coord[[mol_index[j][k], 1]] = s_fragment.coord[[mol_index[j][k], 1]] + 0.01 * (o_all[1] - o_i[1]);
                            final_state.coord[[mol_index[j][k], 2]] = s_fragment.coord[[mol_index[j][k], 2]] + 0.01 * (o_all[2] - o_i[2]);
                        }
                    }

                    // Calculate the RTIP potential energy and atomic forces
                    rtip_dist = rti_dist(&final_state.coord, &s_fragment.coord);
                    let rtip0_pes = Rtip0PES
                    {
                        local_min: &final_state,
                        nearby_ts: &_nearby_ts,
                        a_min,
                        a_ts: 0.0,
                        sigma_min: rtip_dist,
                        sigma_ts: vec![10.0; 0],
                    };
                    (pot_rtip, force_rtip_fragment) = rtip0_pes.get_energy_force(&s_fragment);
                    f_rtip = (&force_rtip_fragment * &force_rtip_fragment).sum().sqrt();
                    for j in 0..atom_add_pot.len()
                    {
                        force_rtip[[atom_add_pot[j], 0]] = force_rtip_fragment[[j, 0]];
                        force_rtip[[atom_add_pot[j], 1]] = force_rtip_fragment[[j, 1]];
                        force_rtip[[atom_add_pot[j], 2]] = force_rtip_fragment[[j, 2]];
                    }
                },
            }



            // Calculate the atomic acceleration from the total force
            force_total = &force_real + &force_rtip;
            for j in 0..s.natom
            {
                acc[[j, 0]] = force_total[[j, 0]] / atom_mass[j];
                acc[[j, 1]] = force_total[[j, 1]] / atom_mass[j];
                acc[[j, 2]] = force_total[[j, 2]] / atom_mass[j];
            }

            // Second step of leapfrog method
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);              // A.U.
            vel *= lambda;

            // Calculate kinetic energy
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );              // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            kin *= 0.5;                 // Hartree

            // Output the information in this iterative step
            if rank == ROOT_RANK
            {
                if (i % para.md_para.print_step) == 0
                {
                    s.write_pdb(&str_output_file, false, i);
                }
                let mut rtip_output = File::options().append(true).open(&rtip_output_file).expect(&error_file("opening", &rtip_output_file));
                rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}      {:?}\n", i, t, rtip_dist, temp, kin, pot_real, pot_rtip, f_real, f_rtip, rtip_status).as_bytes()).expect(&error_file("writing", &rtip_output_file));
            }
        }
    }
}










impl BerendsenNVTMD for DistributedPot
{
    fn berendsen_nvt_md<C: Communicator, P: PES>(&self, comm: &C, real_pes: &P, para: &Para, output_path: &PathBuf)
    {
        // Specify the output files
        let str_output_file: PathBuf = output_path.join("rtip.pdb");
        let rtip_output_file: PathBuf = output_path.join("rtip.out");
        let data_bin_file: PathBuf = output_path.join("data.bin");
        let rtip_pot_bin_file: PathBuf = output_path.join("rtip_pot.bin");

        // Parallel processing
        let rank = comm.rank();

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
        }



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
        let mut pot_real: f64;
        let mut force_real: Array2<f64>;
        let mut f_real: f64;

        let mut pot_rtip: f64;
        let mut force_rtip: Array2<f64>;
        let mut f_rtip: f64;

        let mut force_total: Array2<f64>;
        let dt: f64 = para.md_para.dt;
        let mut t: f64 = 0.0;
        let mut kin: f64;
        let mut temp: f64;
        let mut lambda: f64;
        let mut atom_mass: Vec<f64> = Vec::with_capacity(s.natom);
        for i in 0..s.natom
        {
            atom_mass.push( s.atom_type.as_ref().expect(&error_none_value("s.atom_type"))[i].get_atomic_mass() );
        }
        let mut vel: Array2<f64> = Array2::zeros(s.coord.raw_dim());
        let mut acc: Array2<f64> = Array2::zeros(s.coord.raw_dim());



        // Perform the traditional NVT MD iteractively
        for i in 1..(para.md_para.equi_step+1)
        {
            // First step of leapfrog method
            t += dt;                    // fs
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);                  // A.U.
            s.coord += &(dt * FEMTOSECOND_TO_AU * &vel);                  // A.U.
            
            // Berendsen thermostat
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );          // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            if temp < 1.0
            {
                temp = 1.0;
            }
            lambda = ( 1.0 + (dt / para.md_para.thermostat.tau) * (para.md_para.thermostat.temp_bath / temp - 1.0) ).sqrt();

            // Calculate the real potential energy and atomic forces
            (pot_real, force_real) = real_pes.get_energy_force(&s);
            s.pot = pot_real;
            f_real = (&force_real * &force_real).sum().sqrt();

            // Calculate the RTIP potential energy and atomic forces
            let rtip_pes = RtipPES
            {
                visited_states: &visited_states,
                a: &a,
                sigma: &sigma,
                comm,
            };
            (pot_rtip, force_rtip) = rtip_pes.get_energy_force(&s);
            f_rtip = (&force_rtip * &force_rtip).sum().sqrt();

            // Calculate the atomic acceleration from the total force
            force_total = &force_real + &force_rtip;
            for j in 0..s.natom
            {
                acc[[j, 0]] = force_total[[j, 0]] / atom_mass[j];
                acc[[j, 1]] = force_total[[j, 1]] / atom_mass[j];
                acc[[j, 2]] = force_total[[j, 2]] / atom_mass[j];
            }

            // Second step of leapfrog method
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);              // A.U.
            vel *= lambda;

            // Calculate kinetic energy
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );              // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            kin *= 0.5;                 // Hartree
            
            // Output the information in this iterative step
            if rank == ROOT_RANK
            {
                let mut rtip_output = File::options().append(true).open(&rtip_output_file).expect(&error_file("opening", &rtip_output_file));
                rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, t, temp, kin, pot_real, pot_rtip, f_real, f_rtip).as_bytes()).expect(&error_file("writing", &rtip_output_file));
            }
        }



        // Perform the NVT RTIP-MD iteractively
        t = 0.0;
        for i in 1..(para.md_para.max_step+1)
        {
            // First step of leapfrog method
            t += dt;                    // fs
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);                  // A.U.
            s.coord += &(dt * FEMTOSECOND_TO_AU * &vel);                  // A.U.

            // Berendsen thermostat
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );          // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            if temp < 1.0
            {
                temp = 1.0;
            }
            lambda = ( 1.0 + (dt / para.md_para.thermostat.tau) * (para.md_para.thermostat.temp_bath / temp - 1.0) ).sqrt();

            // Calculate the real potential energy and atomic forces
            (pot_real, force_real) = real_pes.get_energy_force(&s);
            s.pot = pot_real;
            f_real = (&force_real * &force_real).sum().sqrt();

            // Calculate the RTIP potential energy and atomic forces
            let rtip_pes = RtipPES
            {
                visited_states: &visited_states,
                a: &a,
                sigma: &sigma,
                comm,
            };
            (pot_rtip, force_rtip) = rtip_pes.get_energy_force(&s);
            f_rtip = (&force_rtip * &force_rtip).sum().sqrt();

            // Calculate the atomic acceleration from the total force
            force_total = &force_real + &force_rtip;
            for j in 0..s.natom
            {
                acc[[j, 0]] = force_total[[j, 0]] / atom_mass[j];
                acc[[j, 1]] = force_total[[j, 1]] / atom_mass[j];
                acc[[j, 2]] = force_total[[j, 2]] / atom_mass[j];
            }

            // Second step of leapfrog method
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);              // A.U.
            vel *= lambda;

            // Calculate kinetic energy
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );              // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            kin *= 0.5;                 // Hartree

            // Output the information in this iterative step
            if rank == ROOT_RANK
            {
                let mut rtip_output = File::options().append(true).open(&rtip_output_file).expect(&error_file("opening", &rtip_output_file));
                rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, t, temp, kin, pot_real, pot_rtip, f_real, f_rtip).as_bytes()).expect(&error_file("writing", &rtip_output_file));
            }

            // Update the visited states, and reserve the structure, real potential, and real force
            if (i % para.md_para.print_step) == 0
            {
                visited_states.push(s.clone());
                if para.rtip_para.a0 * (i as f64) < para.rtip_para.a_max
                {
                    a.push(para.rtip_para.a0 * (i as f64));
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
                    s.write_pdb(&str_output_file, false, i);
                }
            }
        }



        // Save the DFT data (including structures, potentials, and forces)
        let rtip_pot: (Vec<f64>, Vec<f64>) = (a, sigma);
        if rank == ROOT_RANK
        {
            save_file(&data_bin_file, 0, &data).expect(&error_file("creating", &data_bin_file));
            save_file(&rtip_pot_bin_file, 0, &rtip_pot).expect(&error_file("creating", &rtip_pot_bin_file));
        }
    }
}










impl BerendsenNVTMD2 for DistributedPot
{
    fn berendsen_nvt_md2<C: Communicator, P: PES>(&self, comm: &C, real_pes1: &P, real_pes2: &P, para: &Para, output_path: &PathBuf)
    {
        // Specify the output files
        let str_output_file: PathBuf = output_path.join("rtip.pdb");
        let rtip_output_file: PathBuf = output_path.join("rtip.out");
        let data_output_file: PathBuf = output_path.join("data.out");
        let data_bin_file: PathBuf = output_path.join("data.bin");
        let rtip_pot_bin_file: PathBuf = output_path.join("rtip_pot.bin");

        // Parallel processing
        let rank = comm.rank();

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
            rtip_output.write_all(b"    step            time            temp             kin        pot_real1       pot_rtip          f_real1         f_rtip\n").expect(&error_file("writing", &rtip_output_file));
            let mut data_output = File::create(&data_output_file).expect(&error_file("creating", &data_output_file));
            data_output.write_all(b"   index       pot_real2         f_real2\n").expect(&error_file("writing", &data_output_file));
        }



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
        let mut pot_real1: f64;
        let mut force_real1: Array2<f64>;
        let mut f_real1: f64;
        let mut pot_real2: f64;
        let mut force_real2: Array2<f64>;
        let mut f_real2: f64;

        let mut pot_rtip: f64;
        let mut force_rtip: Array2<f64>;
        let mut f_rtip: f64;

        let mut force_total: Array2<f64>;
        let dt: f64 = para.md_para.dt;
        let mut t: f64 = 0.0;
        let mut kin: f64;
        let mut temp: f64;
        let mut lambda: f64;
        let mut atom_mass: Vec<f64> = Vec::with_capacity(s.natom);
        for i in 0..s.natom
        {
            atom_mass.push( s.atom_type.as_ref().expect(&error_none_value("s.atom_type"))[i].get_atomic_mass() );
        }
        let mut vel: Array2<f64> = Array2::zeros(s.coord.raw_dim());
        let mut acc: Array2<f64> = Array2::zeros(s.coord.raw_dim());



        // Perform the traditional NVT MD iteractively
        for i in 1..(para.md_para.equi_step+1)
        {
            // First step of leapfrog method
            t += dt;                    // fs
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);                  // A.U.
            s.coord += &(dt * FEMTOSECOND_TO_AU * &vel);                  // A.U.
            
            // Berendsen thermostat
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );          // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            if temp < 1.0
            {
                temp = 1.0;
            }
            lambda = ( 1.0 + (dt / para.md_para.thermostat.tau) * (para.md_para.thermostat.temp_bath / temp - 1.0) ).sqrt();

            // Calculate the first real potential energy and atomic forces
            (pot_real1, force_real1) = real_pes1.get_energy_force(&s);
            s.pot = pot_real1;
            f_real1 = (&force_real1 * &force_real1).sum().sqrt();

            // Calculate the RTIP potential energy and atomic forces
            let rtip_pes = RtipPES
            {
                visited_states: &visited_states,
                a: &a,
                sigma: &sigma,
                comm,
            };
            (pot_rtip, force_rtip) = rtip_pes.get_energy_force(&s);
            f_rtip = (&force_rtip * &force_rtip).sum().sqrt();

            // Calculate the atomic acceleration from the total force
            force_total = &force_real1 + &force_rtip;
            for j in 0..s.natom
            {
                acc[[j, 0]] = force_total[[j, 0]] / atom_mass[j];
                acc[[j, 1]] = force_total[[j, 1]] / atom_mass[j];
                acc[[j, 2]] = force_total[[j, 2]] / atom_mass[j];
            }

            // Second step of leapfrog method
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);              // A.U.
            vel *= lambda;

            // Calculate kinetic energy
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );              // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            kin *= 0.5;                 // Hartree
            
            // Output the information in this iterative step
            if rank == ROOT_RANK
            {
                let mut rtip_output = File::options().append(true).open(&rtip_output_file).expect(&error_file("opening", &rtip_output_file));
                rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, t, temp, kin, pot_real1, pot_rtip, f_real1, f_rtip).as_bytes()).expect(&error_file("writing", &rtip_output_file));
            }
        }



        // Perform the NVT RTIP-MD iteractively
        t = 0.0;
        for i in 1..(para.md_para.max_step+1)
        {
            // First step of leapfrog method
            t += dt;                    // fs
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);                  // A.U.
            s.coord += &(dt * FEMTOSECOND_TO_AU * &vel);                  // A.U.

            // Berendsen thermostat
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );          // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            if temp < 1.0
            {
                temp = 1.0;
            }
            lambda = ( 1.0 + (dt / para.md_para.thermostat.tau) * (para.md_para.thermostat.temp_bath / temp - 1.0) ).sqrt();

            // Calculate the first real potential energy and atomic forces
            (pot_real1, force_real1) = real_pes1.get_energy_force(&s);
            s.pot = pot_real1;
            f_real1 = (&force_real1 * &force_real1).sum().sqrt();

            // Calculate the RTIP potential energy and atomic forces
            let rtip_pes = RtipPES
            {
                visited_states: &visited_states,
                a: &a,
                sigma: &sigma,
                comm,
            };
            (pot_rtip, force_rtip) = rtip_pes.get_energy_force(&s);
            f_rtip = (&force_rtip * &force_rtip).sum().sqrt();

            // Calculate the atomic acceleration from the total force
            force_total = &force_real1 + &force_rtip;
            for j in 0..s.natom
            {
                acc[[j, 0]] = force_total[[j, 0]] / atom_mass[j];
                acc[[j, 1]] = force_total[[j, 1]] / atom_mass[j];
                acc[[j, 2]] = force_total[[j, 2]] / atom_mass[j];
            }

            // Second step of leapfrog method
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);              // A.U.
            vel *= lambda;

            // Calculate kinetic energy
            kin = 0.0;
            for j in 0..s.natom
            {
                kin += atom_mass[j] * ( vel[[j, 0]].powi(2) + vel[[j, 1]].powi(2) + vel[[j, 2]].powi(2) );              // Hartree
            }
            temp = kin * HARTREE_TO_JOULE / ( BOLTZMANN * 3.0 * (s.natom-1) as f64 );               // K
            kin *= 0.5;                 // Hartree

            // Output the information in this iterative step
            if rank == ROOT_RANK
            {
                let mut rtip_output = File::options().append(true).open(&rtip_output_file).expect(&error_file("opening", &rtip_output_file));
                rtip_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, t, temp, kin, pot_real1, pot_rtip, f_real1, f_rtip).as_bytes()).expect(&error_file("writing", &rtip_output_file));
            }

            // Update the visited states, and reserve the structure, the second real potential, and atomic force
            if (i % para.md_para.print_step) == 0
            {
                visited_states.push(s.clone());
                if para.rtip_para.a0 * (i as f64) < para.rtip_para.a_max
                {
                    a.push(para.rtip_para.a0 * (i as f64));
                }
                else
                {
                    a.push(para.rtip_para.a_max);
                }
                sigma.push(para.rtip_para.sigma);

                // Calculate the second real potential energy and atomic forces
                (pot_real2, force_real2) = real_pes2.get_energy_force(&s);
                f_real2 = (&force_real2 * &force_real2).sum().sqrt();

                data.nstruct += 1;
                data.coord.append(&mut s.coord.clone().into_raw_vec());
                data.pot.push(pot_real2);
                data.force.append(&mut force_real2.into_raw_vec());

                if rank == ROOT_RANK
                {
                    s.write_pdb(&str_output_file, false, i);
                    let mut data_output = File::options().append(true).open(&data_output_file).expect(&error_file("opening", &data_output_file));
                    data_output.write_all(format!("{:8} {:15.8} {:15.8}\n", data.nstruct, pot_real2, f_real2).as_bytes()).expect(&error_file("writing", &data_output_file));
                }
            }
        }



        // Save the DFT data (including structures, potentials, and forces)
        let rtip_pot: (Vec<f64>, Vec<f64>) = (a, sigma);
        if rank == ROOT_RANK
        {
            save_file(&data_bin_file, 0, &data).expect(&error_file("creating", &data_bin_file));
            save_file(&rtip_pot_bin_file, 0, &rtip_pot).expect(&error_file("creating", &rtip_pot_bin_file));
        }
    }
}










