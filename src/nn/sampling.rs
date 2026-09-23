//! Sample the deviated structures for neural network training
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use crate::common::constants::*;
use crate::common::error::*;
use crate::common::traits::ProteinPES;
use crate::io::input::Para;
use crate::nn::protein::ProteinSystem;
use crate::nn::training_data::DataSaved;
use crate::nn::nn_potential::NNPot;
use mpi::traits::Communicator;
use ndarray::Array2;
use savefile::save_file;





impl NNPot
{
    /// For a pre-training NN, use it to run a MD, and sample the deviated structures with respect to a real PES for further training
    ///
    /// # Parameters
    /// ```
    /// comm: the communicator for MPI calculations
    /// cp2k_pes: the input CP2K PES
    /// s: the input protein system
    /// para: the global parameters for MD simulation
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn md_deviated_samping<C: Communicator, P: ProteinPES>(&self, comm: &C, cp2k_pes: &P, s: &mut ProteinSystem, para: &Para)
    {
        // Specify the output files
        let str_output_file: PathBuf = self.output_path.join("md_dev_samp.pdb");
        let md_output_file: PathBuf = self.output_path.join("md_dev_samp.out");
        let data_output_file: PathBuf = self.output_path.join("data.out");
        let data_bin_file: PathBuf = self.output_path.join("data.bin");

        // Parallel processing
        let rank = comm.rank();

        // Output the initial structure to PDB output file, and output the header to MD and DATA output file
        if rank == ROOT_RANK
        {
            s.write_pdb(&str_output_file, true, 0);
            let mut md_output = File::create(&md_output_file).expect(&error_file("creating", &md_output_file));
            md_output.write_all(b"    step            time            temp             kin       pot_nn_es        pot_cp2k         f_nn_es          f_cp2k\n").expect(&error_file("writing", &md_output_file));
            let mut data_output = File::create(&data_output_file).expect(&error_file("creating", &data_output_file));
            data_output.write_all(b"    step       pot_nn_es        pot_cp2k        pot_diff         f_nn_es          f_cp2k      f_rms_diff\n").expect(&error_file("writing", &data_output_file));
        }

        // Initialize the long-ranged interaction parameters
        s.long_ranged_dir = self.long_ranged_dir.clone();
        s.get_fixed_long_ranged_para();
        *s = s.to_atom_fragment();



        // Define the structure for data saving
        let nstruct: usize = para.md_para.max_step;
        let mut data: DataSaved = match &s.cell
        {
            Some(cell) =>
            {
                DataSaved
                {
                    nstruct: 0,
                    cell: Some(cell.clone().into_raw_vec()),
                    natom: s.natom,
                    atom_type: s.atom_type.clone(),
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
                    atom_type: s.atom_type.clone(),
                    coord: Vec::with_capacity(nstruct * s.natom * 3),
                    pot: Vec::with_capacity(nstruct),
                    force: Vec::with_capacity(nstruct * s.natom * 3),
                }
            },
        };



        // Define the variables for NVT MD
        let mut pot_nn_es: f64;
        let mut force_nn_es: Array2<f64>;
        let mut f_nn_es: f64;
        let mut pot_cp2k: f64;
        let mut force_cp2k: Array2<f64>;
        let mut f_cp2k: f64;
        let mut pot_diff: f64;
        let mut force_diff: Array2<f64>;
        let mut f_rms_diff: f64;

        let dt: f64 = para.md_para.dt;
        let mut t: f64 = 0.0;
        let mut kin: f64;
        let mut temp: f64;
        let mut lambda: f64;
        let mut atom_mass: Vec<f64> = Vec::with_capacity(s.natom);
        for i in 0..s.natom
        {
            atom_mass.push( s.atom_type[i].get_atomic_mass() );
        }
        let mut vel: Array2<f64> = Array2::zeros(s.coord.raw_dim());
        let mut acc: Array2<f64> = Array2::zeros(s.coord.raw_dim());



        // Perform NVT MD iteractively for data sampling
        for i in 1..(para.md_para.max_step+1)
        {
            // First step of leapfrog method
            t += dt;                    // fs
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);                  // A.U.
            s.coord += &(dt * FEMTOSECOND_TO_AU * &vel);                  // A.U.
            for j in 0..s.fragment.len()                // For each fragment in the structure
            {
                let atom_list: Vec<usize> = s.fragment[j].atom_list.clone().expect(&error_none_value("s.fragment[j].atom_list"));
                for k in 0..atom_list.len()                // For each atom in the fragment
                {
                    s.fragment[j].coord[[k, 0]] = s.coord[[atom_list[k], 0]];
                    s.fragment[j].coord[[k, 1]] = s.coord[[atom_list[k], 1]];
                    s.fragment[j].coord[[k, 2]] = s.coord[[atom_list[k], 2]];
                }
            }

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

            // Calculate the nn and es potential energy and atomic forces
            (pot_nn_es, force_nn_es) = self.get_energy_force(s);
            s.pot = pot_nn_es;
            f_nn_es = (&force_nn_es * &force_nn_es).sum().sqrt();

            // Calculate the CP2K potential energy and atomic forces
            (pot_cp2k, force_cp2k) = cp2k_pes.get_energy_force(s);
            f_cp2k = (&force_cp2k * &force_cp2k).sum().sqrt();

            // Calculate the difference between NN PES and CP2K PES
            pot_diff = (pot_nn_es - pot_cp2k).abs();
            force_diff = &force_nn_es - &force_cp2k;
            f_rms_diff = ( (&force_diff * &force_diff).sum() / (3.0 * s.natom as f64) ).sqrt();

            // If the differences between the NN and CP2K potential and force are small, adopt the NN PES
            if pot_diff < para.md_para.dev_md_para.pot_replaced_epsilon && f_rms_diff < para.md_para.dev_md_para.f_replaced_epsilon
            {
                for j in 0..s.natom
                {
                    acc[[j, 0]] = force_nn_es[[j, 0]] / atom_mass[j];
                    acc[[j, 1]] = force_nn_es[[j, 1]] / atom_mass[j];
                    acc[[j, 2]] = force_nn_es[[j, 2]] / atom_mass[j];
                }
            }
            // If there are significant difference between the NN and CP2K potential and force, use CP2K PES to replace NN PES
            else
            {
                for j in 0..s.natom
                {
                    acc[[j, 0]] = force_cp2k[[j, 0]] / atom_mass[j];
                    acc[[j, 1]] = force_cp2k[[j, 1]] / atom_mass[j];
                    acc[[j, 2]] = force_cp2k[[j, 2]] / atom_mass[j];
                }
            }

            // Second step of leapfrog method
            vel += &(0.5 * dt * FEMTOSECOND_TO_AU * &acc);              // A.U.
            vel *= lambda;

            // Calculate kinetic energy and temperature
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
                let mut md_output = File::options().append(true).open(&md_output_file).expect(&error_file("opening", &md_output_file));
                md_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, t, temp, kin, pot_nn_es, pot_cp2k, f_nn_es, f_cp2k).as_bytes()).expect(&error_file("writing", &md_output_file));
            }

             // Reserve the deviated structures and recorded their informations
            if pot_diff > para.md_para.dev_md_para.pot_recorded_epsilon || f_rms_diff > para.md_para.dev_md_para.f_recorded_epsilon
            {
                data.nstruct += 1;
                data.coord.append(&mut s.coord.clone().into_raw_vec());
                data.pot.push(pot_cp2k);
                data.force.append(&mut force_cp2k.into_raw_vec());

                if rank == ROOT_RANK
                {
                    s.write_pdb(&str_output_file, false, i);
                    let mut data_output = File::options().append(true).open(&data_output_file).expect(&error_file("opening", &data_output_file));
                    data_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, pot_nn_es, pot_cp2k, pot_diff, f_nn_es, f_cp2k, f_rms_diff).as_bytes()).expect(&error_file("writing", &data_output_file));
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










