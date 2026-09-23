//! Calculate the root-mean-square-error (RMSE) for the neural network potential energy surface
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::fmt::Debug;
use crate::common::constants::*;
use crate::common::error::*;
use crate::common::traits::ProteinPES;
use crate::io::input::Para;
use crate::nn::protein::ProteinSystem;
use crate::nn::training_data::DataSaved;
use crate::nn::nn_potential::NNPot;
use mpi::traits::Communicator;
use savefile::load_file;
use ndarray::{Array2, Array3, s};





impl NNPot
{
    /// For a NN PES, calculate its RMSE on a given data set
    ///
    /// # Parameters
    /// ```
    /// input_data_dir: the directory containing the input data files
    /// rmse_pot: the output RMSE for potential energy
    /// rmse_force: the output RMSE for atomic force (averaging over each atom and each component)
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn data_set_rmse<P: AsRef<Path> + Debug>(&self, input_data_dir: P) -> (f64, f64)
    {
        // Obtain the input filenames
        let str_input_file: PathBuf = input_data_dir.as_ref().join("str.pdb");
        let data_input_file: PathBuf = input_data_dir.as_ref().join("data.bin");

        // Load the input files
        let mut s: ProteinSystem = ProteinSystem::read_pdb(&str_input_file);
        let data: DataSaved = load_file(&data_input_file, 0).expect(&error_file("reading", &data_input_file));

        // Assert if the input protein system and the input DFT data is matching
        assert_eq!(s.natom, data.natom);
        assert_eq!(s.atom_type, data.atom_type);

        // Extract the atomic coordinates, potentials, and forces from the data
        let coord: Array3<f64> = Array3::from_shape_vec((data.nstruct, data.natom, 3), data.coord).expect(&error_none_value("data.coord"));
        let pot: Vec<f64> = data.pot;
        let force: Array3<f64> = Array3::from_shape_vec((data.nstruct, data.natom, 3), data.force).expect(&error_none_value("data.force"));

        // Initialize the long-ranged interaction parameters
        s.long_ranged_dir = self.long_ranged_dir.clone();
        s.get_fixed_long_ranged_para();
//        s = s.to_atom_fragment();



        // Accumulate the square error to the potential energy and atomic force
        let mut acc_squ_pot: f64 = 0.0;
        let mut acc_squ_force: f64 = 0.0;
        let mut predicted_pot: f64;
        let mut predicted_force: Array2<f64>;
        for i in 0..data.nstruct                // For each structure
        {
            // Copy the atomic coordinates to the ProteinSystem
            s.coord = coord.slice(s![i, .., ..]).to_owned();
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

            // Calculate the neural network potential energy and atomic force
            (predicted_pot, predicted_force) = self.get_energy_force(&mut s);

            // Accumulate the square error
            acc_squ_pot += (predicted_pot - pot[i]).powi(2);
            acc_squ_force += ( predicted_force - force.slice(s![i, .., ..]) ).mapv(|x| x.powi(2)).sum();
        }



        ( (acc_squ_pot / data.nstruct as f64).sqrt(), (acc_squ_force / (data.nstruct * data.natom * 3) as f64).sqrt() )
    }





    /// For a NN PES, calculate its RMSE with respect to a CP2K PES based on MD simulation
    ///
    /// # Parameters
    /// ```
    /// comm: the communicator for MPI calculations
    /// cp2k_pes: the input CP2K PES
    /// s: the input protein system
    /// para: the global parameters for MD simulation
    /// rmse_pot: the output RMSE for potential energy
    /// rmse_force: the output RMSE for atomic force (averaging over each atom and each component)
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn md_rmse<C: Communicator, P: ProteinPES>(&self, comm: &C, cp2k_pes: &P, s: &mut ProteinSystem, para: &Para) -> (f64, f64)
    {
        // Specify the output files
        let str_output_file: PathBuf = self.output_path.join("nn_es.pdb");
        let md_output_file: PathBuf = self.output_path.join("md.out");
        let rmse_output_file: PathBuf = self.output_path.join("rmse.out");

        // Parallel processing
        let rank = comm.rank();

        // Output the initial structure to PDB output file, and output the header to MD and RMSE output file
        if rank == ROOT_RANK
        {
            s.write_pdb(&str_output_file, true, 0);
            let mut md_output = File::create(&md_output_file).expect(&error_file("creating", &md_output_file));
            md_output.write_all(b"    step            time            temp             kin       pot_nn_es         f_nn_es\n").expect(&error_file("writing", &md_output_file));
            let mut rmse_output = File::create(&rmse_output_file).expect(&error_file("creating", &rmse_output_file));
            rmse_output.write_all(b"    step       pot_nn_es        pot_cp2k         f_nn_es          f_cp2k\n").expect(&error_file("writing", &rmse_output_file));
        }

        // Initialize the long-ranged interaction parameters
        s.long_ranged_dir = self.long_ranged_dir.clone();
        s.get_fixed_long_ranged_para();
        *s = s.to_atom_fragment();



        // Define the variables for NVT MD
        let mut pot_nn_es: f64;
        let mut force_nn_es: Array2<f64>;
        let mut f_nn_es: f64;
        let mut pot_cp2k: f64;
        let mut force_cp2k: Array2<f64>;
        let mut f_cp2k: f64;

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



        // Perform NVT MD iteractively for system equilibriant
        for i in 1..(para.md_para.equi_step+1)
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

            // Calculate the atomic acceleration from nn and es atomic force
            for j in 0..s.natom
            {
                acc[[j, 0]] = force_nn_es[[j, 0]] / atom_mass[j];
                acc[[j, 1]] = force_nn_es[[j, 1]] / atom_mass[j];
                acc[[j, 2]] = force_nn_es[[j, 2]] / atom_mass[j];
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
                if (i % para.md_para.print_step) == 0
                {
                    s.write_pdb(&str_output_file, false, i);
                }
                let mut md_output = File::options().append(true).open(&md_output_file).expect(&error_file("opening", &md_output_file));
                md_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, t, temp, kin, pot_nn_es, f_nn_es).as_bytes()).expect(&error_file("writing", &md_output_file));
            }
        }



        // Perform NVT MD iteractively for data sampling
        t = 0.0;
        let mut acc_squ_pot: f64 = 0.0;
        let mut acc_squ_force: f64 = 0.0;
        let mut acc_num: usize = 0;
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

            // Calculate the atomic acceleration from nn and es atomic force
            for j in 0..s.natom
            {
                acc[[j, 0]] = force_nn_es[[j, 0]] / atom_mass[j];
                acc[[j, 1]] = force_nn_es[[j, 1]] / atom_mass[j];
                acc[[j, 2]] = force_nn_es[[j, 2]] / atom_mass[j];
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
                if (i % para.md_para.print_step) == 0
                {
                    s.write_pdb(&str_output_file, false, i);
                }
                let mut md_output = File::options().append(true).open(&md_output_file).expect(&error_file("opening", &md_output_file));
                md_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, t, temp, kin, pot_nn_es, f_nn_es).as_bytes()).expect(&error_file("writing", &md_output_file));
            }

            // Calculate the RMSE with respect to CP2K PES
            if (i % para.md_para.print_step) == 0
            {
                // Calculate the CP2K potential energy and atomic forces
                (pot_cp2k, force_cp2k) = cp2k_pes.get_energy_force(s);
                f_cp2k = (&force_cp2k * &force_cp2k).sum().sqrt();

                // Accumulate the square error
                acc_squ_pot += (pot_nn_es - pot_cp2k).powi(2);
                acc_squ_force += ( force_nn_es - force_cp2k ).mapv(|x| x.powi(2)).sum();
                acc_num += 1;

                // Output the information in this iterative step
                if rank == ROOT_RANK
                {
                    let mut rmse_output = File::options().append(true).open(&rmse_output_file).expect(&error_file("opening", &rmse_output_file));
                    rmse_output.write_all(format!("{:8} {:15.8} {:15.8} {:15.8} {:15.8}\n", i, pot_nn_es, pot_cp2k, f_nn_es, f_cp2k).as_bytes()).expect(&error_file("writing", &rmse_output_file));
                }
            }
        }



        ( (acc_squ_pot / acc_num as f64).sqrt(), (acc_squ_force / (acc_num * s.natom * 3) as f64).sqrt() )
    }
}










