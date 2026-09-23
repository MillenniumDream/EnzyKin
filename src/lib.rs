//! RTIP
//!
//! RTIP (roto-translational invariant potential) is a biased potential appended to the real PES
//! (potential energy surface) to drive the molucule (aperiodic structure) escaping from local minimum
//! along the most flat directions.It aims at intelligent pathway searching for chemical and biological
//! reactions, like protein folding and enzyme catalysis.

#![recursion_limit = "256"]

//extern crate libc;
//extern crate mpi;

pub mod common;
pub mod io;
pub mod math;
pub mod pes_exploration;
pub mod md;
pub mod pot;
pub mod nn;
pub mod long_ranged;
pub mod external;





#[no_mangle]
pub extern fn main() -> i32
{
//    println!("Begin Initialization");
    use crate::external::cp2k::*;
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    cp2k_init_without_mpi();





// Model Potential
/*
    use crate::io::input::Para;
    use crate::io::output;
    use crate::pes_exploration::system::System;
    use crate::pot::linear_pot::*;
    use crate::md::cv::combined_bond_cv::CombinedBondCV;
    use crate::md::eabf_atf::eabf_atf_1d;
    use std::path::PathBuf;
    use mpi::topology::{Color, Communicator};

    let comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let output_path: PathBuf = output::create_output_path(&comm, Some(world.rank() % 1));
    let linear_pot: LinearPot = LinearPot::new();
    let para: Para = Para::new();

    let is: System = System::read_xyz("two_atom_is.xyz");
    let fs: System = System::read_xyz("two_atom_fs.xyz");
    let combined_bond_cv: CombinedBondCV = CombinedBondCV::new(vec![(1.0, 0, 1)], is, fs);

    eabf_atf_1d(&comm, &linear_pot, &combined_bond_cv, &para, &output_path);
*/






// LJ38
/*
    use crate::io::input::Para;
    use crate::io::output;
    use crate::pes_exploration::system::System;
    use crate::pot::lj_pot::*;
    use crate::md::cv::bond_orientational_cv::BondOrientationalCV;
    use crate::md::eabf_atf::eabf_atf_1d;
    use std::path::PathBuf;
    use mpi::topology::{Color, Communicator};

    let comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let output_path: PathBuf = output::create_output_path(&comm, Some(world.rank() % 1));
    let lj_pot: LJPot = LJPot::new();
    let para: Para = Para::new();

    let is: System = System::read_xyz("LJ38_Oh.xyz");
    let fs: System = System::read_xyz("LJ38_C5v.xyz");
    let bond_orientational_cv: BondOrientationalCV = BondOrientationalCV::new(6, 1.2*SIGMA, 1.6*SIGMA, is, fs);

    eabf_atf_1d(&comm, &lj_pot, &bond_orientational_cv, &para, &output_path);
*/







// Emzyme Catalysis
/*
    use crate::io::input::Para;
    use crate::io::output;
    use crate::pes_exploration::system::System;
    use crate::md::cv::combined_bond_cv::CombinedBondCV;
    use crate::md::eabf_atf::eabf_atf_1d;
    use std::path::PathBuf;
    use mpi::topology::{Color, Communicator};


    let comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let output_path: PathBuf = output::create_output_path(&comm, Some(world.rank() % 1));
    let cp2k_output_file: String = output_path.join("cp2k_xTB.out").into_os_string().into_string().unwrap();
    let cp2k_pes: Cp2kPES = Cp2kPES::new("xTB.inp", &cp2k_output_file, 4);
//    let cp2k_output_file: String = output_path.join("cp2k_B97-3C.out").into_os_string().into_string().unwrap();
//    let cp2k_pes: Cp2kPES = Cp2kPES::new("B97-3C.inp", &cp2k_output_file, 4);
    let para: Para = Para::new();


    let is: System = System::read_xyz("IS.xyz");
    let fs: System = System::read_xyz("FS.xyz");
    let combined_bond_cv: CombinedBondCV = CombinedBondCV::new(vec![(1.0, 5669, 5670), (-1.0, 5662, 5674)], is, fs);

    eabf_atf_1d(&comm, &cp2k_pes, &combined_bond_cv, &para, &output_path);
*/











// C4H8
    use crate::io::input::Para;
    use crate::io::output;
    use crate::pes_exploration::system::System;
    use crate::md::cv::combined_bond_cv::CombinedBondCV;
    use crate::md::eabf_atf::eabf_atf_1d;
    use std::path::PathBuf;
    use mpi::topology::{Color, Communicator};


    let comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let output_path: PathBuf = output::create_output_path(&comm, Some(world.rank() % 1));
//    let cp2k_output_file: String = output_path.join("cp2k_xTB.out").into_os_string().into_string().unwrap();
//    let cp2k_pes: Cp2kPES = Cp2kPES::new("xTB.inp", &cp2k_output_file, 4);
    let cp2k_output_file: String = output_path.join("cp2k_B97-3C.out").into_os_string().into_string().unwrap();
    let cp2k_pes: Cp2kPES = Cp2kPES::new("B97-3C.inp", &cp2k_output_file, 4);
    let para: Para = Para::new();


    let is: System = System::read_xyz("IS.xyz");
    let fs: System = System::read_xyz("FS.xyz");
    let combined_bond_cv: CombinedBondCV = CombinedBondCV::new(vec![(1.0, 0, 1)], is, fs);

    eabf_atf_1d(&comm, &cp2k_pes, &combined_bond_cv, &para, &output_path);







/*
{
    use crate::io::input::Para;
    use crate::io::output;
    use std::path::PathBuf;
    use mpi::topology::{Color, Communicator};
    use crate::pes_exploration::system::System;
    use crate::md::projected_path_cv::PPCV;
    use crate::md::eabf_atf::*;


    let comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let output_path: PathBuf = output::create_output_path(&comm, Some(world.rank() % 1));
    let cp2k_xtb_output_file: String = output_path.join("cp2k_xTB.out").into_os_string().into_string().unwrap();
    let cp2k_xtb_pes: Cp2kPES = Cp2kPES::new("xTB.inp", &cp2k_xtb_output_file, 4);
    let para: Para = Para::new();


    let mut path: Vec<System> = Vec::with_capacity(7);
    let mut is: System = System::read_xyz("IS.xyz");
    is.atom_add_pot = Some(vec![0, 1, 2, 3]);
    path.push(is);
    path.push(System::read_xyz("IM1.xyz"));
    path.push(System::read_xyz("IM2.xyz"));
    path.push(System::read_xyz("IM3.xyz"));
    path.push(System::read_xyz("IM4.xyz"));
    path.push(System::read_xyz("IM5.xyz"));
    path.push(System::read_xyz("FS.xyz"));
    let ppcv: PPCV = PPCV::from_system(&path, 10.0);


    eabf_atf_1d(&comm, &cp2k_xtb_pes, &ppcv, &para, &output_path);
}
*/










/*
    use crate::common::traits::ColVar;
    use crate::pes_exploration::system::System;
    use crate::md::projected_path_cv::PPCV;
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;


    let mut path: Vec<System> = Vec::with_capacity(8);
    let mut is: System = System::read_xyz("IS.xyz");
    is.atom_add_pot = Some(vec![0, 1, 2, 5, 6]);
    path.push(is);
    path.push(System::read_xyz("IM1.xyz"));
    path.push(System::read_xyz("IM2.xyz"));
    path.push(System::read_xyz("IM3.xyz"));
    path.push(System::read_xyz("IM4.xyz"));
    path.push(System::read_xyz("IM5.xyz"));
    path.push(System::read_xyz("IM6.xyz"));
    path.push(System::read_xyz("FS.xyz"));
    let lambda: f64 = 100.0;
    let ppcv: PPCV = PPCV::from_system(&path, lambda);


    let mut s: System = System::read_xyz("IS.xyz");
    s.coord = Array2::random((s.natom, 3), Uniform::new(0.0, 10.0));
    let (_cv, grads): (f64, Array2<f64>) = ppcv.get_cv_grads(&s);


    let eps: f64 = 0.00001;
    let mut grads_diff: Array2<f64> = Array2::zeros(s.coord.raw_dim());
    for i in 0..s.natom
    {
        for j in 0..3
        {
            let mut ss: System = s.clone();
            ss.coord[[i,j]] -= eps;
            let cv1: f64 = ppcv.get_cv(&ss);

            let mut ss: System = s.clone();
            ss.coord[[i,j]] += eps;
            let cv2: f64 = ppcv.get_cv(&ss);

            grads_diff[[i,j]] = (cv2 - cv1) / (2.0 * eps);
        }
    }


    let diff: Array2<f64> = &grads_diff - &grads;
    let diff1: Array2<f64> = &diff / &grads;
    let mut diff1_max: f64 = 0.0;
    for i in 0..s.natom
    {
        for j in 0..3
        {
            if diff1[[i,j]].abs() > diff1_max
            {
                diff1_max = diff1[[i,j]].abs();
            }
        }
    }


    dbg!(&grads);
    dbg!(&grads_diff);
    dbg!(&diff);
    dbg!(&diff1);
    dbg!(diff1_max);
*/








/*
    // Test Langevin MD
{
    use crate::io::input::Para;
    use crate::io::output;
    use std::path::PathBuf;
    use mpi::topology::{Color, Communicator};


    let comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let output_path: PathBuf = output::create_output_path(&comm, Some(world.rank() % 1));
    let cp2k_xtb_output_file: String = output_path.join("cp2k_xTB.out").into_os_string().into_string().unwrap();
    let cp2k_xtb_pes: Cp2kPES = Cp2kPES::new("xTB.inp", &cp2k_xtb_output_file, 4);
    let para: Para = Para::new();


    use crate::common::traits::*;
//    use crate::nn::protein::ProteinSystem;
    use crate::pes_exploration::system::System;
    let s: System = System::read_pdb("str.pdb");
    s.langevin_nvt_md(&comm, &cp2k_xtb_pes, &para, &output_path);
}
*/









/*
    // Test vel_init()
    use crate::nn::protein::ProteinSystem;
    use crate::pes_exploration::system::System;
    use crate::md::initialization::vel_init;

    let s: ProteinSystem = ProteinSystem::read_pdb("aaa.pdb");
    let s: System = s.to_system();

    let mut atom_mass: Vec<f64> = Vec::with_capacity(s.natom);
    for i in 0..s.natom
    {
        atom_mass.push( s.atom_type.as_ref().unwrap()[i].get_atomic_mass() );
    }

    vel_init(300.0, &atom_mass);
*/











/*
    use crate::long_ranged::global_para::GlobalPara;

    let global_para: GlobalPara = GlobalPara::initialize();
    let mut charge: f64 = 0.0;
    for i in 0..global_para.amino_acid_hip_para.len()
    {
        charge += global_para.amino_acid_hip_para[i].charge;
    }

    dbg!(charge);
*/











/*
    use ndarray::{Array1, Array2};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use crate::pes_exploration::rtip::*;

    let coord: Array2<f64> = Array2::random((10, 3), Uniform::new(0.0, 1.0));
    let coord1: Array2<f64> = Array2::random((10, 3), Uniform::new(0.0, 1.0));

    let dist1: f64 = rti_dist(&coord1, &coord);
    let (rot, tran): (Array2<f64>, Array1<f64>) = rti_rot_tran(&coord1, &coord);
    let coord_1: Array2<f64> = coord1.dot(&rot) + tran;
    let mut dist_1: f64 = 0.0;
    for i in 0..10
    {
        for j in 0..3
        {
            dist_1 += (coord[[i, j]] - coord_1[[i, j]]).powi(2);
        }
    }
    dist_1 = dist_1.sqrt();

    let coord2: Array2<f64> = &coord * 0.9 + &coord_1 * 0.1;
    let dist2: f64 = rti_dist(&coord2, &coord);
    let (rot, tran): (Array2<f64>, Array1<f64>) = rti_rot_tran(&coord2, &coord);

    dbg!(dist1);
    dbg!(dist_1);
    dbg!(dist2);
    dbg!(&rot);
    dbg!(&tran);
*/













/*
    use ndarray::{Array1, Array2};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use crate::pes_exploration::rtip::*;

    let coord: Array2<f64> = Array2::random((10, 3), Uniform::new(0.0, 1.0));
    let coord1: Array2<f64> = Array2::random((10, 3), Uniform::new(0.0, 1.0));
    let coord2: Array2<f64> = Array2::random((10, 3), Uniform::new(0.0, 1.0));

    let (rot1, tran1): (Array2<f64>, Array1<f64>) = rti_rot_tran(&coord1, &coord);
    let (rot2, tran2): (Array2<f64>, Array1<f64>) = rti_rot_tran(&coord2, &coord);

    let coord11: Array2<f64> = coord1.dot(&rot1) + tran1;
    let coord22: Array2<f64> = coord2.dot(&rot2) + tran2;

    let dist1: f64 = rti_dist(&coord1, &coord);
    let dist2: f64 = rti_dist(&coord2, &coord);

    let dist11: f64 = ( (&coord11 - &coord) * (&coord11 - &coord) ).sum().sqrt();
    let dist22: f64 = ( (&coord22 - &coord) * (&coord22 - &coord) ).sum().sqrt();

    let (rot, tran): (Array2<f64>, Array1<f64>) = rti_rot_tran(&coord11, &coord22);

    dbg!((dist1, dist11));
    dbg!((dist2, dist22));
    dbg!(rot);
    dbg!(tran);

 //   let (rot, tran): (Array2<f64>, Array1<f64>) = rti_rot_tran(&coord11, &coord);
    let diff: Array2<f64> = &coord11 - &coord;
    let coord111: Array2<f64> = &coord + 2.0 * &diff;
    let (rot, tran): (Array2<f64>, Array1<f64>) = rti_rot_tran(&coord111, &coord);

    dbg!(rot);
    dbg!(tran);
*/












/*
    // RTIP-MD sampling
    use std::path::PathBuf;
    use crate::io::input::Para;
    use crate::io::output;
    use crate::common::traits::*;
    use crate::pes_exploration::system::System;
    use crate::pes_exploration::potential::*;
    use crate::nn::protein::ProteinSystem;
    use mpi::topology::{Color, Communicator};

    let my_comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let output_path: PathBuf = output::create_output_path(&my_comm, Some(world.rank() % 1));
//    let cp2k_xtb_output_file: String = output_path.join("cp2k_xTB.out").into_os_string().into_string().unwrap();
    let cp2k_b973c_output_file: String = output_path.join("cp2k_B97-3C.out").into_os_string().into_string().unwrap();

    let s: ProteinSystem = ProteinSystem::read_pdb("str.pdb");
    let s: System = s.to_system();
    let para: Para = Para::new();

    let distribute_pot = DistributedPot
    {
        visited_states: vec![s],
        a: vec![0.0],
        sigma: vec![para.rtip_para.sigma],
    };

    {
//        let cp2k_xtb_pes = Cp2kPES::new("xTB.inp", &cp2k_xtb_output_file, 4);
        let cp2k_b973c_pes = Cp2kPES::new("B97-3C.inp", &cp2k_b973c_output_file, 4);
        distribute_pot.bussi_nvt_md(&my_comm, &cp2k_b973c_pes, &para, &output_path);
    }
*/











/*
    // Deviated Sampling
    use std::path::PathBuf;
    use crate::io::output;
    use crate::io::input::Para;
    use crate::common::constants::Device;
    use crate::nn::protein::ProteinSystem;
    use crate::nn::global_nn::GlobalNN;
    use crate::nn::normalization::GlobalNorm;
    use crate::nn::nn_potential::NNPot;
    use crate::external::cp2k::{cp2k_init_without_mpi, Cp2kPES};
    use mpi::topology::{Color, Communicator};

    let my_comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let output_path: PathBuf = output::create_output_path(&my_comm, Some(world.rank() % 1));
    let cp2k_xtb_output_file: String = output_path.join("cp2k_xTB.out").into_os_string().into_string().unwrap();
    let cp2k_xtb_pes: Cp2kPES = Cp2kPES::new("xTB.inp", &cp2k_xtb_output_file, 4);
//    let cp2k_b973c_output_file: String = output_path.join("cp2k_B97-3C.out").into_os_string().into_string().unwrap();
//    let cp2k_b973c_pes: Cp2kPES = Cp2kPES::new("B97-3C.inp", &cp2k_b973c_output_file, 4);

    let dev: Device = Device::seed_from_u64(1314);
    let global_nn: GlobalNN = GlobalNN::load("/data/lixt/Rust/NN/nn");
    let global_norm: GlobalNorm = GlobalNorm::load("/data/lixt/Rust/NN/norm");
    let mut s: ProteinSystem = ProteinSystem::read_pdb("str.pdb");
    let para: Para = Para::new();

    let nn_pot: NNPot = NNPot
    {
        dev,
        global_nn,
        global_norm,
        long_ranged_dir: None,
        output_path,
    };

    nn_pot.md_deviated_samping(&my_comm, &cp2k_xtb_pes, &mut s, &para);
*/






/*
    use std::path::PathBuf;
    use crate::nn::normalization::*;

    let global_norm: GlobalNorm = GlobalNorm::from_data(&vec![PathBuf::from("data/gly")]);
    dbg!(&global_norm.amino_acid_gly_norm);
    global_norm.save("norm");
*/






/*
    use crate::nn::global_nn::GlobalNN;
    let global_nn: GlobalNN = GlobalNN::new();

    global_nn.save("nn/0");
*/








/*
    use crate::io::input::Para;
    use crate::nn::training::training_on_multiple_data;

    let para: Para = Para::new();
    training_on_multiple_data(&para);
*/







/*
    // MD RMSE
    use std::path::PathBuf;
    use crate::io::output;
    use crate::io::input::Para;
    use crate::common::constants::Device;
    use crate::nn::protein::ProteinSystem;
    use crate::nn::global_nn::GlobalNN;
    use crate::nn::normalization::GlobalNorm;
    use crate::nn::nn_potential::NNPot;
    use crate::external::cp2k::{cp2k_init_without_mpi, Cp2kPES};
    use mpi::topology::{Color, Communicator};

    let my_comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let output_path: PathBuf = output::create_output_path(&my_comm, Some(world.rank() % 1));
    let cp2k_xtb_output_file: String = output_path.join("cp2k_xTB.out").into_os_string().into_string().unwrap();
    let cp2k_xtb_pes: Cp2kPES = Cp2kPES::new("xTB.inp", &cp2k_xtb_output_file, 4);
//    let cp2k_b973c_output_file: String = output_path.join("cp2k_B97-3C.out").into_os_string().into_string().unwrap();
//    let cp2k_b973c_pes: Cp2kPES = Cp2kPES::new("B97-3C.inp", &cp2k_b973c_output_file, 4);

    let dev: Device = Device::seed_from_u64(1314);
    let global_nn: GlobalNN = GlobalNN::load("nn");
    let global_norm: GlobalNorm = GlobalNorm::load("norm");
    let mut s: ProteinSystem = ProteinSystem::read_pdb("str.pdb");
    let para: Para = Para::new();

    let nn_pot: NNPot = NNPot
    {
        dev,
        global_nn,
        global_norm,
        long_ranged_dir: None,
        output_path,
    };

    let (rmse_pot, rmse_force): (f64, f64) = nn_pot.md_rmse(&my_comm, &cp2k_xtb_pes, &mut s, &para);
//    let (rmse_pot, rmse_force): (f64, f64) = nn_pot.data_set_rmse("monomer/ace_gly_nme");
    dbg!((rmse_pot, rmse_force));
*/








    cp2k_finalize_without_mpi();
//    println!("Finalization done");
    return 0;
}










