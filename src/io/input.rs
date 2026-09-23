//! About the input files.
use std::path::PathBuf;
use crate::common::constants::Element;
use crate::nn::global_nn::NnUpdate;





/// The structure containing the parameters of the roto-translationally invariant potential (RTIP).
///
/// # Fields
/// ```
/// a0: specifying how quickly the Gaussian potential is increasing (Unit: Hartree)
/// a_max: the maximum height of the Gaussian potential (Unit: Hartree)
/// sigma: specifying the width of the Gaussian potential (Unit: bohr)
/// scale_ts_a0: scaling factor of a0 for ts with respect to local_min
/// scale_ts_sigma: scaling factor of sigma for ts with respect to local_min
/// ```
#[derive(Clone)]
pub struct RtipPara
{
    pub a0: f64,
    pub a_max: f64,
    pub sigma: f64,
    pub scale_ts_a0: f64,
    pub scale_ts_sigma: Option<f64>,
}





/// The structure containing the parameters of the pathway sampling
///
/// # Fields
/// ```
/// pot_climb: when the real potential energy (i.e. e_cp2k) becomes larger than (e_min + pot_climb), stop this failed pathway samping and initiate a new one (Unit: Hartree)
/// pot_drop: when the real potential energy (i.e. e_cp2k) becomes smaller than (e_max - e_drop), remove the Gaussian potential and perform a local optimization (Unit: Hartree)
/// pot_epsilon: in each steeping searching, when the difference of two adjacent e_total (i.e. e_cp2k + e_rtip) becomes smaller than e_epsilon, stop the searching (Unit: Hartree/atom)
/// f_epsilon: in local optimization without Gaussian potential, when force_rtip becomes smaller than force_epsilon, stop the optimization (Unit: Hartree/(Bohr*atom))
/// max_step: maximum step of the pathway sampling
/// print_step: specifying how many steps to output a structure
/// ```
#[derive(Clone)]
pub struct PathSampPara
{
    pub pot_climb: f64,
    pub pot_drop: f64,
    pub pot_epsilon: f64,
    pub f_epsilon: f64,
    pub max_step: usize,
    pub print_step: usize,
}





/// The structure containing the parameters of thermostat
///
/// # Fields
/// ```
/// temp_bath: temperature of the bath (Unit: K)
/// tau: temperature relaxation time in Berendsen or Bussi thermostat (Unit: fs)
/// gamma: friction coefficient in Langevin thermostat (Unit: fs-1)
/// ```
#[derive(Clone)]
pub struct Thermostat
{
    pub temp_bath: f64,
    pub tau: f64,
    pub gamma: f64,
}

/// The structure containing the parameters of extended MD
///
/// # Fields
/// ```
/// sigma: thermal width of the coupling (Unit: Å)
/// tau: period of the oscillator (Unit: fs)
/// boundary: add a boundary (harmonic potential) to the extended degree of freedom or not.
/// boundary: if add a boundary, provide the multiplier of the elastic coefficient.
/// ```
#[derive(Clone)]
pub struct ExtendedMDPara
{
    pub sigma: f64,
    pub tau: f64,
    pub boundary: Option<f64>,
}

/// The structure containing the parameters of eABF-ATF
///
/// # Fields
/// ```
/// cv_bin: the reference CV bin in eABF-ATF (the actual CV bin may have a little deviation as restrainted by the CV bound, Unit: Å)
/// cv_ext: extended CV in PPCV (Unit: Å)
/// num_half_full: half of number of full sample for the biasing force
/// gamma: control parameter of the thermoleveling force
/// ```
#[derive(Clone)]
pub struct EABFATFPara
{
    pub cv_bin: f64,
    pub cv_ext: f64,
    pub num_half_full: usize,
    pub gamma: f64,
}

/// The structure containing the parameters of well-tempered metadynamics
///
/// # Fields
/// ```
/// height: height of the initial Gaussian potential (Unit: Hartree)
/// sigma: standard deviation of all the Gaussian potentials (Unit: Å)
/// bias_factor: bias factor in well-tempered metadynamics, i.e., (T + delta_T) / T
/// pace_step: specifying how many steps to add a Gaussian
/// ```
#[derive(Clone)]
pub struct WTMPara
{
    pub height: f64,
    pub sigma: f64,
    pub bias_factor: f64,
    pub pace_step: usize,
}

/// The structure containing the parameters of evolution MD
///
/// # Fields
/// ```
/// decreasing_multiple: multiple of the decreasing rate with respect to the increasing rate
/// decreasing_bound: lower bound for the decreasing of RTIP (Unit: Hartree)
/// falling_multiple: multiple of the falling rate with respect to the increasing rate
/// ignored_pair: ignoring these pairs of elements in bonding judgment
/// split_step: specifying how many step to split the molecules
/// circulate: circulating the simulation or not
/// ```
#[derive(Clone)]
pub struct EvolMdPara
{
    pub decreasing_multiple: f64,
    pub decreasing_bound: f64,
    pub falling_multiple: f64,
    pub ignored_pair: Vec<(Element, Element)>,
    pub split_step: Option<usize>,
    pub circulate: bool,
}

/// The structure containing the parameters of deviated sampling
///
/// # Fields
/// ```
/// pot_recorded_epsilon: when the potential difference of NN and real PES is larger than this epsilon, record this structure (Unit: Hartree)
/// f_recorded_epsilon: when the force difference of NN and real PES is larger than this epsilon, record this structure (Unit: A.U.)
/// pot_replaced_epsilon: when the potential difference of NN and real PES is larger than this epsilon, replace NN with real PES and record this structure (Unit: Hartree)
/// f_replaced_epsilon: when the force difference of NN and real PES is larger than this epsilon, replace NN with real PES and record this structure (Unit: A.U.)
/// ```
#[derive(Clone)]
pub struct DevMdPara
{
    pub pot_recorded_epsilon: f64,
    pub f_recorded_epsilon: f64,
    pub pot_replaced_epsilon: f64,
    pub f_replaced_epsilon: f64,
}

/// The structure containing the parameters of MD simulations
///
/// # Fields
/// ```
/// dt: time step (fs)
/// equi_step: equilibrium step
/// max_step: maximum step
/// print_step: specifying how many steps to output a structure
/// remove_tran_step: specifying how many steps to remove the overall translational speed of the mass center
/// ```
#[derive(Clone)]
pub struct MdPara
{
    pub dt: f64,
    pub equi_step: usize,
    pub max_step: usize,
    pub print_step: usize,
    pub remove_tran_step: usize,
    pub thermostat: Thermostat,
    pub extended_md_para: ExtendedMDPara,
    pub eabf_atf_para: EABFATFPara,
    pub wtm_para: WTMPara,
    pub evol_md_para: EvolMdPara,
    pub dev_md_para: DevMdPara,
}





/// The structure containing the parameters of neural netword
///
/// # Fields
/// ```
/// lr: learning rate of NN
/// training_batch_size: batch size in training
/// validation_batch_size: batch size in validation
/// max_step: maximum step of training
/// print_step: specifying how many steps to output a training NN potential
/// input_nn_dir: specifying the directory of the input NN potential
/// output_nn_dir: specifying the directory for the output NN potential
/// input_data_dir: specifying the directories of the training data set
/// norm_dir: specifying the directory of the normalization parameters
/// long_ranged_dir: specifying the directory of the long ranged parameters
/// ```
#[derive(Clone)]
pub struct NnTrainPara
{
    pub lr: f64,
    pub training_batch_size: usize,
    pub validation_batch_size: usize,
    pub max_step: usize,
    pub print_step: usize,
    pub input_nn_dir: PathBuf,
    pub output_nn_dir: PathBuf,
    pub input_data_dir: Vec<PathBuf>,
    pub norm_dir: PathBuf,
    pub long_ranged_dir: Option<PathBuf>,
    pub nn_update: NnUpdate,
}










#[derive(Clone)]
pub struct Para
{
    // RTIP parameters
    pub rtip_para: RtipPara,

    // Pathway sampling parameters
    pub path_samp_para: PathSampPara,

    // MD parameters
    pub md_para: MdPara,

    // NN training parameters
    pub nn_train_para: NnTrainPara,
}










impl Para
{
    pub fn new() -> Self
    {
        Para
        {
            // RTIP parameters
            rtip_para: RtipPara
            {
                a0: 0.000003,
                a_max: 0.03,
                sigma: 3.0,
                scale_ts_a0: 1.0,
                scale_ts_sigma: Some(0.25),
            },

            // Pathway sampling parameters
            path_samp_para: PathSampPara
            {
                pot_climb: 0.185,
                pot_drop: 0.02,
                pot_epsilon: 0.00005,
                f_epsilon: 0.001,
                max_step: 2000,
                print_step: 1,
            },

            // MD parameters
            md_para: MdPara
            {
                dt: 1.0,
                equi_step: 1000,
                max_step: 100000000,
                print_step: 10,
                remove_tran_step: 100,
                thermostat: Thermostat
                {
                    temp_bath: 300.0,
                    tau: 10.0,
                    gamma: 0.01,
                },
                extended_md_para: ExtendedMDPara
                {
                    sigma: 0.005,
                    tau: 50.0,
                    boundary: Some(10.0),
                },
                eabf_atf_para: EABFATFPara
                {
                    cv_bin: 0.05,
                    cv_ext: 0.05,
                    num_half_full: 100,
                    gamma: 4.0,
                },
                wtm_para: WTMPara
                {
                    height: 0.005,
                    sigma: 0.15,
                    bias_factor: 100.0,
                    pace_step: 100,
                },
                evol_md_para: EvolMdPara
                {
                    decreasing_multiple: 5.0,
                    decreasing_bound: 0.0,
                    falling_multiple: 0.0,
                    ignored_pair: Vec::new(),
                    split_step: None,
                    circulate: false,
                },
                dev_md_para: DevMdPara
                {
                    pot_recorded_epsilon: 0.000,
                    f_recorded_epsilon: 0.000,
                    pot_replaced_epsilon: 10.00,
                    f_replaced_epsilon: 10.00,
                },
            },

            // NN training parameters
            nn_train_para: NnTrainPara
            {
                lr: 0.000001,
                training_batch_size: 64,
                validation_batch_size: 8,
                max_step: 1000000,
                print_step: 10000,
                input_nn_dir: PathBuf::from("nn/0"),
                output_nn_dir: PathBuf::from("nn/gly"),
                input_data_dir: vec![PathBuf::from("data/gly")],
                norm_dir: PathBuf::from("norm"),
                long_ranged_dir: None,
                nn_update: NnUpdate
                {
                    update_element_h_nn: false,
                    update_element_b_nn: false,
                    update_element_c_nn: false,
                    update_element_n_nn: false,
                    update_element_o_nn: false,
                    update_element_f_nn: false,
                    update_element_na_nn: false,
                    update_element_mg_nn: false,
                    update_element_si_nn: false,
                    update_element_p_nn: false,
                    update_element_s_nn: false,
                    update_element_cl_nn: false,
                    update_element_k_nn: false,
                    update_element_ca_nn: false,
                    update_element_v_nn: false,
                    update_element_cr_nn: false,
                    update_element_mn_nn: false,
                    update_element_fe_nn: false,
                    update_element_co_nn: false,
                    update_element_ni_nn: false,
                    update_element_cu_nn: false,
                    update_element_zn_nn: false,
                    update_element_as_nn: false,
                    update_element_se_nn: false,
                    update_element_br_nn: false,
                    update_element_mo_nn: false,
                    update_element_cd_nn: false,
                    update_element_sn_nn: false,
                    update_element_i_nn: false,

                    update_amino_acid_gly_nn: true,
                    update_amino_acid_ala_nn: false,
                    update_amino_acid_val_nn: false,
                    update_amino_acid_leu_nn: false,
                    update_amino_acid_ile_nn: false,
                    update_amino_acid_ser_nn: false,
                    update_amino_acid_thr_nn: false,
                    update_amino_acid_asp_nn: false,
                    update_amino_acid_ash_nn: false,
                    update_amino_acid_asn_nn: false,
                    update_amino_acid_glu_nn: false,
                    update_amino_acid_glh_nn: false,
                    update_amino_acid_gln_nn: false,
                    update_amino_acid_lys_nn: false,
                    update_amino_acid_lyn_nn: false,
                    update_amino_acid_arg_nn: false,
//                    update_amino_acid_arn_nn: false,
                    update_amino_acid_cys_nn: false,
                    update_amino_acid_cyx_nn: false,
//                    update_amino_acid_cym_nn: false,
                    update_amino_acid_met_nn: false,
                    update_amino_acid_hid_nn: false,
                    update_amino_acid_hie_nn: false,
                    update_amino_acid_hip_nn: false,
                    update_amino_acid_phe_nn: false,
                    update_amino_acid_tyr_nn: false,
                    update_amino_acid_trp_nn: false,
                    update_amino_acid_pro_nn: false,

                    update_head_h_nn: false,
                    update_tail_o_nn: false,
                    update_tail_h_nn: false,
                    update_head_ace_nn: true,
                    update_tail_nme_nn: true,
                    update_tail_nhe_nn: false,

                    update_molecule_wat_nn: false,
                },
            },
        }
    }
}










