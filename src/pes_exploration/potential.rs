use crate::pes_exploration::system::System;
use ndarray::Array2;





/// The structure containing all the information about the Gaussian repulsive potential,
/// which is introduced to push the system out of the local minimum along an unknown pathway.
///
/// # Fields
/// ```
/// local_min: the local minimum structure
/// nearby_ts: the known nearby transition state (TS) structures
/// ```
pub struct RepulsivePot
{
    pub local_min: System,
    pub nearby_ts: Vec<System>,
}





/// The structure containing all the information about the Gaussian attractive potential,
/// which is introduced to draft the system towards the final state.
///
/// # Fields
/// ```
/// initial_state: the current system to be drafted
/// final_state: the objective system to be drafted towards
/// ```
pub struct AttractivePot
{
    pub initial_state: System,
    pub final_state: System,
}





/// The structure containing all the information about the synthesis potential,
/// which is introduced to synthesize a product from several molecules.
///
/// # Fields
/// ```
/// initial_state: the initial system containing several separated molecules
/// mol_index: atomic index of the molecules for synthesis
/// ```
pub struct SynthesisPot
{
    pub initial_state: System,
    pub mol_index: Vec<Vec<usize>>,
}





/// The structure containing all the information about the evolution potential,
/// which is introduced to simulate the evolution process of a series of original molecules.
///
/// # Fields
/// ```
/// initial_state: the initial system containing a series of original molecules
/// ```
pub struct EvolutionPot
{
    pub initial_state: System,
    pub initial_velocity: Option<Array2<f64>>,
}





/// The structure containing all the information about the distributed potential,
/// which is introduced to circumvent the visited configurations.
///
/// # Fields
/// ```
/// visited_states: the already visited structures for circumvention
/// ```
pub struct DistributedPot
{
    pub visited_states: Vec<System>,
    pub a: Vec<f64>,
    pub sigma: Vec<f64>,
}










