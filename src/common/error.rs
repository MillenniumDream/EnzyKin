//! About the warning and error information when an interrupt occurs at running time.
use std::path::Path;
use std::fmt::Debug;
use crate::common::constants::Element;
use crate::nn::fragment_definition::FragmentType;





/// Error message for File reading, creating, opening, and writing.
pub fn error_file<P: AsRef<Path> + Debug>(operation: &str, filename: P) -> String
{
    format!("\n\n\n ERROR: There is some problem in {} the file '{:?}'. \n\n\n", operation, filename)
}

/// Error message for reading function
pub fn error_read<P: AsRef<Path> + Debug>(filename: P) -> String
{
    format!("\n\n\n ERROR: There is some problem with the input file '{:?}'. Please check it. \n\n\n", filename)
}

/// Error message for Directory creating
pub fn error_dir<P: AsRef<Path> + Debug>(operation: &str, dir: P) -> String
{
    format!("\n\n\n ERROR: There is some problem in {} the directory '{:?}'. Maybe it already exists or you have no permission. \n\n\n", operation, dir)
}





/// Error message of illegal type for chemical element and fragment (e.g. atom, residue, and molecule)
pub fn error_type(variable: &str, illegal_type: &str) -> String
{
    format!("\n\n\n ERROR: Illegal '{}' type '{}' has read from the input file. Please check it. \n\n\n", variable, illegal_type)
}

/// Error message for illegal biological elements (e.g. C, H, O, N, P, S, ...)
pub fn error_biological_element(illegal_biological_element: &str) -> String
{
    format!("\n\n\n ERROR: Illegal biological element '{}' has read from the input file. Please check it. \n\n\n", illegal_biological_element)
}

/// Error message for illegal format of fragment (e.g. atom, residue, and molecule)
pub fn error_fragment_format<P: AsRef<Path> + Debug>(fragment_type: FragmentType, filename: P) -> String
{
    format!("\n\n\n ERROR: Fragment {:?} in the input file '{:?}' has an illegal format. Please check it. \n\n\n", fragment_type, filename)
}

/// Error message for illegal fragment without parameterization of long-ranged interaction
pub fn error_unparameterized_fragment(fragment_name: &str) -> String
{
    format!("\n\n\n ERROR: Fragment {} hasn't been parameterized for long-ranged interaction. Please check it. \n\n\n", fragment_name)
}





/// Error message for getting value by key in static HashMap
pub fn error_static_hashmap(key: &str, value: &str, hashmap: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem in getting the value '{}' by key '{}' in the static HashMap '{}'. Please check it. \n\n\n", value, key, hashmap)
}





/// Error message in getting the specific property of the object
pub fn error_getting_property(property: &str, object: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem in getting the {} of {}. Please check it. \n\n\n", property, object)
}





/// Error message for CString::new() 
pub fn error_str_to_cstring(str_name: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem in transforming str '{}' to CString. \n\n\n", str_name)
}

/// Error message for try_into()
pub fn error_type_transformation(variable: &str, type1: &str, type2: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem in type transformation of '{}' from {} to {}. \n\n\n", variable, type1, type2)
}

/// Error message for as_slice() and as_slice_mut()
pub fn error_as_slice(variable: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem in getting the slice of the variable '{}'. \n\n\n", variable)
}





/// Error message for `Some<A>`, Result<T, E>
pub fn error_none_value(variable: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem with variable '{}', which has none/wrong value. \n\n\n", variable)
}

/// Error message for cloned()
pub fn error_cloning(variable: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem in cloning the variable '{}'. \n\n\n", variable)
}





/// Error message for unestablished neural network (corresponding to the non-bioelements)
pub fn error_non_bioelement_nn(element: &Element) -> String
{
    format!("\n\n\n ERROR: The neural network of Element '{:?}' has't been established yet, since it's not the bioelement. Please check it. \n\n\n", element)
}

/// Error message for global data size in NN training
pub fn error_global_data_size() -> String
{
    format!("\n\n\n ERROR: In NN training, there should have at least two data for a specific protein system, one for training set, the other for validation set. Please check it. \n\n\n")
}

/// Error message for parameters updating in NN training
pub fn error_nn_para_update(fragment: &str, optimizer: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem in updating the parameters of sub-NN of '{}' using the optimizer '{}'. Please check it. \n\n\n", fragment, optimizer)
}





/// Error message for general CV mismatched
pub fn error_general_cv_mismatched() -> String
{
    format!("\n\n\n ERROR: The initial state is mismatched (of different numbers of atoms or different atomic types) with the final state. Please check it. \n\n\n")
}

/// Error message for general CV coincided
pub fn error_general_cv_coincided() -> String
{
    format!("\n\n\n ERROR: The CV values of initial state and final state are coincided. Please check it. \n\n\n")
}





/// Error message for Bond CV
pub fn error_combined_bond_cv() -> String
{
    format!("\n\n\n ERROR: The indices of the two bonded atoms are exceeded the range. Please check it. \n\n\n")
}





/// Error message for the reference path in construction of projected path CV
pub fn error_ref_path() -> String
{
    format!("\n\n\n ERROR: There are at least two structures of the same number of atoms and the same atomic types for the construction of reference path. Please check it. \n\n\n")
}

/// Error message in calculation of projected path CV
pub fn error_ppcv() -> String
{
    format!("\n\n\n ERROR: The input structure is mismatched (of different numbers of atoms) with the reference path. Please check it. \n\n\n")
}





/// Error message for min_1d function
pub fn error_min_1d() -> String
{
    format!("\n\n\n ERROR: There is some problem with the function min_1d: the input fun is increasing along +x direction, or the default minimum step is too large. \n\n\n")
}










