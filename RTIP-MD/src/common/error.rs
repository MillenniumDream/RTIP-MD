//! About the warning and error information when an interrupt occurs at running time.

use crate::common::constants::{Element, AminoAcid, FragmentType};





/// Error message for File reading, creating, opening, and writing.
pub fn error_file(operation: &str, filename: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem in {} the file '{}'. \n\n\n", operation, filename)
}

/// Error message for reading function
pub fn error_read(filename: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem with the input file '{}'. Please check it. \n\n\n", filename)
}

/// Error message for Directory creating
pub fn error_dir(operation: &str, dir: &str) -> String
{
    format!("\n\n\n ERROR: There is some problem in {} the directory '{}'. Maybe it already exists or you have no permission. \n\n\n", operation, dir)
}





/// Error message of illegal type for chemical element and fragment (e.g. atom, residue, and molecule)
pub fn error_type(variable: &str, illegal_type: &str) -> String
{
    format!("\n\n\n ERROR: Illegal '{}' type '{}' has read from the input file. Please check it. \n\n\n", variable, illegal_type)
}

/// Error message for illegal format of fragment (e.g. atom, residue, and molecule)
pub fn error_fragment_format(fragment_type: FragmentType, filename: &str) -> String
{
    format!("\n\n\n ERROR: Fragment {:?} in the input file '{}' has an illegal format. Please check it. \n\n\n", fragment_type, filename)
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

/// Error message for the intrafragment descriptor of amino acid
pub fn error_intrafragment_descriptor(amino_acid: &AminoAcid, n: usize) -> String
{
    format!("\n\n\n ERROR: The intrafragment descriptor of Amino_Acid '{:?}' should have '{}' values. Please check it. \n\n\n", amino_acid, n)
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





/// Error message for min_1d function
pub fn error_min_1d() -> String
{
    format!("\n\n\n ERROR: There is some problem with the function min_1d: the input fun is increasing along +x direction, or the default minimum step is too large. \n\n\n")
}










