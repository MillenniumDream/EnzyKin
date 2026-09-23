//! About the file processing
use std::fs;
use std::path::PathBuf;
use crate::common::error::*;





/// Input a series of directories, traverse them and output all the leaf directories
///
/// # Parameters
/// ```
/// dir: the input directories for traversing
/// leaf_dir: the output leaf directories
/// ```
///
/// # Examples
/// ```
/// ```
pub fn traverse_leaf_dir(dir: &Vec<PathBuf>) -> Vec<PathBuf>
{
    let mut leaf_dir: Vec<PathBuf> = Vec::new();

    // For each input directory
    for i in 0..dir.len()
    {
        let entries = fs::read_dir(&dir[i]).expect(&error_none_value("entries"));
        let mut is_leaf_dir: bool = true;
        for entry in entries
        {
            let entry_path: PathBuf = entry.expect(&error_none_value("entry")).path();
            // If the directory contains sub-directory, then it isn't a leaf directory.
            // Traverse all its sub-directories in sequence.
            if entry_path.is_dir()
            {
                leaf_dir.append(&mut traverse_leaf_dir(&vec![entry_path]));
                is_leaf_dir = false;
            }
        }
        // If the directory doesn't contain any sub-directory, then it is a leaf directory.
        // Reserve it to the leaf_dir.
        if is_leaf_dir
        {
            leaf_dir.push(dir[i].clone());
        }
    }

    leaf_dir
}










