//! About the output files.
use std::fs;
use crate::common::constants::ROOT_RANK;
use crate::common::error::*;
use mpi::traits::Communicator;





/// Specify the output path for the output files
///
/// # Parameters
/// ```
/// comm: the input communicator
/// index: the input index that specifies where to output the files
/// output_path: the path for the output files
/// ```
pub fn create_output_path<C: Communicator>(comm: &C, index: Option<i32>) -> String
{
    match index
    {
        // If index exists, create a directory and output the files to it
        Some(index) =>
        {
            if comm.rank() == ROOT_RANK
            {
                let dir: String = format!("{}", index);
                let dir_exist = fs::metadata(&dir);
                // If the directory already exist, do nothing; otherwise, create the directory
                match dir_exist
                {
                    Ok(_) => (),
                    Err(_) => fs::create_dir(&dir).expect(&error_dir("creating", &dir)),
                }
            }

            format!("{}/", index)
        },

        // If index non-exists, output the files to the current directory
        None =>
        {
            String::new()
        },
    }
}










