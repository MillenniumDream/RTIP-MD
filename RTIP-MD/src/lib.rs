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
pub mod matrix;
pub mod pes_exploration;
pub mod nn;
pub mod external;

//use mpi::traits::*;
use crate::external::cp2k::*;
use crate::pes_exploration::system::System;
use crate::pes_exploration::potential::*;
//use crate::common::constants::*;
//use crate::rtip::*;
use crate::io::input::Para;
use crate::io::output;
use crate::pes_exploration::traits::*;
//use crate::pes_exploration::synthesis::*;
//use crate::nn::protein::ProteinSystem;
//use mpi::topology::UserCommunicator;
use mpi::topology::{Color, Communicator};
use mpi::traits::*;
use ndarray::prelude::*;
//use ndarray_rand::RandomExt;
//use ndarray_rand::rand_distr::Uniform;
//use ndarray_linalg::{SVD, Determinant, EighInto, UPLO};
//use std::ffi::CString;
//use crate::common::constants::FragmentType;
use crate::common::constants::{Device, ROOT_RANK};
use crate::common::error::*;
//use crate::nn::intrafragment_descriptor::*;
//use crate::nn::interfragment_descriptor::*;
//use crate::nn::protein::Fragment;
//use crate::matrix::rand_rot;






#[no_mangle]
pub extern fn main() -> i32
{
    println!("Begin Initialization");
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    cp2k_init_without_mpi();
    let _dev: Device = Device::default();



/*
    let s_co: System = System::read_xyz("/datc/lixt/Rust/NN/CO.xyz");
    let s_h2: System = System::read_xyz("/datc/lixt/Rust/NN/H2.xyz");
    let s_h2o: System = System::read_xyz("/datc/lixt/Rust/NN/H2O.xyz");
    let s_nh3: System = System::read_xyz("/datc/lixt/Rust/NN/NH3.xyz");
    let s: Vec<System> = vec![s_co, s_h2, s_h2o, s_nh3];
    let mul: Vec<usize> = vec![5, 5, 5, 5];
    let size: [f64; 3] = [18.0, 18.0, 18.0];
    let s: System = System::super_cell(s, mul, size, 3.0);
    s.write_xyz("supercell.xyz", true, 0);
*/



    let para: Para = Para::new();
    let my_comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let output_path: String = output::create_output_path(&my_comm, Some(world.rank() % 1));
    let cp2k_output_file: String = format!("{}cp2k.out", output_path);

    //let s_ch2o: System = System::read_xyz("CH2O.xyz");
    //let s_h2o: System = System::read_xyz("H2O.xyz");
    //let s_ca: System = System::read_xyz("Ca.xyz");
    //let s_oh: System = System::read_xyz("OH.xyz");
    //let s_hco: System = System::read_xyz("HCO.xyz");
    //let s_glycolaldehyde: System = System::read_xyz("Glycolaldehyde.xyz");
    //let s_ethylene_glycol: System = System::read_xyz("Ethylene_Glycol.xyz");
    //let s_glycolaldehyde_anion: System = System::read_xyz("Glycolaldehyde_anion.xyz");
    //let s_hydroxymethyl: System = System::read_xyz("Hydroxymethyl.xyz");
    //let s: Vec<System> = vec![s_glycolaldehyde, s_ethylene_glycol, s_glycolaldehyde_anion, s_hydroxymethyl, s_ca, s_ch2o, s_h2o, s_oh, s_hco];   //generation according to the sequence
    //let mul: Vec<usize> = vec![0, 0, 4, 0, 4, 16, 16, 4, 0];     //5 i.e. 5 h20 will be added
    //let mut s: System = System::auto_super_cell(&s, &mul, 3.0); //3 is the dist btw the,, in [A]
  
    let s_ch2o: System = System::read_xyz("CH2O.xyz");
    let s_h2o: System = System::read_xyz("H2O.xyz");
    let s_ca: System = System::read_xyz("Ca.xyz");
    let s_oh: System = System::read_xyz("OH.xyz");
    let s_hco: System = System::read_xyz("HCO.xyz");
    let s_ca_oh_oh: System = System::read_xyz("Ca-OH-OH.xyz");
    let s_glycolaldehyde: System = System::read_xyz("Glycolaldehyde.xyz");
    let s_ethylene_glycol: System = System::read_xyz("Ethylene_Glycol.xyz");
    let s_ca_oh_ch2oh: System = System::read_xyz("Ca-OH-CH2OH.xyz");
    let s_ca_oh_hcoch2o: System = System::read_xyz("Ca-OH-HCOCH2O.xyz");
    let s_ca_oh_hoch2ch2o: System = System::read_xyz("Ca-OH-HOCH2CH2O.xyz");
    let s_10_1_hcochoh: System = System::read_xyz("10-1-HCOCHOH.xyz");
    let s_10_ca_oh_hcochoh: System = System::read_xyz("10-Ca-OH-HCOCHOH.xyz");
    let s_ca_oh_oh_hoch2ch2oh: System = System::read_xyz("Ca-OH-OH-HOCH2CH2OH.xyz");
    let s_12_hoch2ch2oh: System = System::read_xyz("12-HOCH2CH2OH.xyz");
    let s_13_ca_oh_coch2oh: System = System::read_xyz("13-Ca-OH-COCH2OH.xyz");
    let s_14_ca_oh_hoch2choh: System = System::read_xyz("14-Ca-OH-HOCH2CHOH.xyz");
    let s_15_hochchoh: System = System::read_xyz("15-HOCHCHOH.xyz");
    let s_16_ca_oh_hcochohchoh: System = System::read_xyz("16-Ca-OH-HCOCHOHCHOH.xyz");
    let s_18_glyceraldehyde: System = System::read_xyz("18-glyceraldehyde.xyz");
    let s_19_tetrose: System = System::read_xyz("19-tetrose.xyz");
    let s_20_linear_pentose: System = System::read_xyz("20-linear-pentose.xyz");
    let s: Vec<System> = vec![s_ca, s_oh, s_20_linear_pentose, s_19_tetrose, s_ca_oh_oh, s_10_1_hcochoh, s_10_ca_oh_hcochoh, s_16_ca_oh_hcochohchoh, s_glycolaldehyde, s_18_glyceraldehyde, s_ethylene_glycol, s_ca_oh_hcoch2o, s_ca_oh_hoch2ch2o, s_ca_oh_ch2oh, s_ca_oh_oh_hoch2ch2oh, s_12_hoch2ch2oh, s_13_ca_oh_coch2oh, s_14_ca_oh_hoch2choh, s_15_hochchoh, s_ch2o, s_h2o, s_hco];
    let mul: Vec<usize> = vec![1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 0];     //5 i.e. 5 h20 will be added
    let mut s: System = System::auto_super_cell(&s, &mul, 3.5); //3 is the dist btw the,, in [A]
    
    //above xyz updated on 24/6 at 2222


    // Parallel processing
    let rank = my_comm.rank();
    let root_process = my_comm.process_at_rank(ROOT_RANK);
    let mut cell: Array2<f64> = s.cell.clone().expect(&error_none_value("s.cell"));
    let coord_p: &mut [f64] = s.coord.as_slice_mut().expect(&error_as_slice("s.coord"));
    let cell_p: &mut [f64] = cell.as_slice_mut().expect(&error_as_slice("cell"));
    root_process.broadcast_into(coord_p);
    root_process.broadcast_into(cell_p);
    s.cell = Some(cell);
    if rank == ROOT_RANK
    {
        s.write_xyz("supercell.xyz", true, 0);
    }

    let evolution_pot: EvolutionPot = EvolutionPot
    {
        initial_state: s,
        initial_velocity: None,
        para: &para,
        output_path,
    };

    {
        let cp2k_pes = Cp2kPES::new("B97-3C.inp", &cp2k_output_file, 4);
        evolution_pot.rtip_nvt_md(&my_comm, &cp2k_pes);
    }





/*
    let mut s: System = System::read_xyz("6EQE.xyz");
    let mut atom_add_pot: Vec<usize> = Vec::new();
    for i in 1907..1912
    {
        atom_add_pot.push(i);
    }
    for i in 2982..2993
    {
        atom_add_pot.push(i);
    }
//    atom_add_pot.push(3836);
//    atom_add_pot.push(3837);
//    atom_add_pot.push(3850);
//    atom_add_pot.push(3851);
//    atom_add_pot.push(3867);
//    atom_add_pot.push(3868);

    for i in 3832..3878
    {
        atom_add_pot.push(i);
    }
    s.atom_add_pot = Some(atom_add_pot);

    let para: Para = Para::new();
    let my_comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let (str_output_file, output_file) = output::output_rtip(&my_comm, Some(world.rank() % 1));
    let cp2k_output_file = output::output_cp2k(Some(world.rank() % 1));

    let evolution_pot: EvolutionPot = EvolutionPot
    {
        initial_state: s,
        para: &para,
        str_output_file,
        output_file,
    };

    {
        let cp2k_pes = Cp2kPES::new("6EQE.inp", &cp2k_output_file, 4);
        evolution_pot.rtip_nvt_md(&my_comm, &cp2k_pes);
    }
*/



/*
    let ps: ProteinSystem = ProteinSystem::read_pdb("glycine.pdb");
    let s: System = ps.to_system();
    let para: Para = Para::new();

    let my_comm = world.split_by_color(Color::with_value(world.rank() % 1)).unwrap();
    let (str_output_file, output_file) = output::output_rtip(&my_comm, Some(world.rank() % 1));
    let cp2k_output_file = output::output_cp2k(Some(world.rank() % 1));

    let repulsive_pot: RepulsivePot = RepulsivePot
    {
        local_min: s,
        nearby_ts: Vec::<System>::new(),
        para: &para,
        str_output_file,
        output_file,
    };
    let cp2k_pes = Cp2kPES::new("cp2k.inp", &cp2k_output_file, 4);
    repulsive_pot.rtip_nvt_md(&my_comm, &cp2k_pes);
*/





//    for i in 0..s.fragment.len()
//    {
//        update_neighbor_list(&mut s, i);
//    }
//    let (coord_within_rcut, _list_within_rcut, atomic_number_within_rcut): (Array2<f64>, Vec<usize>, Vec<usize>) = truncate_cluster(&s, 3);
//    get_interfragment_descriptor(coord_within_rcut, atomic_number_within_rcut, s.fragment[3].natom);





//    s.write_pdb("out.pdb", true, 1);






    cp2k_finalize_without_mpi();
    println!("Finalization done");
    return 0;
}










