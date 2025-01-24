//! The module for DFT data saving and loading
use crate::common::constants::{Device, Element, FragmentType};
use crate::common::error::*;
use crate::nn::protein::ProteinSystem;
use crate::nn::interfragment_descriptor::{N_DES, truncate_cluster, get_interfragment_descriptor};
use crate::nn::intrafragment_descriptor::get_intrafragment_descriptor;
use ndarray::{Array2, Array3, s};
use dfdx::shapes::{Const, Rank1};
use dfdx::tensor::{Tensor, TensorFrom, TensorFromVec};
use savefile::load_file;
use savefile_derive::Savefile;





pub const EPS: f64 = 0.000001;





/// The basic structure reserving the DFT data for a series of structures of a molecular system, which is going to be saved into the disk
///
/// # Fields
/// ```
/// nstruct: the number of structures of the molecular system
/// cell: the fixed cell of the molecular system (3*3 Vec, Unit: Bohr)
/// natom: the number of atoms of the molecular system
/// atom_type: the types of the atoms within the molecular system (natom Vec)
/// coord: the atomic coordinates of all the structures of the molecular system (nstruct*natom*3 Vec, Unit: Bohr)
/// pot: the potential energies of all the structures of the molecular system (nstruct Vec, Unit: Hartree)
/// force: the atomic forces of all the structures of the molecular system (nstruct*natom*3 Vec, Unit: Hartree/Bohr)
/// ```
#[derive(Debug, Savefile)]
pub struct DataSaved
{
    pub nstruct: usize,
    pub cell: Option< Vec<f64> >,
    pub natom: usize,
    pub atom_type: Vec<Element>,
    pub coord: Vec<f64>,
    pub pot: Vec<f64>,
    pub force: Vec<f64>,
}





/// Basic structure for the structural descriptor of a fragment, derived from the fragment structure and its finite difference perturbation (only for training)
///
/// # Fields
/// ```
/// fragment_type: the type of the fragment (Atom, Residue, or Molecule)
/// interfragment_descriptor: the interfragment descriptor, derived from the fragment structure
/// intrafragment_descriptor: the intrafragment descriptor, derived from the fragment structure (only exist for Residue(AminoAcid))
/// interfragment_descriptor_diff: the interfragment descriptor, derived from the finite difference perturbation of the fragment structure
/// intrafragment_descriptor_diff: the intrafragment descriptor, derived from the finite difference perturbation of the fragment structure (only exist for Residue(AminoAcid))
/// ```
#[derive(Debug)]
pub struct FragmentDescriptor
{
    pub fragment_type: FragmentType,
    pub interfragment_descriptor: Tensor<Rank1<N_DES>, f64, Device>,
    pub intrafragment_descriptor: Option< Tensor<(usize,), f64, Device> >,
    pub interfragment_descriptor_diff: Tensor<(usize, Const<N_DES>), f64, Device>,
    pub intrafragment_descriptor_diff: Option< Tensor<(usize, usize), f64, Device> >,
}





/// Basic structure for the structural descriptor of a protein system, derived from the protein system structure and its finite difference perturbation (only for training)
///
/// # Fields
/// ```
/// n_diff: the number of finite difference perturbation structures
/// fragment_descriptor: Assembly of structural descriptor of all the fragments in a protein system
/// pot: DFT potential energy of the protein system
/// pot_diff: DFT potential energy of the finite difference perturbation of the protein system
/// ```
#[derive(Debug)]
pub struct ProteinSystemDescriptor
{
    pub n_diff: usize,
    pub fragment_descriptor: Vec<FragmentDescriptor>,
    pub pot: Tensor<Rank1<1>, f64, Device>,
    pub pot_diff: Tensor<(usize, Const<1>), f64, Device>,
}










impl ProteinSystemDescriptor
{
    /// Derive the structural descriptor and finite difference perturbation for the input ProteinSystem
    ///
    /// # Parameters
    /// ```
    /// s: the input ProteinSystem, specially containing the potential energy and atomic forces from DFT calculations
    /// des: the output descriptors for the ProteinSystem and its finite difference perturbation
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    fn from_struct(s: &ProteinSystem) -> Self
    {
        // Define a Device (CPU or Cuda) to build NNs
        let dev: Device = Device::seed_from_u64(1314);

        let n_diff: usize = s.natom * 6;
        let mut within_list: bool;                // To reserve if the atom is within the list or not
        let mut index: usize = 0;                // To reserve the index of the atom in the list
        let mut row: usize;



        // Achieve the fragment descriptors for all the fragments in the protein
        let mut fragment_descriptor: Vec<FragmentDescriptor> = Vec::with_capacity(s.fragment.len());
        // For each fragment in the protein system
        for i in 0..s.fragment.len()
        {
            // Achieve the interfragment descriptors for Fragment i
            let (coord_within_rcut, list_within_rcut, atomic_number_within_rcut): (Array2<f64>, Vec<usize>, Vec<usize>) = truncate_cluster(&s, i);
            let (interfragment_descriptor, interfragment_gradient): (Vec<f64>, Array2<f64>) = get_interfragment_descriptor(coord_within_rcut, atomic_number_within_rcut, s.fragment[i].natom);
            let mut interfragment_descriptor_diff: Vec<f64> = Vec::with_capacity(n_diff * N_DES);
            // For each atom in the protein system
            for j in 0..s.natom
            {
                // Jubge if the atom is within the list or not. If within the list, reserve its index in the list
                within_list = false;
                for k in 0..list_within_rcut.len()
                {
                    if j == list_within_rcut[k]
                    {
                        within_list = true;
                        index = k;
                        break
                    }
                }

                // Obtain the interfragment descriptor for the finite difference perturbation on the atom
                if within_list                // If the atom is within the list, induce the finite difference perturbation on the interfragment descriptor
                {
                    // For x, y, z axis
                    for axis in 0..3
                    {
                        row = index * 3 + axis;
                        for k in 0..N_DES
                        {
                            interfragment_descriptor_diff.push(interfragment_descriptor[k] + EPS * interfragment_gradient[[row, k]]);
                        }
                        for k in 0..N_DES
                        {
                            interfragment_descriptor_diff.push(interfragment_descriptor[k] - EPS * interfragment_gradient[[row, k]]);
                        }
                    }
                }
                else                // If the atom is outside the list, its finite difference perturbation doesn't affect the interfragment descriptor
                {
                    for _ in 0..6
                    {
                        for k in 0..N_DES
                        {
                            interfragment_descriptor_diff.push(interfragment_descriptor[k]);
                        }
                    }
                } 
            }

            // Achieve the intrafragment descriptors for Fragment i
            let (intrafragment_descriptor, intrafragment_descriptor_diff): (Option< Tensor<(usize,), f64, Device> >, Option< Tensor<(usize, usize), f64, Device> >) = match get_intrafragment_descriptor(&s.fragment[i])
            {
                Some((descriptor, gradient)) =>
                {
                    let n_des: usize = descriptor.len();
                    let mut descriptor_diff: Vec<f64> = Vec::with_capacity(n_diff * n_des);
                    // For each atom in the protein system
                    for j in 0..s.natom
                    {
                        // Jubge if the atom is within the list or not. If within the list, reserve its index in the list
                        within_list = false;
                        let atom_list: &Vec<usize> = s.fragment[i].atom_list.as_ref().expect(&error_none_value("s.fragment[i].atom_list"));
                        for k in 0..atom_list.len()
                        {
                            if j == atom_list[k]
                            {
                                within_list = true;
                                index = k;
                                break
                            }
                        }

                        // Obtain the intrafragment descriptor for the finite difference perturbation on the atom
                        if within_list                // If the atom is within the list, induce the finite difference perturbation on the intrafragment descriptor
                        {
                            // For x, y, z axis
                            for axis in 0..3
                            {
                                row = index * 3 + axis;
                                for k in 0..n_des
                                {
                                    descriptor_diff.push(descriptor[k] + EPS * gradient[[row, k]]);
                                }
                                for k in 0..n_des
                                {
                                    descriptor_diff.push(descriptor[k] - EPS * gradient[[row, k]]);
                                }
                            }
                        }
                        else
                        {
                            for _ in 0..6
                            {
                                for k in 0..n_des
                                {
                                    descriptor_diff.push(descriptor[k]);
                                }
                            }
                        }

                    }

                    (Some(dev.tensor_from_vec(descriptor, (n_des, ))), Some(dev.tensor_from_vec(descriptor_diff, (n_diff, n_des))))
                },

                None =>
                {
                    (None, None)
                },
            };

            // Save the interfragment descriptors and intrafragment descriptors of Fragment i
            fragment_descriptor.push
            (
                FragmentDescriptor
                {
                    fragment_type: s.fragment[i].fragment_type.clone(),
                    interfragment_descriptor: dev.tensor(interfragment_descriptor),
                    intrafragment_descriptor,
                    interfragment_descriptor_diff: dev.tensor_from_vec(interfragment_descriptor_diff, (n_diff, Const::<N_DES>)),
                    intrafragment_descriptor_diff,
                }
            );
        }



        // Achieve the potential energies for the protein system and its finite difference perturbation
        let pot: f64 = s.pot;
        let force: &Array2<f64> = s.force.as_ref().expect(&error_none_value("s.force"));
        let mut pot_diff: Vec<f64> = Vec::with_capacity(n_diff);
        // For each atom in the protein system
        for i in 0..s.natom
        {
            for j in 0..3
            {
                pot_diff.push(-EPS * force[[i, j]]);
                pot_diff.push(EPS * force[[i, j]]);
            }
        }



        // Return the whole structure
        ProteinSystemDescriptor
        {
            n_diff,
            fragment_descriptor,
            pot: dev.tensor(vec![pot]),
            pot_diff: dev.tensor_from_vec(pot_diff, (n_diff, Const::<1>)),
        }
    }





    /// Load the structural data for a specific protein system, and calculate the descriptors
    ///
    /// # Parameters
    /// ```
    /// input_sub_dir: the sub_directory (the path with respect to directory 'data') containing the input files
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn from_data(input_sub_dir: &str) -> Vec<Self>
    {
        // Obtain the input filenames
        let str_input_file: String = format!("data/{}/str.pdb", input_sub_dir);
        let data_input_file: String = format!("data/{}/data.bin", input_sub_dir);

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



        // Define the Vec to contain the protein descriptors
        let mut protein_descriptor: Vec<ProteinSystemDescriptor> = Vec::with_capacity(data.nstruct);

        // Calculate the descriptors for all the structures
        for i in 0..data.nstruct                // For each structure
        {
            // Copy the atomic coordinates, potential, forces to the ProteinSystem
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
            s.pot = pot[i];
            s.force = Some(force.slice(s![i, .., ..]).to_owned());

            // Calculate the descriptors
            protein_descriptor.push(Self::from_struct(&s));
        }

        protein_descriptor
    }
}










