//! Data structure for protein





use crate::common::constants::{PI, ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, Element, FragmentType, H_ATOM_TYPE, O_ATOM_TYPE};
use crate::common::error::*;
use crate::pes_exploration::system::System;
use std::fs;
use std::fs::File;
use std::io::Write;
use ndarray::{Array2, array};





/// The basic structure describing a fragment (e.g. atom, residue, or molecule).
///
/// # Fields
/// ```
/// fragment_type: the type of the fragment (Atom, Residue, or Molecule)
/// natom: the number of atoms of the fragment
/// coord: the coordinates of the atoms of the fragment (natom*3 Array, Unit: Bohr)
/// atom_type: the types of the atoms of the fragment (natom vec)
/// pot: the potential energy of the fragment
/// force: the forces of the atoms of the fragment (natom*3 Array, Unit: Hartree/Bohr)
/// atom_list: the indices of the atoms of the fragment in the whole protein system (natom vec)
/// neighbor_list: the indices of the neighbor atoms around the fragment in the whole protein system
/// ```
#[derive(Debug, Clone)]
pub struct Fragment
{
    pub fragment_type: FragmentType,
    pub natom: usize,
    pub coord: Array2<f64>,
    pub atom_type: &'static [Element],
    pub pot: f64,
    pub force: Option< Array2<f64> >,
    pub atom_list: Option< Vec<usize> >,
    pub neighbor_list: Option< Vec<usize> >,
}





/// The basic structure describing a protein system (constructed by atoms, residues, and molecules).
///
/// # Fields
/// ```
/// natom: the number of atoms of the protein system
/// coord: the coordinates of the atoms of the protein system (natom*3 Array, Unit: Bohr)
/// cell: the cell of the protein system (3*3 Array, Unit: Bohr)
/// atom_type: the types of the atoms of the protein system (natom Vec)
/// pot: the potential energy of the protein system
/// force: the forces of the atoms of the protein system (natom*3 Array, Unit: Hartree/Bohr)
/// fragment: the constructed fragments of the protein system
/// ```
#[derive(Debug, Clone)]
pub struct ProteinSystem
{
    pub natom: usize,
    pub coord: Array2<f64>,
    pub cell: Option< Array2<f64> >,
    pub atom_type: Vec<Element>,
    pub pot: f64,
    pub force: Option< Array2<f64> >,
    pub fragment: Vec<Fragment>,
}










impl ProteinSystem
{
    /// Read a protein system from a PDB file
    ///
    /// # Parameters
    /// ```
    /// filename: name of the PDB file to read from
    /// ```
    ///
    /// # Examples
    /// ```
    /// let s = ProteinSystem::read_pdb("filename.pdb");
    /// ```
    pub fn read_pdb(filename: &str) -> Self
    {

        let content = fs::read_to_string(filename).expect(&error_file("reading", filename));                // Read the whole file
        let mut line = content.lines();                 // Take the iteractor for the lines of the file

        let mut cell: Option< Array2<f64> > = None;                // Initialize the cell of the protein system
        let mut fragment: Vec<Fragment> = Vec::new();                // Initialize the fragments of the protein system
        let mut fragment_index: i32 = -1;                // Initialize the index of the fragment
        let mut fragment_str: Vec< Vec<&str> > = Vec::new();                // Initialize the str of the fragment
        loop
        {
            // Convert the line to str
            let line_str: Vec<&str> = match line.next()
            {
                Some(value) => value.split_whitespace().collect(),
                None => panic!("{}", error_read(filename)),
            };

            match line_str[0]
            {
                // Read the cell of the protein system from the PDB file.
                "CRYST1" =>
                {
                    if line_str.len() < 7
                    {
                        panic!("{}", error_read(filename));
                    }
                    let a: f64 = line_str[1].parse().expect(&error_read(filename));
                    let b: f64 = line_str[2].parse().expect(&error_read(filename));
                    let c: f64 = line_str[3].parse().expect(&error_read(filename));
                    let alpha: f64 = line_str[4].parse::<f64>().expect(&error_read(filename)) * PI / 180.0;
                    let beta: f64 = line_str[5].parse::<f64>().expect(&error_read(filename)) * PI / 180.0;
                    let gamma: f64 = line_str[6].parse::<f64>().expect(&error_read(filename)) * PI / 180.0;
                    let cell_21: f64 = gamma.cos();
                    let cell_22: f64 = gamma.sin();
                    let cell_31: f64 = beta.cos();
                    let cell_32: f64 = ( alpha.cos() - beta.cos() * cell_21 ) / cell_22;
                    let cell_33: f64 = ( 1.0 - cell_31 * cell_31 - cell_32 * cell_32 ).sqrt();
                    // Warning: The unit of the cell should be transformed from Angstrom to Bohr
                    cell = Some(array!
                    [
                        [         a*ANGSTROM_TO_BOHR,                        0.0,                        0.0 ],
                        [ cell_21*b*ANGSTROM_TO_BOHR, cell_22*b*ANGSTROM_TO_BOHR,                        0.0 ],
                        [ cell_31*c*ANGSTROM_TO_BOHR, cell_32*c*ANGSTROM_TO_BOHR, cell_33*c*ANGSTROM_TO_BOHR ],
                    ]);
                },

                "ATOM" =>
                {
                    if line_str.len() < 11
                    {
                        panic!("{}", error_read(filename));
                    }
                    // If the index of the fragment is changed, recognize the last fragment and initialize the new fragment 
                    if line_str[4].parse::<i32>().expect(&error_read(filename)) != fragment_index
                    {
                        // If the last fragment isn't empty, recognize the last fragment
                        if !fragment_str.is_empty()
                        {
                            fragment.append(&mut recognize_fragment(&fragment_str, filename));
                        }
                        // Initialize the new fragment
                        fragment_index = line_str[4].parse::<i32>().expect(&error_read(filename));
                        fragment_str = vec![line_str];
                    }
                    // Else, update the fragment 
                    else
                    {
                        fragment_str.push(line_str);
                    }
                },

                "TER" =>
                {
                    // If the last fragment isn't empty, recognize the last fragment
                    if !fragment_str.is_empty()
                    {
                        fragment.append(&mut recognize_fragment(&fragment_str, filename));
                    }
                    // Initialize the new fragment
                    fragment_str = vec![];
                },

                "END" =>
                {
                    // If the last fragment isn't empty, recognize the last fragment
                    if !fragment_str.is_empty()
                    {
                        fragment.append(&mut recognize_fragment(&fragment_str, filename));
                    }
                    break
                },

                &_ => (),
            }
        }

        // Integrate the fragment information
        let mut natom: usize = 0;
        for i in 0..fragment.len()
        {
            natom += fragment[i].natom;
        }
        let mut coord: Array2<f64> = Array2::zeros((natom, 3));
        let mut atom_type: Vec<Element> = Vec::with_capacity(natom);
        let mut index: usize = 0;
        for i in 0..fragment.len()
        {
            let mut atom_list: Vec<usize> = Vec::with_capacity(fragment[i].natom);
            for j in 0..fragment[i].natom
            {
                coord[[index,0]] = fragment[i].coord[[j,0]];
                coord[[index,1]] = fragment[i].coord[[j,1]];
                coord[[index,2]] = fragment[i].coord[[j,2]];
                atom_type.push(fragment[i].atom_type[j].clone());
                atom_list.push(index);
                index += 1;
            }
            fragment[i].atom_list = Some(atom_list);
        }

        // Return the protein system
        return ProteinSystem
        {
            natom,
            coord,
            cell,
            atom_type,
            pot: 0.0,
            force: None,
            fragment,
        };



        // An inner function for recognition of the fragments read from the PDB file.
        fn recognize_fragment(fragment_str: &Vec< Vec<&str> >, filename: &str) -> Vec<Fragment>
        {
            // Matching from str to the properties of the fragment
            let fragment_type: FragmentType = FragmentType::from_str(fragment_str[0][3]);
            let natom: usize = FragmentType::get_natom(fragment_str[0][3]);
            let atom_type: &[Element] = FragmentType::get_atom_type(fragment_str[0][3]);

            match fragment_type
            {
                // For atoms and molecules, read in the fragments directly
                FragmentType::Atom(_) | FragmentType::Molecule(_) =>
                {
                    // If the read-in fragment has a different number of atoms compared with the coded fragment, panic the error.
                    if fragment_str.len() != natom
                    {
                        panic!("{}", error_fragment_format(fragment_type, filename));
                    }

                    let mut coord: Array2<f64> = Array2::zeros((natom, 3));
                    for i in 0..natom
                    {
                        // If the read-in fragment has different atom types compared with the coded fragment, panic the error.
                        if (Element::from_str(fragment_str[i][10]) != atom_type[i]) || (FragmentType::from_str(fragment_str[i][3]) != fragment_type)
                        {
                            panic!("{}", error_fragment_format(fragment_type, filename));
                        }
                        // Else, read in the coordinates
                        // Warning: The unit of the atomic coordinates should be transformed from Angstrom to Bohr
                        else
                        {
                            coord[[i, 0]] = fragment_str[i][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                            coord[[i, 1]] = fragment_str[i][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                            coord[[i, 2]] = fragment_str[i][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                        }
                    }
                    // Return the fragment vector
                    vec!
                    [
                        Fragment
                        {
                            fragment_type,
                            natom,
                            coord,
                            atom_type,
                            pot: 0.0,
                            force: None,
                            atom_list:None,
                            neighbor_list: None,
                        },
                    ]
                },

                // For residues, first recognize the heads and tails, and then read in the fragments
                FragmentType::Residue(_) =>
                {
                    // If the residue has no special head and tail
                    if fragment_str.len() == natom
                    {
                        let mut coord: Array2<f64> = Array2::zeros((natom, 3));
                        for i in 0..natom
                        {
                            // If the read-in residue has different atom types compared with the coded residue, panic the error.
                            if (Element::from_str(fragment_str[i][10]) != atom_type[i]) || (FragmentType::from_str(fragment_str[i][3]) != fragment_type)
                            {
                                panic!("{}", error_fragment_format(fragment_type, filename));
                            }
                            // Else, read in the coordinates
                            // Warning: The unit of the atomic coordinates should be transformed from Angstrom to Bohr
                            else
                            {
                                coord[[i, 0]] = fragment_str[i][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                coord[[i, 1]] = fragment_str[i][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                coord[[i, 2]] = fragment_str[i][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                            }
                        }
                        // Return the fragment vector
                        vec!
                        [
                            Fragment
                            {
                                fragment_type,
                                natom,
                                coord,
                                atom_type,
                                pot: 0.0,
                                force: None,
                                atom_list:None,
                                neighbor_list: None,
                            },
                        ]
                    }

                    // If the residue has a special head or a special tail with one more atom
                    else if fragment_str.len() == (natom + 1)
                    {
                        // Match the penult atom of the read-in residue
                        match Element::from_str(fragment_str[natom-1][10])
                        {
                            // For the residue with a head 'NH2' and an ordinary tail 'CO'
                            Element::C =>
                            {
                                let mut coord: Array2<f64> = Array2::zeros((natom, 3));
                                let mut coord_head: Array2<f64> = Array2::zeros((1, 3));
                                // Handle the first atom 'N'
                                if Element::from_str(fragment_str[0][10]) != atom_type[0]
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord[[0, 0]] = fragment_str[0][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord[[0, 1]] = fragment_str[0][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord[[0, 2]] = fragment_str[0][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the second atom 'H'
                                if (Element::from_str(fragment_str[1][10]) != Element::H) || (FragmentType::from_str(fragment_str[1][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_head[[0, 0]] = fragment_str[1][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head[[0, 1]] = fragment_str[1][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head[[0, 2]] = fragment_str[1][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the rest atoms
                                for i in 2..(natom+1)
                                {
                                    if (Element::from_str(fragment_str[i][10]) != atom_type[i-1]) || (FragmentType::from_str(fragment_str[i][3]) != fragment_type)
                                    {
                                        panic!("{}", error_fragment_format(fragment_type, filename));
                                    }
                                    else
                                    {
                                        coord[[i-1, 0]] = fragment_str[i][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i-1, 1]] = fragment_str[i][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i-1, 2]] = fragment_str[i][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    }
                                }
                                // Return the fragment vector
                                vec!
                                [
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Head(Element::H),
                                        natom: 1,
                                        coord: coord_head,
                                        atom_type: &H_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type,
                                        natom,
                                        coord,
                                        atom_type,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                ]
                            },

                            // For the residue with an ordinary head 'NH' and a special tail 'COO'
                            Element::O =>
                            {
                                let mut coord: Array2<f64> = Array2::zeros((natom, 3));
                                let mut coord_tail: Array2<f64> = Array2::zeros((1, 3));
                                // Handle the first natom atoms
                                for i in 0..natom
                                {
                                    if (Element::from_str(fragment_str[i][10]) != atom_type[i]) || (FragmentType::from_str(fragment_str[i][3]) != fragment_type)
                                    {
                                        panic!("{}", error_fragment_format(fragment_type, filename));
                                    }
                                    else
                                    {
                                        coord[[i, 0]] = fragment_str[i][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i, 1]] = fragment_str[i][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i, 2]] = fragment_str[i][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    }
                                }
                                // Handle the last atom 'O'
                                if (Element::from_str(fragment_str[natom][10]) != Element::O) || (FragmentType::from_str(fragment_str[natom][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_tail[[0, 0]] = fragment_str[natom][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail[[0, 1]] = fragment_str[natom][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail[[0, 2]] = fragment_str[natom][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Return the fragment vector
                                vec!
                                [
                                    Fragment
                                    {
                                        fragment_type,
                                        natom,
                                        coord,
                                        atom_type,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Tail(Element::O),
                                        natom: 1,
                                        coord: coord_tail,
                                        atom_type: &O_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                ]
                            },

                            _ => panic!("{}", error_fragment_format(fragment_type, filename)),
                        }
                    }

                    // If the residue has a special head or a special tail with two more atoms
                    else if fragment_str.len() == (natom + 2)
                    {
                        // Match the last atom of the read-in residue
                        match Element::from_str(fragment_str[natom+1][10])
                        {
                            // For the residue with a head 'NH3' and an ordinary tail 'CO'
                            Element::O =>
                            {
                                let mut coord: Array2<f64> = Array2::zeros((natom, 3));
                                let mut coord_head1: Array2<f64> = Array2::zeros((1, 3));
                                let mut coord_head2: Array2<f64> = Array2::zeros((1, 3));
                                // Handle the first atom 'N'
                                if Element::from_str(fragment_str[0][10]) != atom_type[0]
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord[[0, 0]] = fragment_str[0][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord[[0, 1]] = fragment_str[0][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord[[0, 2]] = fragment_str[0][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the second atom 'H'
                                if (Element::from_str(fragment_str[1][10]) != Element::H) || (FragmentType::from_str(fragment_str[1][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_head1[[0, 0]] = fragment_str[1][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head1[[0, 1]] = fragment_str[1][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head1[[0, 2]] = fragment_str[1][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the third atom 'H'
                                if (Element::from_str(fragment_str[2][10]) != Element::H) || (FragmentType::from_str(fragment_str[2][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_head2[[0, 0]] = fragment_str[2][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head2[[0, 1]] = fragment_str[2][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head2[[0, 2]] = fragment_str[2][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the rest atoms
                                for i in 3..(natom+2)
                                {
                                    if (Element::from_str(fragment_str[i][10]) != atom_type[i-2]) || (FragmentType::from_str(fragment_str[i][3]) != fragment_type)
                                    {
                                        panic!("{}", error_fragment_format(fragment_type, filename));
                                    }
                                    else
                                    {
                                        coord[[i-2, 0]] = fragment_str[i][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i-2, 1]] = fragment_str[i][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i-2, 2]] = fragment_str[i][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    }
                                }
                                // Return the fragment vector
                                vec!
                                [
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Head(Element::H),
                                        natom: 1,
                                        coord: coord_head1,
                                        atom_type: &H_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Head(Element::H),
                                        natom: 1,
                                        coord: coord_head2,
                                        atom_type: &H_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type,
                                        natom,
                                        coord,
                                        atom_type,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                ]
                            },

                            // For the residue with an ordinary head 'NH' and a special tail 'COOH'
                            Element::H =>
                            {
                                let mut coord: Array2<f64> = Array2::zeros((natom, 3));
                                let mut coord_tail1: Array2<f64> = Array2::zeros((1, 3));
                                let mut coord_tail2: Array2<f64> = Array2::zeros((1, 3));
                                // Handle the first natom atoms
                                for i in 0..natom
                                {
                                    if (Element::from_str(fragment_str[i][10]) != atom_type[i]) || (FragmentType::from_str(fragment_str[i][3]) != fragment_type)
                                    {
                                        panic!("{}", error_fragment_format(fragment_type, filename));
                                    }
                                    else
                                    {
                                        coord[[i, 0]] = fragment_str[i][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i, 1]] = fragment_str[i][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i, 2]] = fragment_str[i][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    }
                                }
                                // Handle the penult atom 'O'
                                if (Element::from_str(fragment_str[natom][10]) != Element::O) || (FragmentType::from_str(fragment_str[natom][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_tail1[[0, 0]] = fragment_str[natom][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail1[[0, 1]] = fragment_str[natom][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail1[[0, 2]] = fragment_str[natom][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the last atom 'H'
                                if (Element::from_str(fragment_str[natom+1][10]) != Element::H) || (FragmentType::from_str(fragment_str[natom+1][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_tail2[[0, 0]] = fragment_str[natom+1][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail2[[0, 1]] = fragment_str[natom+1][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail2[[0, 2]] = fragment_str[natom+1][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Return the fragment vector
                                vec!
                                [
                                    Fragment
                                    {
                                        fragment_type,
                                        natom,
                                        coord,
                                        atom_type,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Tail(Element::O),
                                        natom: 1,
                                        coord: coord_tail1,
                                        atom_type: &O_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Tail(Element::H),
                                        natom: 1,
                                        coord: coord_tail2,
                                        atom_type: &H_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                ]
                            },

                            _ => panic!("{}", error_fragment_format(fragment_type, filename)),
                        }
                    }

                    // If the residue has a special head and a special tail
                    else if fragment_str.len() == (natom + 3)
                    {
                        // Match the last atom of the read-in residue
                        match Element::from_str(fragment_str[natom+2][10])
                        {
                            // For the residue with a special head 'NH3' and a special tail 'COO'
                            Element::O =>
                            {
                                let mut coord: Array2<f64> = Array2::zeros((natom, 3));
                                let mut coord_head1: Array2<f64> = Array2::zeros((1, 3));
                                let mut coord_head2: Array2<f64> = Array2::zeros((1, 3));
                                let mut coord_tail: Array2<f64> = Array2::zeros((1, 3));
                                // Handle the first atom 'N'
                                if Element::from_str(fragment_str[0][10]) != atom_type[0]
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord[[0, 0]] = fragment_str[0][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord[[0, 1]] = fragment_str[0][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord[[0, 2]] = fragment_str[0][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the second atom 'H'
                                if (Element::from_str(fragment_str[1][10]) != Element::H) || (FragmentType::from_str(fragment_str[1][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_head1[[0, 0]] = fragment_str[1][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head1[[0, 1]] = fragment_str[1][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head1[[0, 2]] = fragment_str[1][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the third atom 'H'
                                if (Element::from_str(fragment_str[2][10]) != Element::H) || (FragmentType::from_str(fragment_str[2][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_head2[[0, 0]] = fragment_str[2][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head2[[0, 1]] = fragment_str[2][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head2[[0, 2]] = fragment_str[2][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the middle atoms
                                for i in 3..(natom+2)
                                {
                                    if (Element::from_str(fragment_str[i][10]) != atom_type[i-2]) || (FragmentType::from_str(fragment_str[i][3]) != fragment_type)
                                    {
                                        panic!("{}", error_fragment_format(fragment_type, filename));
                                    }
                                    else
                                    {
                                        coord[[i-2, 0]] = fragment_str[i][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i-2, 1]] = fragment_str[i][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i-2, 2]] = fragment_str[i][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    }
                                }
                                // Handle the last atom 'O'
                                if (Element::from_str(fragment_str[natom+2][10]) != Element::O) || (FragmentType::from_str(fragment_str[natom+2][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_tail[[0, 0]] = fragment_str[natom+2][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail[[0, 1]] = fragment_str[natom+2][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail[[0, 2]] = fragment_str[natom+2][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Return the fragment vector
                                vec!
                                [
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Head(Element::H),
                                        natom: 1,
                                        coord: coord_head1,
                                        atom_type: &H_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Head(Element::H),
                                        natom: 1,
                                        coord: coord_head2,
                                        atom_type: &H_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type,
                                        natom,
                                        coord,
                                        atom_type,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Tail(Element::O),
                                        natom: 1,
                                        coord: coord_tail,
                                        atom_type: &O_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                ]
                            },

                            // For the residue with a special head 'NH2' and a special tail 'COOH'
                            Element::H =>
                            {
                                let mut coord: Array2<f64> = Array2::zeros((natom, 3));
                                let mut coord_head: Array2<f64> = Array2::zeros((1, 3));
                                let mut coord_tail1: Array2<f64> = Array2::zeros((1, 3));
                                let mut coord_tail2: Array2<f64> = Array2::zeros((1, 3));
                                // Handle the first atom 'N'
                                if Element::from_str(fragment_str[0][10]) != atom_type[0]
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord[[0, 0]] = fragment_str[0][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord[[0, 1]] = fragment_str[0][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord[[0, 2]] = fragment_str[0][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the second atom 'H'
                                if (Element::from_str(fragment_str[1][10]) != Element::H) || (FragmentType::from_str(fragment_str[1][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_head[[0, 0]] = fragment_str[1][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head[[0, 1]] = fragment_str[1][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_head[[0, 2]] = fragment_str[1][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the middle atoms
                                for i in 2..(natom+1)
                                {
                                    if (Element::from_str(fragment_str[i][10]) != atom_type[i-1]) || (FragmentType::from_str(fragment_str[i][3]) != fragment_type)
                                    {
                                        panic!("{}", error_fragment_format(fragment_type, filename));
                                    }
                                    else
                                    {
                                        coord[[i-1, 0]] = fragment_str[i][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i-1, 1]] = fragment_str[i][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                        coord[[i-1, 2]] = fragment_str[i][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    }
                                }
                                // Handle the penult atom 'O'
                                if (Element::from_str(fragment_str[natom+1][10]) != Element::O) || (FragmentType::from_str(fragment_str[natom+1][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_tail1[[0, 0]] = fragment_str[natom+1][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail1[[0, 1]] = fragment_str[natom+1][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail1[[0, 2]] = fragment_str[natom+1][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Handle the last atom 'H'
                                if (Element::from_str(fragment_str[natom+2][10]) != Element::H) || (FragmentType::from_str(fragment_str[natom+2][3]) != fragment_type)
                                {
                                    panic!("{}", error_fragment_format(fragment_type, filename));
                                }
                                else
                                {
                                    coord_tail2[[0, 0]] = fragment_str[natom+2][5].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail2[[0, 1]] = fragment_str[natom+2][6].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                    coord_tail2[[0, 2]] = fragment_str[natom+2][7].parse::<f64>().expect(&error_fragment_format(fragment_type.clone(), filename)) * ANGSTROM_TO_BOHR;
                                }
                                // Return the fragment vector
                                vec!
                                [
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Head(Element::H),
                                        natom: 1,
                                        coord: coord_head,
                                        atom_type: &H_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type,
                                        natom,
                                        coord,
                                        atom_type,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Tail(Element::O),
                                        natom: 1,
                                        coord: coord_tail1,
                                        atom_type: &O_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                    Fragment
                                    {
                                        fragment_type: FragmentType::Tail(Element::H),
                                        natom: 1,
                                        coord: coord_tail2,
                                        atom_type: &H_ATOM_TYPE,
                                        pot: 0.0,
                                        force: None,
                                        atom_list:None,
                                        neighbor_list: None,
                                    },
                                ]
                            },

                            _ => panic!("{}", error_fragment_format(fragment_type, filename)),
                        }
                    }

                    // The read-in residue has a different number of atoms compared with the coded residue, panic the error.
                    else
                    {
                        panic!("{}", error_fragment_format(fragment_type, filename));
                    }
                },

                // Heads and tails are included in residues branch
                FragmentType::Head(_) | FragmentType::Tail(_) => panic!("{}", error_fragment_format(fragment_type, filename)),
            }
        }
    }





    /// Create a new PDB file (if already existed, truncate it), or open an old PDB file, and write the structure (in Angstrom) into it
    ///
    /// # Parameters
    /// ```
    /// filename: name of the PDB file to be writen
    /// create_new_file: whether to create a new PDB file or not
    /// step: current step of the structure
    /// ```
    ///
    /// # Examples
    /// ```
    /// s.write_pdb("filename.pdb", true, 1);
    /// ```
    pub fn write_pdb(&self, filename: &str, create_new_file: bool, step: usize)
    {
        let mut pdb = match create_new_file
        {
            // If create_new_file == true, create a new file and write the PDB TITLE
            true =>
            {
                let mut pdb = File::create(filename).expect(&error_file("creating", filename));       // Create the PDB file
                pdb.write_all(b"TITLE     PDB file created by Protein_NN\n").expect(&error_file("writing", filename));       // Write the PDB TITLE
                pdb
            },
            // If create_new_file == false, open an old file and append to it
            false =>
            {
                File::options().append(true).open(filename).expect(&error_file("opening", filename))
            },
        };

        // Write the potential energy
        pdb.write_all(format!("REMARK    , Step = {:8}, E = {:15.8}\n", step, self.pot).as_bytes()).expect(&error_file("writing", filename));

        // Write the simulation cell
        match &self.cell
        {
            Some(cell) =>
            {
                let a: f64 = ( cell[[0,0]]*cell[[0,0]] + cell[[0,1]]*cell[[0,1]] + cell[[0,2]]*cell[[0,2]] ).sqrt() * BOHR_TO_ANGSTROM;
                let b: f64 = ( cell[[1,0]]*cell[[1,0]] + cell[[1,1]]*cell[[1,1]] + cell[[1,2]]*cell[[1,2]] ).sqrt() * BOHR_TO_ANGSTROM;
                let c: f64 = ( cell[[2,0]]*cell[[2,0]] + cell[[2,1]]*cell[[2,1]] + cell[[2,2]]*cell[[2,2]] ).sqrt() * BOHR_TO_ANGSTROM;
                let alpha: f64 = ( (cell[[1,0]]*cell[[2,0]] + cell[[1,1]]*cell[[2,1]] + cell[[1,2]]*cell[[2,2]]) / (b*c) ).acos() * 180.0 / PI;
                let beta: f64 = ( (cell[[2,0]]*cell[[0,0]] + cell[[2,1]]*cell[[0,1]] + cell[[2,2]]*cell[[0,2]]) / (c*a) ).acos() * 180.0 / PI;
                let gamma: f64 = ( (cell[[0,0]]*cell[[1,0]] + cell[[0,1]]*cell[[1,1]] + cell[[0,2]]*cell[[1,2]]) / (a*b) ).acos() * 180.0 / PI;
                // Warning: The unit of the atomic coordinates should be transformd from Bohr to Angstrom
                pdb.write_all(format!("CRYST1{:9.3}{:9.3}{:9.3}{:7.2}{:7.2}{:7.2} P 1           1\n", a, b, c, alpha, beta, gamma).as_bytes()).expect(&error_file("writing", filename));
            },
            None => (),
        }

        let mut index_atom: usize = 1;
        let mut index_residue: usize = 1;
        let mut num_head: usize = 0;
        // Write the elements, coordinates, and fragment types
        for i in 0..self.fragment.len()
        {
            match &self.fragment[i].fragment_type
            {
                // Write the atomic fragment directly
                FragmentType::Atom(fragment_type) =>
                {
                    // Warning: The unit of the atomic coordinates should be transformd from Bohr to Angstrom
                    pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i].atom_type[0]), format!("{:?}", fragment_type), index_residue%10000, self.fragment[i].coord[[0,0]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[0,1]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[0,2]] * BOHR_TO_ANGSTROM, format!("{:?}", self.fragment[i].atom_type[0]) ).as_bytes()).expect(&error_file("writing", filename));
                    index_atom += 1;
                    index_residue += 1;
                    pdb.write_all(b"TER\n").expect(&error_file("writing", filename));
                },

                // Write the molecular fragment directly 
                FragmentType::Molecule(fragment_type) =>
                {
                    for j in 0..self.fragment[i].natom
                    {
                        // Warning: The unit of the atomic coordinates should be transformd from Bohr to Angstrom
                        pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i].atom_type[j]), format!("{:?}", fragment_type), index_residue%10000, self.fragment[i].coord[[j,0]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[j,1]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[j,2]] * BOHR_TO_ANGSTROM, format!("{:?}", self.fragment[i].atom_type[j]) ).as_bytes()).expect(&error_file("writing", filename));
                        index_atom += 1;
                    }
                    index_residue += 1;
                    pdb.write_all(b"TER\n").expect(&error_file("writing", filename));
                },

                // Add the possible heads before the residue
                FragmentType::Residue(fragment_type) =>
                {
                    match num_head
                    {
                        // If there is no head, write the residue directly
                        0 =>
                        {
                            for j in 0..self.fragment[i].natom
                            {
                                // Warning: The unit of the atomic coordinates should be transformd from Bohr to Angstrom
                                pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i].atom_type[j]), format!("{:?}", fragment_type), index_residue%10000, self.fragment[i].coord[[j,0]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[j,1]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[j,2]] * BOHR_TO_ANGSTROM, format!("{:?}", self.fragment[i].atom_type[j]) ).as_bytes()).expect(&error_file("writing", filename));
                                index_atom += 1;
                            }
                            index_residue += 1;
                        },

                         // If there is one head atom, first write the 'N' atom, and then write the head 'H' atom, and then write the rest atoms of the residue
                        1 =>
                        {
                            // Warning: The unit of the atomic coordinates should be transformd from Bohr to Angstrom
                            pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i].atom_type[0]), format!("{:?}", fragment_type), index_residue%10000, self.fragment[i].coord[[0,0]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[0,1]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[0,2]] * BOHR_TO_ANGSTROM,format!("{:?}", self.fragment[i].atom_type[0]) ).as_bytes()).expect(&error_file("writing", filename));
                            index_atom += 1;
                            pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i-1].atom_type[0]), format!("{:?}", fragment_type), index_residue%10000, self.fragment[i-1].coord[[0,0]] * BOHR_TO_ANGSTROM, self.fragment[i-1].coord[[0,1]] * BOHR_TO_ANGSTROM, self.fragment[i-1].coord[[0,2]] * BOHR_TO_ANGSTROM,format!("{:?}", self.fragment[i-1].atom_type[0]) ).as_bytes()).expect(&error_file("writing", filename));
                            index_atom += 1;
                            for j in 1..self.fragment[i].natom
                            {
                                pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i].atom_type[j]), format!("{:?}", fragment_type), index_residue%10000, self.fragment[i].coord[[j,0]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[j,1]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[j,2]] * BOHR_TO_ANGSTROM,format!("{:?}", self.fragment[i].atom_type[j]) ).as_bytes()).expect(&error_file("writing", filename));
                                index_atom += 1;
                            }
                            index_residue += 1;
                            num_head = 0;
                        },

                        // If there is two head atoms, first write the 'N' atom, and then write the two head 'H' atoms, and then write the rest atoms of the residue
                        2 =>
                        {
                            // Warning: The unit of the atomic coordinates should be transformd from Bohr to Angstrom
                            pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i].atom_type[0]), format!("{:?}", fragment_type), index_residue%10000, self.fragment[i].coord[[0,0]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[0,1]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[0,2]] * BOHR_TO_ANGSTROM,format!("{:?}", self.fragment[i].atom_type[0]) ).as_bytes()).expect(&error_file("writing", filename));
                            index_atom += 1;
                            pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i-2].atom_type[0]),format!("{:?}", fragment_type), index_residue%10000, self.fragment[i-2].coord[[0,0]] * BOHR_TO_ANGSTROM, self.fragment[i-2].coord[[0,1]] * BOHR_TO_ANGSTROM, self.fragment[i-2].coord[[0,2]] * BOHR_TO_ANGSTROM,format!("{:?}", self.fragment[i-2].atom_type[0]) ).as_bytes()).expect(&error_file("writing", filename));
                            index_atom += 1;
                            pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i-1].atom_type[0]),format!("{:?}", fragment_type), index_residue%10000, self.fragment[i-1].coord[[0,0]] * BOHR_TO_ANGSTROM, self.fragment[i-1].coord[[0,1]] * BOHR_TO_ANGSTROM, self.fragment[i-1].coord[[0,2]] * BOHR_TO_ANGSTROM,format!("{:?}", self.fragment[i-1].atom_type[0]) ).as_bytes()).expect(&error_file("writing", filename));
                            index_atom += 1;
                            for j in 1..self.fragment[i].natom
                            {
                                pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i].atom_type[j]), format!("{:?}", fragment_type), index_residue%10000, self.fragment[i].coord[[j,0]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[j,1]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[j,2]] * BOHR_TO_ANGSTROM,format!("{:?}", self.fragment[i].atom_type[j]) ).as_bytes()).expect(&error_file("writing", filename));
                                index_atom += 1;
                            }
                            index_residue += 1;
                            num_head = 0;
                        },

                        _ => panic!("{}", error_fragment_format(self.fragment[i].fragment_type.clone(), filename)),
                    }
                },

                // Record the number of heads for the next residue
                FragmentType::Head(_) => num_head += 1,

                // Append the tails to the last residue
                FragmentType::Tail(fragment_type) =>
                {
                    match fragment_type
                    {
                        Element::O =>
                        {
                            match &self.fragment[i-1].fragment_type
                            {
                                FragmentType::Residue(residue_type) =>
                                {
                                    // Warning: The unit of the atomic coordinates should be transformd from Bohr to Angstrom
                                    pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i].atom_type[0]), format!("{:?}", residue_type), (index_residue-1)%10000, self.fragment[i].coord[[0,0]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[0,1]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[0,2]] * BOHR_TO_ANGSTROM, format!("{:?}", self.fragment[i].atom_type[0]) ).as_bytes()).expect(&error_file("writing", filename));
                                    index_atom += 1;
                                    if i == (self.fragment.len()-1)
                                    {
                                        pdb.write_all(b"TER\n").expect(&error_file("writing", filename));
                                    }
                                    else
                                    {
                                        match &self.fragment[i+1].fragment_type
                                        {
                                            FragmentType::Tail(_) => (),
                                            _ => pdb.write_all(b"TER\n").expect(&error_file("writing", filename)),
                                        }
                                    }
                                },
                                _ => panic!("{}", error_fragment_format(self.fragment[i].fragment_type.clone(), filename)),
                            }
                        },

                        Element::H =>
                        {
                            match &self.fragment[i-2].fragment_type
                            {
                                FragmentType::Residue(residue_type) =>
                                {
                                    // Warning: The unit of the atomic coordinates should be transformd from Bohr to Angstrom
                                    pdb.write_all(format!( "ATOM  {:>5} {:>4} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", index_atom%100000, format!("{:?}", self.fragment[i].atom_type[0]), format!("{:?}", residue_type), (index_residue-1)%10000, self.fragment[i].coord[[0,0]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[0,1]] * BOHR_TO_ANGSTROM, self.fragment[i].coord[[0,2]] * BOHR_TO_ANGSTROM, format!("{:?}", self.fragment[i].atom_type[0]) ).as_bytes()).expect(&error_file("writing", filename));
                                    index_atom += 1;
                                    pdb.write_all(b"TER\n").expect(&error_file("writing", filename));
                                },
                                _ => panic!("{}", error_fragment_format(self.fragment[i].fragment_type.clone(), filename)),
                            }
                        },

                        _ => panic!("{}", error_fragment_format(self.fragment[i].fragment_type.clone(), filename)),
                    }
                },
            }
        }

        // Write the PDB END
        pdb.write_all(b"END\n").expect(&error_file("writing", filename));
    }





    /// Copy the information of the input ProteinSystem, and create a System to return
    ///
    /// # Examples
    /// ```
    /// s.to_system();
    /// ```
    pub fn to_system(&self) -> System
    {
        System
        {
            natom: self.natom,
            coord: self.coord.clone(),
            cell: self.cell.clone(),
            atom_type: Some(self.atom_type.clone()),
            atom_add_pot: None,
            mutable: None,
            pot: self.pot,
        }
    }
}










