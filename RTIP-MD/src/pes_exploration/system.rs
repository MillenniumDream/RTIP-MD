//! About the structure, constraint, and status of the research system.





use crate::common::constants::{Element, BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR};
use crate::external::cp2k::*;
use crate::common::error::*;
use crate::matrix;
use std::fs;
use std::fs::File;
use std::io::Write;
use ndarray::{Array, Array1, Array2, array, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;





/// The basic structure describing the research system (e.g. molecule and crystal).
///
/// # Fields
/// ```
/// natom: the number of atoms in the research system
/// coord: the coordinates of the atoms in the research system (natom*3 Array, Unit: Bohr)
/// cell: the unit cell of the research system (3*3 Array, Unit: Bohr)
/// atom_type: the types of the atoms in the research system (natom Vec)
/// add_pot: whether to add the biase Gaussian potential on the atoms or not (natom Vec)
/// mutable: whether the atomic positions is mutable or not (natom*3 Array)
/// pot: the potential energy of the system
/// ```
#[derive(Clone, Debug)]
pub struct System
{
    pub natom: usize,
    pub coord: Array2<f64>,
    pub cell: Option< Array2<f64> >,
    pub atom_type: Option< Vec<Element> >,
    pub atom_add_pot: Option< Vec<usize> >,
    pub mutable: Option< Array2<bool> >,
    pub pot: f64,
}





impl System
{
    /// Input a CP2K force environment, get the number and coordinates of the particles from CP2K, and construct a System to return
    ///
    /// # Parameters
    /// ```
    /// force_env: the previously built CP2K force environment
    /// ```
    ///
    /// # Examples
    /// ```
    /// let s: System = System::from_cp2k(force_env);
    /// ```
    pub fn from_cp2k(force_env: ForceEnv) -> Self
    {
        let mut nparticle: i32 = 0;
        cp2k_get_nparticle(force_env, &mut nparticle);
        let natom: usize = nparticle.try_into().expect(&error_type_transformation("nparticle", "i32", "usize"));           // From i32 to usize

        let mut pos: Vec<f64> = vec![0.0; natom*3];
        cp2k_get_positions(force_env, &mut pos, nparticle*3);
        let coord: Array2<f64> = Array::from_shape_vec((natom,3), pos).expect(&error_none_value("coord"));          // From Vec to Array2
        
        let mut pot: f64 = 0.0;
        cp2k_get_potential_energy(force_env, &mut pot);

        System
        {
            natom,
            coord,
            cell: None,
            atom_type: None,
            atom_add_pot: None,
            mutable: None,
            pot,
        }
    }

    /// Input a CP2K force environment, consume the System and set the coordinates of the particles for CP2K
    ///
    /// # Parameters
    /// ```
    /// force_env: the previously built CP2K force environment
    /// ```
    ///
    /// # Examples
    /// ```
    /// s.to_cp2k(force_env)
    /// ```
    pub fn to_cp2k(&self, force_env: ForceEnv)
    {
        let n_el: i32 = (self.natom * 3).try_into().expect(&error_type_transformation("natom", "usize", "i32"));
        let pos: &[f64] = self.coord.as_slice().expect(&error_none_value("pos"));
        cp2k_set_positions(force_env, pos, n_el);
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
                pdb.write_all(b"TITLE     PDB file created by RTIP\n").expect(&error_file("writing", filename));       // Write the PDB TITLE
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
        
        // Write the atomic elements and coordinates
        match &self.atom_type
        {
            // If the elements of the atoms are reserved in System
            Some(atom_type) =>
            {
                for i in 0..self.natom
                {
                    // Warning: The unit of the atomic coordinates should be transformd from Bohr to Angstrom
                    pdb.write_all(format!("ATOM  {:>5} {:>4}              {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", (i+1)%100000, format!("{:?}", atom_type[i]), self.coord[[i,0]] * BOHR_TO_ANGSTROM, self.coord[[i,1]] * BOHR_TO_ANGSTROM, self.coord[[i,2]] * BOHR_TO_ANGSTROM, format!("{:?}", atom_type[i]) ).as_bytes()).expect(&error_file("writing", filename));
                }
            },

            // If the elements of the atoms are not reserved in System
            None =>
            {
                for i in 0..self.natom
                {
                    // Warning: The unit of the atomic coordinates should be transformed from Bohr to Angstrom
                    pdb.write_all(format!("ATOM  {:>5} {:>4}              {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n", (i+1)%100000, "XX", self.coord[[i,0]] * BOHR_TO_ANGSTROM, self.coord[[i,1]] * BOHR_TO_ANGSTROM, self.coord[[i,2]] * BOHR_TO_ANGSTROM, "XX" ).as_bytes()).expect(&error_file("writing", filename));
                }
            },
        }

        pdb.write_all(b"END\n").expect(&error_file("writing", filename));        // Write the PDB END
    }



    /// Read a system from a XYZ file
    ///
    /// # Parameters
    /// ```
    /// filename: name of the XYZ file to read from
    /// ```
    ///
    /// # Examples
    /// ```
    /// let s: System = System::read_xyz("filename.xyz");
    /// ```
    pub fn read_xyz(filename: &str) -> Self
    {
        let content = fs::read_to_string(filename).expect(&error_file("reading", filename));                // Read the whole file
        let mut line = content.lines();                 // Take the iteractor for the lines of the file
        
        // Read the number of atom from the first line
        let natom: usize = match line.next()
        {
            Some(value) => value.trim().parse().expect(&error_read(filename)),
            None => panic!("{}", error_read(filename)),
        };
        line.next();                // Ignore the second line
        
        // Read the element type and Cartesian coordinates of each atom iteractively
        let mut atom_type: Vec<Element> = vec![Element::H; natom];
        let mut coord: Array2<f64> = Array2::zeros((natom, 3));
        for i in 0..natom
        {
            // Read the element type and atomic coordinates
            let type_coord: Vec<&str> = match line.next()
            {
                Some(value) => value.split_whitespace().collect(),
                None => panic!("{}", error_read(filename)),
            };
            if type_coord.len() < 4
            {
                panic!("{}", error_read(filename));
            }

            // Restore the element type and atomic coordinates
            atom_type[i] = Element::from_str(type_coord[0]);
            coord[[i, 0]] = type_coord[1].parse().expect(&error_read(filename));
            coord[[i, 1]] = type_coord[2].parse().expect(&error_read(filename));
            coord[[i, 2]] = type_coord[3].parse().expect(&error_read(filename));
        }
        // Warning: The unit of the atomic coordinates should be transformed from Angstrom to Bohr
        coord *= ANGSTROM_TO_BOHR;

        System
        {
            natom,
            coord,
            cell: None,
            atom_type: Some(atom_type),
            atom_add_pot: None,
            mutable: None,
            pot: 0.0,
        }
    }

    /// Create a new XYZ file (if already existed, truncate it), or open an old XYZ file, and write the structure (in Angstrom) into it
    ///
    /// # Parameters
    /// ```
    /// filename: name of the XYZ file to be writen
    /// create_new_file: whether to create a new XYZ file or not
    /// step: current step of the structure
    /// ```
    ///
    /// # Examples
    /// ```
    /// s.write_xyz("filename.xyz", true, 1);
    /// ```
    pub fn write_xyz(&self, filename: &str, create_new_file: bool, step: usize)
    {
        let mut xyz = match create_new_file
        {
            // If create_new_file == true, create a new file
            true =>
            {
                File::create(filename).expect(&error_file("creating", filename))
            },
            // If create_new_file == false, open an old file and append to it
            false =>
            {
                File::options().append(true).open(filename).expect(&error_file("opening", filename))
            },
        };

        // Write the number of atoms
        xyz.write_all(format!("{:8}\n", self.natom).as_bytes()).expect(&error_file("writing", filename));

        // Write the potential energy
        xyz.write_all(format!("Step = {:8}, E = {:15.8}\n", step, self.pot).as_bytes()).expect(&error_file("writing", filename));

        // Write the atomic elements and coordinates
        match &self.atom_type
        {
            // If the elements of the atoms are reserved in System
            Some(atom_type) =>
            {
                for i in 0..self.natom
                {
                    // Warning: The unit of the atomic coordinates should be transformd from Bohr to Angstrom
                    xyz.write_all(format!("{:>4} {:20.10} {:20.10} {:20.10}\n", format!("{:?}", atom_type[i]), self.coord[[i,0]] * BOHR_TO_ANGSTROM, self.coord[[i,1]] * BOHR_TO_ANGSTROM, self.coord[[i,2]] * BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", filename));
                }
            },

            // If the elements of the atoms are not reserved in System
            None =>
            {
                for i in 0..self.natom
                {
                    // Warning: The unit of the atomic coordinates should be transformed from Bohr to Angstrom
                    xyz.write_all(format!("{:>4} {:20.10} {:20.10} {:20.10}\n", "XX", self.coord[[i,0]] * BOHR_TO_ANGSTROM, self.coord[[i,1]] * BOHR_TO_ANGSTROM, self.coord[[i,2]] * BOHR_TO_ANGSTROM).as_bytes()).expect(&error_file("writing", filename));
                }
            },
        }
    }





    /// Construct a large simulation cell containing multiple small molecules for RTIP MD simulation
    ///
    /// # Parameters
    /// ```
    /// s: a Vec containing the small molecules
    /// mul: a Vec containing the multiples of the molecules
    /// cell: the size of the simulation cell (in Angstrom)
    /// min_dist: the allowed minimum distance between the small molecules (in Angstrom)
    /// ```
    ///
    /// # Examples
    /// ```
    /// super_cell(s, mul, size, 3.0);
    /// ```
    pub fn super_cell(s: &Vec<System>, mul: &Vec<usize>, size: [f64; 3], mut min_dist: f64) -> Result<Self, String>
    {
        assert_eq!(s.len(), mul.len());
        min_dist *= ANGSTROM_TO_BOHR;

        // Initialization of the output System
        let mut natom: usize = 0;
        for i in 0..s.len()
        {
            natom += s[i].natom * mul[i];
        }
        let mut coord: Array2<f64> = Array2::zeros((natom, 3));
        let cell: Array2<f64> = array!
        [
            [size[0]*ANGSTROM_TO_BOHR,                      0.0,                      0.0],
            [                     0.0, size[0]*ANGSTROM_TO_BOHR,                      0.0],
            [                     0.0,                      0.0, size[0]*ANGSTROM_TO_BOHR],
        ];
        let mut atom_type: Vec<Element> = Vec::with_capacity(natom);

        // put the small molecules into the cell iterately
        natom = 0;
        for i in 0..s.len()                // For each small molecule
        {
            for _j in 0..mul[i]                // For each multiple
            {
                let mut suitable: bool = false;
                let mut coord_mol = s[i].coord.clone();
                for _ in 0..1000
                {
                    coord_mol = s[i].coord.clone();

                    // Move the geometric centers of the molecules to the origin
                    coord_mol -= &coord_mol.mean_axis(Axis(0)).expect(&error_none_value("coord_mol"));

                    // Rotate the molecule
                    let rot: Array2<f64> = matrix::rand_rot();
                    coord_mol = coord_mol.dot(&rot);

                    // Translate the molecule
                    let tran_x: f64 = Array1::random(1, Uniform::new(min_dist, cell[[0,0]] - min_dist))[0];
                    let tran_y: f64 = Array1::random(1, Uniform::new(min_dist, cell[[1,1]] - min_dist))[0];
                    let tran_z: f64 = Array1::random(1, Uniform::new(min_dist, cell[[2,2]] - min_dist))[0];
                    for k in 0..coord_mol.nrows()
                    {
                        coord_mol[[k,0]] += tran_x;
                        coord_mol[[k,1]] += tran_y;
                        coord_mol[[k,2]] += tran_z;
                    }

                    // Judge if the molecule is out of the cell or too close to the other atoms
                    let mut too_close: bool = false;
                    for k in 0..coord_mol.nrows()
                    {
                        if (coord_mol[[k,0]] < 0.0) || (coord_mol[[k,0]] > cell[[0,0]]) || (coord_mol[[k,1]] < 0.0) || (coord_mol[[k,1]] > cell[[1,1]]) || (coord_mol[[k,2]] < 0.0) || (coord_mol[[k,2]] > cell[[2,2]])
                        {
                            too_close = true;
                            break
                        }
                        for l in 0..natom
                        {
                            if (coord_mol[[k,0]] - coord[[l,0]]).powi(2) + (coord_mol[[k,1]] - coord[[l,1]]).powi(2) + (coord_mol[[k,2]] - coord[[l,2]]).powi(2) < min_dist.powi(2)
                            {
                                too_close = true;
                                break
                            }
                        }
                        if too_close == true
                        {
                            break
                        }
                    }

                    // If the molecule is not out of the cell or too close to the other atoms, set the molecule suitable and jump out of the loop
                    if too_close == false
                    {
                        suitable = true;
                        break
                    }
                }

                // Put the molecule into the cell if suitable; otherwise, panic the error
                if suitable == true
                {
                    for k in 0..coord_mol.nrows()
                    {
                        coord[[natom+k,0]] = coord_mol[[k,0]];
                        coord[[natom+k,1]] = coord_mol[[k,1]];
                        coord[[natom+k,2]] = coord_mol[[k,2]];
                        match &s[i].atom_type
                        {
                            Some(atom_type_mol) => atom_type.push(atom_type_mol[k].clone()),
                            None => (),
                        }
                    }
                    natom += coord_mol.nrows();
                }
                else
                {
                    return Err(format!("\n\n\n ERROR: The constructed simulation cell is too small. Please increase the size of the cell. \n\n\n"))
                }
            }
        }

        if atom_type.len() > 0
        {
            Ok(System
            {
                natom,
                coord,
                cell: Some(cell),
                atom_type: Some(atom_type),
                atom_add_pot: None,
                mutable: None,
                pot: 0.0,
            })
        }
        else
        {
            Ok(System
            {
                natom,
                coord,
                cell: Some(cell),
                atom_type: None,
                atom_add_pot: None,
                mutable: None,
                pot: 0.0,
            })
        }
    }





    /// Construct a large simulation cell containing multiple small molecules for RTIP MD simulation, in a automated manner.
    ///
    /// # Parameters
    /// ```
    /// s: a Vec containing the small molecules
    /// mul: a Vec containing the multiples of the molecules
    /// min_dist: the allowed minimum distance between the small molecules (in Angstrom)
    /// ```
    ///
    /// # Examples
    /// ```
    /// let s: System = System::auto_super_cell(s, mul, 3.0);
    /// ```
    pub fn auto_super_cell(s: &Vec<System>, mul: &Vec<usize>, min_dist: f64) -> Self
    {
        let mut error_message: String = String::new();
        let min: usize = min_dist.ceil() as usize;

        for i in (2*min + 1)..(2*min + 100)
        {
            let size: [f64; 3] = [i as f64, i as f64, i as f64];
            let supercell: Result<System, String> = Self::super_cell(s, mul, size, min_dist);
            match supercell
            {
                Ok(supercell) => return supercell,
                Err(message) => error_message = message,
            }
        }

        panic!("{}", error_message);
    }





    /// Get the atomic radii (in bohr) for a system
    ///
    /// # Parameters
    /// ```
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// let atomic_radii: Vec<f64> = s.get_atomic_radii();
    /// ```
    pub fn get_atomic_radii(&self) -> Vec<f64>
    {
        let mut atomic_radii: Vec<f64> = Vec::with_capacity(self.natom);
        for i in 0..self.natom
        {
            atomic_radii.push( self.atom_type.as_ref().expect(&error_none_value("self.atom_type"))[i].get_atomic_radius() );
        }

        atomic_radii
    }





    /// Get the distance matrix (in bohr) for a system
    ///
    /// # Parameters
    /// ```
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// let dist_mat: Array2<f64> = s.get_dist_mat();
    /// ```
    pub fn get_dist_mat(&self) -> Array2<f64>
    {
        let mut dist_mat: Array2<f64> = Array2::zeros((self.natom, self.natom));

        let mut dist: f64;
        for i in 0..(self.natom-1)                // For atom i
        {
            for j in (i+1)..self.natom                // For atom j
            {
                dist = ( (self.coord[[i,0]] - self.coord[[j,0]]).powi(2) + (self.coord[[i,1]] - self.coord[[j,1]]).powi(2) + (self.coord[[i,2]] - self.coord[[j,2]]).powi(2) ).sqrt();
                dist_mat[[i, j]] = dist;
                dist_mat[[j, i]] = dist;
            }
        }

        dist_mat
    }





    /// Split a System into molecules by calculating the atomic distance
    ///
    /// # Parameters
    /// ```
    /// atomic_radii: the atomic radii of the system
    /// dist_mat: the distance matrix of the current system
    /// transition_multiple: if the distance of two atoms is small than the multiple of their radii, they are considered connected (belonging to the same molecule)
    /// mol_index: the output indices of the molecules
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// let mol_index: Vec<Vec<usize>> = System::split_into_mol(&atomic_radii, &dist_mat, 1.25);
    /// ```
    pub fn split_into_mol(atomic_radii: &Vec<f64>, dist_mat: &Array2<f64>, transition_multiple: f64) -> Vec<Vec<usize>>
    {
        let mut mol_index: Vec<Vec<usize>> = Vec::new();

        // Initialize the indices for the treated and untreated atoms
        let mut index_untreated: Vec<usize> = Vec::with_capacity(atomic_radii.len());
        for i in 0..atomic_radii.len()
        {
            index_untreated.push(i);
        }

        // Loop until there is no atom untreated
        let mut n: usize;
        let mut m: usize;
        while index_untreated.len() != 0
        {
            // Move an atom into the next molecule and remove the atom out of the untreated list
            let mut index: Vec<usize> = vec![index_untreated[0]];
            index_untreated.remove(0);

            // Loop until all the bonding atoms have been included into the molecule (i.e. index)
            n = 0;                // For the atoms of the next molecule
            while n < index.len()
            {
                // Loop until all the untreated atoms have been judged with atom n of the next molecule
                m = 0;                // For the untreated atoms (i.e. index_untreated)
                while m < index_untreated.len()
                {
                    // If the distance between atom n (of the next molecule) and atom m (in the untreated list) is smaller than transition_multiple * (r_n + r_m), move atom m into the molecule
                    if dist_mat[[index[n], index_untreated[m]]] < (atomic_radii[index[n]] + atomic_radii[index_untreated[m]]) * transition_multiple
                    {
                        index.push(index_untreated[m]);
                        index_untreated.remove(m);
                    }
                    else
                    {
                        m += 1;
                    }
                }
                n += 1;
            }

            // Push the molecule into the molecular list
            mol_index.push(index);
        }

        mol_index
    }





    /// Judge if there are two molecules having a distance smaller than the multiple of the radii of any of their atoms
    ///
    /// # Parameters
    /// ```
    /// mol_index: the molecular index of the system
    /// atomic_radii: the atomic radii of the system
    /// dist_mat: the distance matrix of the current system
    /// transition_multiple: if the distance of two atoms is small than the multiple of their radii, they are considered connected
    /// ```
    ///
    /// # Examples
    /// ```
    /// let : bool = judge_adj_of_mol(&mol_index, &atomic_radii, &adj_mat, &dist_mat, 1.2);
    /// ```
    pub fn judge_adj_of_mol(mol_index: &Vec<Vec<usize>>, atomic_radii: &Vec<f64>, adj_mat: &Array2<i8>, dist_mat: &Array2<f64>, multiple: f64) -> bool
    {
        if mol_index.len() == 1
        {
            return true
        }

        for i in 0..(mol_index.len()-1)                // For molecule i
        {
            for j in (i+1)..mol_index.len()
            {
                for n in 0..mol_index[i].len()
                {
                    for m in 0..mol_index[j].len()
                    {
                        if ( adj_mat[[mol_index[i][n], mol_index[j][m]]] == -1 ) && ( dist_mat[[mol_index[i][n], mol_index[j][m]]] < (atomic_radii[mol_index[i][n]] + atomic_radii[mol_index[j][m]]) * multiple )
                        {
                            return true
                        }
                    }
                }
            }
        }

        return false
    }





    /// Get the adjacency matrix for a system by calculating the atomic distance
    ///
    /// # Parameters
    /// ```
    /// atomic_type: the element type of the atoms in the system
    /// atomic_radii: the atomic radii of the system
    /// dist_mat: the distance matrix of the current system
    /// transition_multiple: if the distance of two atoms is small than the transition_multiple of their radii, they are considered bonded; otherwise, they are considered unbonded
    /// ignored_pair: the pairs of atoms within this list would be ignored
    /// p: a matrix containing the bonded/unbonded information for a system. 1 for bonded, -1 for unbonded, 0 for untreated
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// let adj_mat: Array2<i8> = System::get_adj_mat(&atomic_type, &atomic_radii, &dist_mat, 1.25, &ignored_pair);
    /// ```
    pub fn get_adj_mat(atomic_type: &Vec<Element>, atomic_radii: &Vec<f64>, dist_mat: &Array2<f64>, transition_multiple: f64, ignored_pair: &Vec<(Element, Element)>) -> Array2<i8>
    {
        let mut adj_mat: Array2<i8> = Array2::zeros((atomic_radii.len(), atomic_radii.len()));
        let mut treated: bool;

        // Achieve the adjacency matrix
        for i in 0..(atomic_radii.len()-1)                // For atom i
        {
            for j in (i+1)..atomic_radii.len()                // For atom j
            {
                // Judge if the atomic pair should be treated or not
                treated = true;                // Initialize to be treated
                for k in 0..ignored_pair.len()                // For each ignored pair
                {
                    // If the atomic type of the pair is within the ignored pair, it should not be treated
                    if (atomic_type[i] == ignored_pair[k].0) && (atomic_type[j] == ignored_pair[k].1)
                    {
                        treated = false;
                        break
                    }
                    if (atomic_type[i] == ignored_pair[k].1) && (atomic_type[j] == ignored_pair[k].0)
                    {
                        treated = false;
                        break
                    }
                }

                // if the atomic pair isn't within the ignore_pair list, treated it
                if treated
                {
                    // Update the bonded pairs
                    if dist_mat[[i, j]] < (atomic_radii[i] + atomic_radii[j]) * transition_multiple
                    {
                        adj_mat[[i,j]] = 1;
                        adj_mat[[j,i]] = 1;
                    }
                    // Update the unbonded pairs
                    else
                    {
                        adj_mat[[i,j]] = -1;
                        adj_mat[[j,i]] = -1;
                    }
                }
            }
        }

        adj_mat
    }





    /// Judge if the bonding in current system is changing with respect to the original system
    /// When two unbonded atoms (in original system) forming bond in current system,
    /// or two bonded atoms (in original system) breaking their bond in current system,
    /// output 'True'; otherwise, output 'False'
    ///
    /// # Parameters
    /// ```
    /// atomic_radii: the atomic radii of the system
    /// dist_mat: the distance matrix of the current system
    /// p: a matrix containing the bonded/unbonded/transition information for the orinal system. 1 for bonded, -1 for unbonded, 0 for untreated pair
    /// bonded_multiple: if the distance of two atoms is small than the bonded_multiple of their radii, they are considered bonded
    /// unbonded_multiple: if the distance of two atoms is larger than the unbonded_multiple of their radii, they are considered unbonded
    /// changing: the output result for the judgement
    ///
    /// ```
    ///
    /// # Examples
    /// ```
    /// let changing: bool = System::judge_variation_of_bonding(&atomic_radii, &dist_mat, &p, 1.0, 1.6);
    /// ```
    pub fn judge_variation_of_bonding(atomic_radii: &Vec<f64>, dist_mat: &Array2<f64>, p: &Array2<i8>, bonded_multiple: f64, unbonded_multiple: f64) -> bool
    {
        // Judge the variation of bonding in the current system
        for i in 0..(atomic_radii.len()-1)                // For atom i
        {
            for j in (i+1)..atomic_radii.len()                // For atom j
            {
                match p[[i,j]]
                {
                    // If atom i and atom j are originallly bonded
                    1 =>
                    {
                        if dist_mat[[i, j]] > (atomic_radii[i] + atomic_radii[j]) * unbonded_multiple
                        {
                            return true
                        }
                    },

                    // If atom i and atom j are originallly unbonded
                    -1 =>
                    {
                        if dist_mat[[i, j]] < (atomic_radii[i] + atomic_radii[j]) * bonded_multiple
                        {
                            return true
                        }
                    },

                    // For untreated pairs, do nothing
                    _ => (),
                }
            }
        }

        return false
    }
}










