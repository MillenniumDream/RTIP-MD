//! About the input files.
use crate::common::constants::Element;





/// The structure containing the parameters of the roto-translationally invariant potential (RTIP).
///
/// # Fields
/// ```
/// a0: specify how quickly the Gaussian potential is increasing (Unit: Hartree)
/// sigma: specify the width of the Gaussian potential (Unit: bohr)
/// ```
#[derive(Clone)]
pub struct RtipPara
{
    pub a0: f64,
    pub sigma: f64,
    pub scale_ts_a0: f64,
    pub scale_ts_sigma: Option<f64>,
}





/// The structure containing the parameters of the pathway sampling
///
/// # Fields
/// ```
/// pot_drop: when the real potential energy (i.e. e_cp2k) becomes smaller than (e_max - e_drop), remove the Gaussian potential and perform a local optimization (Unit: Hartree)
/// pot_epsilon: in each steeping searching, when the difference of two adjacent e_total (i.e. e_cp2k+e_rtip) becomes smaller than e_epsilon, stop the searching (Unit: Hartree/atom)
/// f_epsilon: in local optimization without Gaussian potential, when force_rtip becomes smaller than force_epsilon, stop the optimization (Unit: Hartree/(Bohr*atom))
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





#[derive(Clone)]
pub struct MdPara
{
    pub dt: f64,
    pub tau: f64,
    pub temp_bath: f64,
    pub decreasing_multiple: f64,
    pub decreasing_bound: f64,
    pub ignored_pair: Vec<(Element, Element)>,
    pub split_step: Option<usize>,
    pub max_step: usize,
    pub print_step: usize,
}





#[derive(Clone)]
pub struct NnTrainPara
{
    pub training_batch_size: usize,
    pub validation_batch_size: usize,
    pub max_step: usize,
    pub print_step: usize,
    pub input_nn_sub_dir: String,
    pub output_nn_sub_dir: String,
    pub input_data_sub_dir: String,
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
                a0: 0.0005,    //larger, the larger compression
                sigma: 0.75,
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
                dt: 0.5,
                tau: 10.0,  //tau, smaller, the better performance in temperature controlling, but increase faster initially, easy to explose the system.
                temp_bath: 1500.0,
                decreasing_multiple: 2.0,  
                decreasing_bound: 0.5,    // set to be 1, never die, 0.5 means letting them to be farer away with 0.5 * the origianl size 
                ignored_pair: vec![(Element::H, Element::O), (Element::C, Element::O), (Element::H, Element::Ca), (Element::C, Element::Ca), (Element::O, Element::Ca), (Element::Ca, Element::Ca)],
                split_step: Some(100),
                max_step: 100000000,
                print_step: 1,
            },

            // NN training parameters
            nn_train_para: NnTrainPara
            {
                training_batch_size: 10,
                validation_batch_size: 5,
                max_step: 100,
                print_step: 10,
                input_nn_sub_dir: String::from(""),
                output_nn_sub_dir: String::from("new"),
                input_data_sub_dir: String::from("monomer/Gly"),
            },
        }
    }
}










