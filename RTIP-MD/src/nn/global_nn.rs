//! Initialization, saving, and loading of the neural network for each fragment in protein system
use crate::common::constants::Device;
use crate::common::error::*;
use crate::nn::interfragment_descriptor::N_DES;
use std::fs;
use dfdx::shapes::Rank0;
use dfdx::tensor::{Tensor, ZerosTensor, Gradients, OwnedTape, Trace};
use dfdx::tensor_ops::{Backward, AdamConfig};
use dfdx::nn::{BuildModule, SaveToSafetensors, LoadFromSafetensors, ZeroGrads};
use dfdx::nn::modules::{Linear, Tanh, AddInto};
use dfdx::optim::{Adam, Optimizer};





// Define the NN modules for inter-fragment interaction and intra-fragment interaction
type InterfragmentNNSmall =
(
    (Linear<N_DES, 256, f64, Device>, Tanh),
    (Linear<256, 256, f64, Device>, Tanh),
    (Linear<256, 64, f64, Device>, Tanh),
    (Linear<64, 16, f64, Device>, Tanh),
    (Linear<16, 4, f64, Device>, Tanh),
    (Linear<4, 1, f64, Device>, Tanh),
);

type InterfragmentNNLarge =
(
    (Linear<N_DES, 512, f64, Device>, Tanh),
    (Linear<512, 256, f64, Device>, Tanh),
    (Linear<256, 64, f64, Device>, Tanh),
    (Linear<64, 16, f64, Device>, Tanh),
    (Linear<16, 4, f64, Device>, Tanh),
    (Linear<4, 1, f64, Device>, Tanh),
);

type IntrafragmentNN<const I: usize> =
(
    (Linear<I, 64, f64, Device>, Tanh),
    (Linear<64, 256, f64, Device>, Tanh),
    (Linear<256, 64, f64, Device>, Tanh),
    (Linear<64, 16, f64, Device>, Tanh),
    (Linear<16, 4, f64, Device>, Tanh),
    Linear<4, 1, f64, Device>,
);





/// The global neural network for the protein system, containing all the sub-nn for the framents (i.e. atoms, amino acids, and molecules)
///
/// # Fields
/// ```
/// ```
pub struct GlobalNN
{
    pub element_h_nn: InterfragmentNNLarge,
    pub element_b_nn: InterfragmentNNSmall,
    pub element_c_nn: InterfragmentNNLarge,
    pub element_n_nn: InterfragmentNNLarge,
    pub element_o_nn: InterfragmentNNLarge,
    pub element_f_nn: InterfragmentNNSmall,
    pub element_na_nn: InterfragmentNNSmall,
    pub element_mg_nn: InterfragmentNNSmall,
    pub element_si_nn: InterfragmentNNSmall,
    pub element_p_nn: InterfragmentNNSmall,
    pub element_s_nn: InterfragmentNNLarge,
    pub element_cl_nn: InterfragmentNNSmall,
    pub element_k_nn: InterfragmentNNSmall,
    pub element_ca_nn: InterfragmentNNSmall,
    pub element_v_nn: InterfragmentNNSmall,
    pub element_cr_nn: InterfragmentNNSmall,
    pub element_mn_nn: InterfragmentNNSmall,
    pub element_fe_nn: InterfragmentNNSmall,
    pub element_co_nn: InterfragmentNNSmall,
    pub element_ni_nn: InterfragmentNNSmall,
    pub element_cu_nn: InterfragmentNNSmall,
    pub element_zn_nn: InterfragmentNNSmall,
    pub element_as_nn: InterfragmentNNSmall,
    pub element_se_nn: InterfragmentNNSmall,
    pub element_br_nn: InterfragmentNNSmall,
    pub element_mo_nn: InterfragmentNNSmall,
    pub element_cd_nn: InterfragmentNNSmall,
    pub element_sn_nn: InterfragmentNNSmall,
    pub element_i_nn: InterfragmentNNSmall,

    pub amino_acid_gly_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<15>) >,
    pub amino_acid_ala_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<24>) >,
    pub amino_acid_val_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<42>) >,
    pub amino_acid_leu_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<51>) >,
    pub amino_acid_ile_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<51>) >,
    pub amino_acid_ser_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<27>) >,
    pub amino_acid_thr_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<36>) >,
    pub amino_acid_asp_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<30>) >,
    pub amino_acid_ash_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<33>) >,
    pub amino_acid_asn_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<36>) >,
    pub amino_acid_glu_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<39>) >,
    pub amino_acid_glh_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<42>) >,
    pub amino_acid_gln_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<45>) >,
    pub amino_acid_lys_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<60>) >,
    pub amino_acid_lyn_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<57>) >,
    pub amino_acid_arg_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<66>) >,
    pub amino_acid_arn_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<63>) >,
    pub amino_acid_cys_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<27>) >,
    pub amino_acid_cyx_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<24>) >,
    pub amino_acid_met_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<45>) >,
    pub amino_acid_hid_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<45>) >,
    pub amino_acid_hie_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<45>) >,
    pub amino_acid_hip_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<48>) >,
    pub amino_acid_phe_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<54>) >,
    pub amino_acid_tyr_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<57>) >,
    pub amino_acid_trp_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<66>) >,
    pub amino_acid_pro_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<36>) >,

    pub molecule_wat_nn: InterfragmentNNSmall,
}





/// The global Adam optimizer for the global NN
///
/// # Fields
/// ```
/// ```
pub struct GlobalAdam
{
    pub element_h_adam: Adam<InterfragmentNNLarge, f64, Device>,
    pub element_b_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_c_adam: Adam<InterfragmentNNLarge, f64, Device>,
    pub element_n_adam: Adam<InterfragmentNNLarge, f64, Device>,
    pub element_o_adam: Adam<InterfragmentNNLarge, f64, Device>,
    pub element_f_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_na_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_mg_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_si_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_p_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_s_adam: Adam<InterfragmentNNLarge, f64, Device>,
    pub element_cl_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_k_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_ca_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_v_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_cr_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_mn_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_fe_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_co_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_ni_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_cu_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_zn_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_as_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_se_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_br_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_mo_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_cd_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_sn_adam: Adam<InterfragmentNNSmall, f64, Device>,
    pub element_i_adam: Adam<InterfragmentNNSmall, f64, Device>,

    pub amino_acid_gly_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<15>)> , f64, Device>,
    pub amino_acid_ala_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<24>)> , f64, Device>,
    pub amino_acid_val_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<42>)> , f64, Device>,
    pub amino_acid_leu_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<51>)> , f64, Device>,
    pub amino_acid_ile_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<51>)> , f64, Device>,
    pub amino_acid_ser_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<27>)> , f64, Device>,
    pub amino_acid_thr_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<36>)> , f64, Device>,
    pub amino_acid_asp_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<30>)> , f64, Device>,
    pub amino_acid_ash_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<33>)> , f64, Device>,
    pub amino_acid_asn_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<36>)> , f64, Device>,
    pub amino_acid_glu_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<39>)> , f64, Device>,
    pub amino_acid_glh_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<42>)> , f64, Device>,
    pub amino_acid_gln_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<45>)> , f64, Device>,
    pub amino_acid_lys_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<60>)> , f64, Device>,
    pub amino_acid_lyn_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<57>)> , f64, Device>,
    pub amino_acid_arg_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<66>)> , f64, Device>,
    pub amino_acid_arn_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<63>)> , f64, Device>,
    pub amino_acid_cys_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<27>)> , f64, Device>,
    pub amino_acid_cyx_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<24>)> , f64, Device>,
    pub amino_acid_met_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<45>)> , f64, Device>,
    pub amino_acid_hid_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<45>)> , f64, Device>,
    pub amino_acid_hie_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<45>)> , f64, Device>,
    pub amino_acid_hip_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<48>)> , f64, Device>,
    pub amino_acid_phe_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<54>)> , f64, Device>,
    pub amino_acid_tyr_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<57>)> , f64, Device>,
    pub amino_acid_trp_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<66>)> , f64, Device>,
    pub amino_acid_pro_adam: Adam< AddInto<(InterfragmentNNSmall, IntrafragmentNN<36>)> , f64, Device>,

    pub molecule_wat_adam: Adam<InterfragmentNNSmall, f64, Device>,
}










impl GlobalNN
{
    /// Construct a new global NN for the protein system
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn new() -> Self
    {
        // Define a Device (CPU or Cuda) to build NNs
        let dev: Device = Device::seed_from_u64(1314);

        // Build NN for each FragmentType
        let element_h_nn: InterfragmentNNLarge = BuildModule::build(&dev);
        let element_b_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_c_nn: InterfragmentNNLarge = BuildModule::build(&dev);
        let element_n_nn: InterfragmentNNLarge = BuildModule::build(&dev);
        let element_o_nn: InterfragmentNNLarge = BuildModule::build(&dev);
        let element_f_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_na_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_mg_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_si_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_p_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_s_nn: InterfragmentNNLarge = BuildModule::build(&dev);
        let element_cl_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_k_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_ca_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_v_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_cr_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_mn_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_fe_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_co_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_ni_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_cu_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_zn_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_as_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_se_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_br_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_mo_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_cd_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_sn_nn: InterfragmentNNSmall = BuildModule::build(&dev);
        let element_i_nn: InterfragmentNNSmall = BuildModule::build(&dev);

        let amino_acid_gly_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<15>) > = BuildModule::build(&dev);
        let amino_acid_ala_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<24>) > = BuildModule::build(&dev);
        let amino_acid_val_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<42>) > = BuildModule::build(&dev);
        let amino_acid_leu_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<51>) > = BuildModule::build(&dev);
        let amino_acid_ile_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<51>) > = BuildModule::build(&dev);
        let amino_acid_ser_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<27>) > = BuildModule::build(&dev);
        let amino_acid_thr_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<36>) > = BuildModule::build(&dev);
        let amino_acid_asp_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<30>) > = BuildModule::build(&dev);
        let amino_acid_ash_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<33>) > = BuildModule::build(&dev);
        let amino_acid_asn_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<36>) > = BuildModule::build(&dev);
        let amino_acid_glu_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<39>) > = BuildModule::build(&dev);
        let amino_acid_glh_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<42>) > = BuildModule::build(&dev);
        let amino_acid_gln_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<45>) > = BuildModule::build(&dev);
        let amino_acid_lys_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<60>) > = BuildModule::build(&dev);
        let amino_acid_lyn_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<57>) > = BuildModule::build(&dev);
        let amino_acid_arg_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<66>) > = BuildModule::build(&dev);
        let amino_acid_arn_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<63>) > = BuildModule::build(&dev);
        let amino_acid_cys_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<27>) > = BuildModule::build(&dev);
        let amino_acid_cyx_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<24>) > = BuildModule::build(&dev);
        let amino_acid_met_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<45>) > = BuildModule::build(&dev);
        let amino_acid_hid_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<45>) > = BuildModule::build(&dev);
        let amino_acid_hie_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<45>) > = BuildModule::build(&dev);
        let amino_acid_hip_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<48>) > = BuildModule::build(&dev);
        let amino_acid_phe_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<54>) > = BuildModule::build(&dev);
        let amino_acid_tyr_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<57>) > = BuildModule::build(&dev);
        let amino_acid_trp_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<66>) > = BuildModule::build(&dev);
        let amino_acid_pro_nn: AddInto< (InterfragmentNNSmall, IntrafragmentNN<36>) > = BuildModule::build(&dev);

        let molecule_wat_nn: InterfragmentNNSmall = BuildModule::build(&dev);

        // Return the global NN
        GlobalNN
        {
            element_h_nn,
            element_b_nn,
            element_c_nn,
            element_n_nn,
            element_o_nn,
            element_f_nn,
            element_na_nn,
            element_mg_nn,
            element_si_nn,
            element_p_nn,
            element_s_nn,
            element_cl_nn,
            element_k_nn,
            element_ca_nn,
            element_v_nn,
            element_cr_nn,
            element_mn_nn,
            element_fe_nn,
            element_co_nn,
            element_ni_nn,
            element_cu_nn,
            element_zn_nn,
            element_as_nn,
            element_se_nn,
            element_br_nn,
            element_mo_nn,
            element_cd_nn,
            element_sn_nn,
            element_i_nn,

            amino_acid_gly_nn,
            amino_acid_ala_nn,
            amino_acid_val_nn,
            amino_acid_leu_nn,
            amino_acid_ile_nn,
            amino_acid_ser_nn,
            amino_acid_thr_nn,
            amino_acid_asp_nn,
            amino_acid_ash_nn,
            amino_acid_asn_nn,
            amino_acid_glu_nn,
            amino_acid_glh_nn,
            amino_acid_gln_nn,
            amino_acid_lys_nn,
            amino_acid_lyn_nn,
            amino_acid_arg_nn,
            amino_acid_arn_nn,
            amino_acid_cys_nn,
            amino_acid_cyx_nn,
            amino_acid_met_nn,
            amino_acid_hid_nn,
            amino_acid_hie_nn,
            amino_acid_hip_nn,
            amino_acid_phe_nn,
            amino_acid_tyr_nn,
            amino_acid_trp_nn,
            amino_acid_pro_nn,

            molecule_wat_nn,
        }
    }





    /// Save the global NN to the dir 'nn/sub_dir'
    ///
    /// # Parameters
    /// ```
    /// sub_dir: the sub-directory with respect to 'nn'
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn save(&self, sub_dir: &str)
    {
        // If directory 'nn' already exist, do nothing; otherwise, create the directory
        let dir_exist = fs::metadata("nn");
        match dir_exist
        {
            Ok(_) => (),
            Err(_) => fs::create_dir("nn").expect(&error_dir("creating", "nn")),
        }

        // If directory 'nn/sub_dir' already exist, do nothing; otherwise, create the directory
        let dir: String = format!("nn/{sub_dir}");
        let dir_exist = fs::metadata(&dir);
        match dir_exist
        {
            Ok(_) => (),
            Err(_) => fs::create_dir(&dir).expect(&error_dir("creating", &dir)),
        }

        // Save the NNs
        self.element_h_nn.save_safetensors(format!("{dir}/Element_H_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_H_NN.safetensors")));
        self.element_b_nn.save_safetensors(format!("{dir}/Element_B_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_B_NN.safetensors")));
        self.element_c_nn.save_safetensors(format!("{dir}/Element_C_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_C_NN.safetensors")));
        self.element_n_nn.save_safetensors(format!("{dir}/Element_N_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_N_NN.safetensors")));
        self.element_o_nn.save_safetensors(format!("{dir}/Element_O_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_O_NN.safetensors")));
        self.element_f_nn.save_safetensors(format!("{dir}/Element_F_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_F_NN.safetensors")));
        self.element_na_nn.save_safetensors(format!("{dir}/Element_Na_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Na_NN.safetensors")));
        self.element_mg_nn.save_safetensors(format!("{dir}/Element_Mg_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Mg_NN.safetensors")));
        self.element_si_nn.save_safetensors(format!("{dir}/Element_Si_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Si_NN.safetensors")));
        self.element_p_nn.save_safetensors(format!("{dir}/Element_P_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_P_NN.safetensors")));
        self.element_s_nn.save_safetensors(format!("{dir}/Element_S_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_S_NN.safetensors")));
        self.element_cl_nn.save_safetensors(format!("{dir}/Element_Cl_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Cl_NN.safetensors")));
        self.element_k_nn.save_safetensors(format!("{dir}/Element_K_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_K_NN.safetensors")));
        self.element_ca_nn.save_safetensors(format!("{dir}/Element_Ca_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Ca_NN.safetensors")));
        self.element_v_nn.save_safetensors(format!("{dir}/Element_V_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_V_NN.safetensors")));
        self.element_cr_nn.save_safetensors(format!("{dir}/Element_Cr_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Cr_NN.safetensors")));
        self.element_mn_nn.save_safetensors(format!("{dir}/Element_Mn_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Mn_NN.safetensors")));
        self.element_fe_nn.save_safetensors(format!("{dir}/Element_Fe_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Fe_NN.safetensors")));
        self.element_co_nn.save_safetensors(format!("{dir}/Element_Co_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Co_NN.safetensors")));
        self.element_ni_nn.save_safetensors(format!("{dir}/Element_Ni_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Ni_NN.safetensors")));
        self.element_cu_nn.save_safetensors(format!("{dir}/Element_Cu_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Cu_NN.safetensors")));
        self.element_zn_nn.save_safetensors(format!("{dir}/Element_Zn_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Zn_NN.safetensors")));
        self.element_as_nn.save_safetensors(format!("{dir}/Element_As_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_As_NN.safetensors")));
        self.element_se_nn.save_safetensors(format!("{dir}/Element_Se_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Se_NN.safetensors")));
        self.element_br_nn.save_safetensors(format!("{dir}/Element_Br_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Br_NN.safetensors")));
        self.element_mo_nn.save_safetensors(format!("{dir}/Element_Mo_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Mo_NN.safetensors")));
        self.element_cd_nn.save_safetensors(format!("{dir}/Element_Cd_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Cd_NN.safetensors")));
        self.element_sn_nn.save_safetensors(format!("{dir}/Element_Sn_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_Sn_NN.safetensors")));
        self.element_i_nn.save_safetensors(format!("{dir}/Element_I_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Element_I_NN.safetensors")));

        self.amino_acid_gly_nn.save_safetensors(format!("{dir}/Amino_Acid_Gly_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Gly_NN.safetensors")));
        self.amino_acid_ala_nn.save_safetensors(format!("{dir}/Amino_Acid_Ala_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Ala_NN.safetensors")));
        self.amino_acid_val_nn.save_safetensors(format!("{dir}/Amino_Acid_Val_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Val_NN.safetensors")));
        self.amino_acid_leu_nn.save_safetensors(format!("{dir}/Amino_Acid_Leu_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Leu_NN.safetensors")));
        self.amino_acid_ile_nn.save_safetensors(format!("{dir}/Amino_Acid_Ile_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Ile_NN.safetensors")));
        self.amino_acid_ser_nn.save_safetensors(format!("{dir}/Amino_Acid_Ser_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Ser_NN.safetensors")));
        self.amino_acid_thr_nn.save_safetensors(format!("{dir}/Amino_Acid_Thr_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Thr_NN.safetensors")));
        self.amino_acid_asp_nn.save_safetensors(format!("{dir}/Amino_Acid_Asp_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Asp_NN.safetensors")));
        self.amino_acid_ash_nn.save_safetensors(format!("{dir}/Amino_Acid_Ash_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Ash_NN.safetensors")));
        self.amino_acid_asn_nn.save_safetensors(format!("{dir}/Amino_Acid_Asn_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Asn_NN.safetensors")));
        self.amino_acid_glu_nn.save_safetensors(format!("{dir}/Amino_Acid_Glu_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Glu_NN.safetensors")));
        self.amino_acid_glh_nn.save_safetensors(format!("{dir}/Amino_Acid_Glh_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Glh_NN.safetensors")));
        self.amino_acid_gln_nn.save_safetensors(format!("{dir}/Amino_Acid_Gln_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Gln_NN.safetensors")));
        self.amino_acid_lys_nn.save_safetensors(format!("{dir}/Amino_Acid_Lys_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Lys_NN.safetensors")));
        self.amino_acid_lyn_nn.save_safetensors(format!("{dir}/Amino_Acid_Lyn_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Lyn_NN.safetensors")));
        self.amino_acid_arg_nn.save_safetensors(format!("{dir}/Amino_Acid_Arg_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Arg_NN.safetensors")));
        self.amino_acid_arn_nn.save_safetensors(format!("{dir}/Amino_Acid_Arn_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Arn_NN.safetensors")));
        self.amino_acid_cys_nn.save_safetensors(format!("{dir}/Amino_Acid_Cys_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Cys_NN.safetensors")));
        self.amino_acid_cyx_nn.save_safetensors(format!("{dir}/Amino_Acid_Cyx_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Cyx_NN.safetensors")));
        self.amino_acid_met_nn.save_safetensors(format!("{dir}/Amino_Acid_Met_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Met_NN.safetensors")));
        self.amino_acid_hid_nn.save_safetensors(format!("{dir}/Amino_Acid_Hid_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Hid_NN.safetensors")));
        self.amino_acid_hie_nn.save_safetensors(format!("{dir}/Amino_Acid_Hie_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Hie_NN.safetensors")));
        self.amino_acid_hip_nn.save_safetensors(format!("{dir}/Amino_Acid_Hip_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Hip_NN.safetensors")));
        self.amino_acid_phe_nn.save_safetensors(format!("{dir}/Amino_Acid_Phe_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Phe_NN.safetensors")));
        self.amino_acid_tyr_nn.save_safetensors(format!("{dir}/Amino_Acid_Tyr_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Tyr_NN.safetensors")));
        self.amino_acid_trp_nn.save_safetensors(format!("{dir}/Amino_Acid_Trp_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Trp_NN.safetensors")));
        self.amino_acid_pro_nn.save_safetensors(format!("{dir}/Amino_Acid_Pro_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Amino_Acid_Pro_NN.safetensors")));

        self.molecule_wat_nn.save_safetensors(format!("{dir}/Molecule_Wat_NN.safetensors")).expect(&error_file("creating", &format!("{dir}/Molecule_Wat_NN.safetensors")));
    }





    /// Load the global NN from the dir 'nn/sub_dir'
    ///
    /// # Parameters
    /// ```
    /// sub_dir: the sub-directory with respect to 'nn'
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn load(sub_dir: &str) -> Self
    {
        let mut global_nn: GlobalNN = GlobalNN::new();
        let dir: String = format!("nn/{sub_dir}");

        // Read the NNs
        global_nn.element_h_nn.load_safetensors(format!("{dir}/Element_H_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_H_NN.safetensors")));
        global_nn.element_b_nn.load_safetensors(format!("{dir}/Element_B_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_B_NN.safetensors")));
        global_nn.element_c_nn.load_safetensors(format!("{dir}/Element_C_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_C_NN.safetensors")));
        global_nn.element_n_nn.load_safetensors(format!("{dir}/Element_N_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_N_NN.safetensors")));
        global_nn.element_o_nn.load_safetensors(format!("{dir}/Element_O_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_O_NN.safetensors")));
        global_nn.element_f_nn.load_safetensors(format!("{dir}/Element_F_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_F_NN.safetensors")));
        global_nn.element_na_nn.load_safetensors(format!("{dir}/Element_Na_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Na_NN.safetensors")));
        global_nn.element_mg_nn.load_safetensors(format!("{dir}/Element_Mg_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Mg_NN.safetensors")));
        global_nn.element_si_nn.load_safetensors(format!("{dir}/Element_Si_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Si_NN.safetensors")));
        global_nn.element_p_nn.load_safetensors(format!("{dir}/Element_P_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_P_NN.safetensors")));
        global_nn.element_s_nn.load_safetensors(format!("{dir}/Element_S_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_S_NN.safetensors")));
        global_nn.element_cl_nn.load_safetensors(format!("{dir}/Element_Cl_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Cl_NN.safetensors")));
        global_nn.element_k_nn.load_safetensors(format!("{dir}/Element_K_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_K_NN.safetensors")));
        global_nn.element_ca_nn.load_safetensors(format!("{dir}/Element_Ca_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Ca_NN.safetensors")));
        global_nn.element_v_nn.load_safetensors(format!("{dir}/Element_V_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_V_NN.safetensors")));
        global_nn.element_cr_nn.load_safetensors(format!("{dir}/Element_Cr_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Cr_NN.safetensors")));
        global_nn.element_mn_nn.load_safetensors(format!("{dir}/Element_Mn_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Mn_NN.safetensors")));
        global_nn.element_fe_nn.load_safetensors(format!("{dir}/Element_Fe_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Fe_NN.safetensors")));
        global_nn.element_co_nn.load_safetensors(format!("{dir}/Element_Co_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Co_NN.safetensors")));
        global_nn.element_ni_nn.load_safetensors(format!("{dir}/Element_Ni_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Ni_NN.safetensors")));
        global_nn.element_cu_nn.load_safetensors(format!("{dir}/Element_Cu_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Cu_NN.safetensors")));
        global_nn.element_zn_nn.load_safetensors(format!("{dir}/Element_Zn_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Zn_NN.safetensors")));
        global_nn.element_as_nn.load_safetensors(format!("{dir}/Element_As_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_As_NN.safetensors")));
        global_nn.element_se_nn.load_safetensors(format!("{dir}/Element_Se_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Se_NN.safetensors")));
        global_nn.element_br_nn.load_safetensors(format!("{dir}/Element_Br_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Br_NN.safetensors")));
        global_nn.element_mo_nn.load_safetensors(format!("{dir}/Element_Mo_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Mo_NN.safetensors")));
        global_nn.element_cd_nn.load_safetensors(format!("{dir}/Element_Cd_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Cd_NN.safetensors")));
        global_nn.element_sn_nn.load_safetensors(format!("{dir}/Element_Sn_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_Sn_NN.safetensors")));
        global_nn.element_i_nn.load_safetensors(format!("{dir}/Element_I_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Element_I_NN.safetensors")));

        global_nn.amino_acid_gly_nn.load_safetensors(format!("{dir}/Amino_Acid_Gly_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Gly_NN.safetensors")));
        global_nn.amino_acid_ala_nn.load_safetensors(format!("{dir}/Amino_Acid_Ala_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Ala_NN.safetensors")));
        global_nn.amino_acid_val_nn.load_safetensors(format!("{dir}/Amino_Acid_Val_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Val_NN.safetensors")));
        global_nn.amino_acid_leu_nn.load_safetensors(format!("{dir}/Amino_Acid_Leu_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Leu_NN.safetensors")));
        global_nn.amino_acid_ile_nn.load_safetensors(format!("{dir}/Amino_Acid_Ile_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Ile_NN.safetensors")));
        global_nn.amino_acid_ser_nn.load_safetensors(format!("{dir}/Amino_Acid_Ser_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Ser_NN.safetensors")));
        global_nn.amino_acid_thr_nn.load_safetensors(format!("{dir}/Amino_Acid_Thr_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Thr_NN.safetensors")));
        global_nn.amino_acid_asp_nn.load_safetensors(format!("{dir}/Amino_Acid_Asp_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Asp_NN.safetensors")));
        global_nn.amino_acid_ash_nn.load_safetensors(format!("{dir}/Amino_Acid_Ash_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Ash_NN.safetensors")));
        global_nn.amino_acid_asn_nn.load_safetensors(format!("{dir}/Amino_Acid_Asn_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Asn_NN.safetensors")));
        global_nn.amino_acid_glu_nn.load_safetensors(format!("{dir}/Amino_Acid_Glu_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Glu_NN.safetensors")));
        global_nn.amino_acid_glh_nn.load_safetensors(format!("{dir}/Amino_Acid_Glh_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Glh_NN.safetensors")));
        global_nn.amino_acid_gln_nn.load_safetensors(format!("{dir}/Amino_Acid_Gln_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Gln_NN.safetensors")));
        global_nn.amino_acid_lys_nn.load_safetensors(format!("{dir}/Amino_Acid_Lys_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Lys_NN.safetensors")));
        global_nn.amino_acid_lyn_nn.load_safetensors(format!("{dir}/Amino_Acid_Lyn_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Lyn_NN.safetensors")));
        global_nn.amino_acid_arg_nn.load_safetensors(format!("{dir}/Amino_Acid_Arg_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Arg_NN.safetensors")));
        global_nn.amino_acid_arn_nn.load_safetensors(format!("{dir}/Amino_Acid_Arn_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Arn_NN.safetensors")));
        global_nn.amino_acid_cys_nn.load_safetensors(format!("{dir}/Amino_Acid_Cys_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Cys_NN.safetensors")));
        global_nn.amino_acid_cyx_nn.load_safetensors(format!("{dir}/Amino_Acid_Cyx_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Cyx_NN.safetensors")));
        global_nn.amino_acid_met_nn.load_safetensors(format!("{dir}/Amino_Acid_Met_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Met_NN.safetensors")));
        global_nn.amino_acid_hid_nn.load_safetensors(format!("{dir}/Amino_Acid_Hid_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Hid_NN.safetensors")));
        global_nn.amino_acid_hie_nn.load_safetensors(format!("{dir}/Amino_Acid_Hie_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Hie_NN.safetensors")));
        global_nn.amino_acid_hip_nn.load_safetensors(format!("{dir}/Amino_Acid_Hip_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Hip_NN.safetensors")));
        global_nn.amino_acid_phe_nn.load_safetensors(format!("{dir}/Amino_Acid_Phe_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Phe_NN.safetensors")));
        global_nn.amino_acid_tyr_nn.load_safetensors(format!("{dir}/Amino_Acid_Tyr_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Tyr_NN.safetensors")));
        global_nn.amino_acid_trp_nn.load_safetensors(format!("{dir}/Amino_Acid_Trp_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Trp_NN.safetensors")));
        global_nn.amino_acid_pro_nn.load_safetensors(format!("{dir}/Amino_Acid_Pro_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Amino_Acid_Pro_NN.safetensors")));

        global_nn.molecule_wat_nn.load_safetensors(format!("{dir}/Molecule_Wat_NN.safetensors")).expect(&error_file("reading", &format!("{dir}/Molecule_Wat_NN.safetensors")));

        global_nn
    }





    /// Allocate gradients for all the neural networks in global NN
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn alloc_grads(&self) -> Gradients<f64, Device>
    {
        // Define a Device (CPU or Cuda) to build NNs
        let dev: Device = Device::seed_from_u64(1314);

        // Allocate gradient for all the neural networks
        let element_h_nn_grads: Gradients<f64, Device> = self.element_h_nn.alloc_grads();
        let element_b_nn_grads: Gradients<f64, Device> = self.element_b_nn.alloc_grads();
        let element_c_nn_grads: Gradients<f64, Device> = self.element_c_nn.alloc_grads();
        let element_n_nn_grads: Gradients<f64, Device> = self.element_n_nn.alloc_grads();
        let element_o_nn_grads: Gradients<f64, Device> = self.element_o_nn.alloc_grads();
        let element_f_nn_grads: Gradients<f64, Device> = self.element_f_nn.alloc_grads();
        let element_na_nn_grads: Gradients<f64, Device> = self.element_na_nn.alloc_grads();
        let element_mg_nn_grads: Gradients<f64, Device> = self.element_mg_nn.alloc_grads();
        let element_si_nn_grads: Gradients<f64, Device> = self.element_si_nn.alloc_grads();
        let element_p_nn_grads: Gradients<f64, Device> = self.element_p_nn.alloc_grads();
        let element_s_nn_grads: Gradients<f64, Device> = self.element_s_nn.alloc_grads();
        let element_cl_nn_grads: Gradients<f64, Device> = self.element_cl_nn.alloc_grads();
        let element_k_nn_grads: Gradients<f64, Device> = self.element_k_nn.alloc_grads();
        let element_ca_nn_grads: Gradients<f64, Device> = self.element_ca_nn.alloc_grads();
        let element_v_nn_grads: Gradients<f64, Device> = self.element_v_nn.alloc_grads();
        let element_cr_nn_grads: Gradients<f64, Device> = self.element_cr_nn.alloc_grads();
        let element_mn_nn_grads: Gradients<f64, Device> = self.element_mn_nn.alloc_grads();
        let element_fe_nn_grads: Gradients<f64, Device> = self.element_fe_nn.alloc_grads();
        let element_co_nn_grads: Gradients<f64, Device> = self.element_co_nn.alloc_grads();
        let element_ni_nn_grads: Gradients<f64, Device> = self.element_ni_nn.alloc_grads();
        let element_cu_nn_grads: Gradients<f64, Device> = self.element_cu_nn.alloc_grads();
        let element_zn_nn_grads: Gradients<f64, Device> = self.element_zn_nn.alloc_grads();
        let element_as_nn_grads: Gradients<f64, Device> = self.element_as_nn.alloc_grads();
        let element_se_nn_grads: Gradients<f64, Device> = self.element_se_nn.alloc_grads();
        let element_br_nn_grads: Gradients<f64, Device> = self.element_br_nn.alloc_grads();
        let element_mo_nn_grads: Gradients<f64, Device> = self.element_mo_nn.alloc_grads();
        let element_cd_nn_grads: Gradients<f64, Device> = self.element_cd_nn.alloc_grads();
        let element_sn_nn_grads: Gradients<f64, Device> = self.element_sn_nn.alloc_grads();
        let element_i_nn_grads: Gradients<f64, Device> = self.element_i_nn.alloc_grads();

        let amino_acid_gly_nn_grads: Gradients<f64, Device> = self.amino_acid_gly_nn.alloc_grads();
        let amino_acid_ala_nn_grads: Gradients<f64, Device> = self.amino_acid_ala_nn.alloc_grads();
        let amino_acid_val_nn_grads: Gradients<f64, Device> = self.amino_acid_val_nn.alloc_grads();
        let amino_acid_leu_nn_grads: Gradients<f64, Device> = self.amino_acid_leu_nn.alloc_grads();
        let amino_acid_ile_nn_grads: Gradients<f64, Device> = self.amino_acid_ile_nn.alloc_grads();
        let amino_acid_ser_nn_grads: Gradients<f64, Device> = self.amino_acid_ser_nn.alloc_grads();
        let amino_acid_thr_nn_grads: Gradients<f64, Device> = self.amino_acid_thr_nn.alloc_grads();
        let amino_acid_asp_nn_grads: Gradients<f64, Device> = self.amino_acid_asp_nn.alloc_grads();
        let amino_acid_ash_nn_grads: Gradients<f64, Device> = self.amino_acid_ash_nn.alloc_grads();
        let amino_acid_asn_nn_grads: Gradients<f64, Device> = self.amino_acid_asn_nn.alloc_grads();
        let amino_acid_glu_nn_grads: Gradients<f64, Device> = self.amino_acid_glu_nn.alloc_grads();
        let amino_acid_glh_nn_grads: Gradients<f64, Device> = self.amino_acid_glh_nn.alloc_grads();
        let amino_acid_gln_nn_grads: Gradients<f64, Device> = self.amino_acid_gln_nn.alloc_grads();
        let amino_acid_lys_nn_grads: Gradients<f64, Device> = self.amino_acid_lys_nn.alloc_grads();
        let amino_acid_lyn_nn_grads: Gradients<f64, Device> = self.amino_acid_lyn_nn.alloc_grads();
        let amino_acid_arg_nn_grads: Gradients<f64, Device> = self.amino_acid_arg_nn.alloc_grads();
        let amino_acid_arn_nn_grads: Gradients<f64, Device> = self.amino_acid_arn_nn.alloc_grads();
        let amino_acid_cys_nn_grads: Gradients<f64, Device> = self.amino_acid_cys_nn.alloc_grads();
        let amino_acid_cyx_nn_grads: Gradients<f64, Device> = self.amino_acid_cyx_nn.alloc_grads();
        let amino_acid_met_nn_grads: Gradients<f64, Device> = self.amino_acid_met_nn.alloc_grads();
        let amino_acid_hid_nn_grads: Gradients<f64, Device> = self.amino_acid_hid_nn.alloc_grads();
        let amino_acid_hie_nn_grads: Gradients<f64, Device> = self.amino_acid_hie_nn.alloc_grads();
        let amino_acid_hip_nn_grads: Gradients<f64, Device> = self.amino_acid_hip_nn.alloc_grads();
        let amino_acid_phe_nn_grads: Gradients<f64, Device> = self.amino_acid_phe_nn.alloc_grads();
        let amino_acid_tyr_nn_grads: Gradients<f64, Device> = self.amino_acid_tyr_nn.alloc_grads();
        let amino_acid_trp_nn_grads: Gradients<f64, Device> = self.amino_acid_trp_nn.alloc_grads();
        let amino_acid_pro_nn_grads: Gradients<f64, Device> = self.amino_acid_pro_nn.alloc_grads();

        let molecule_wat_nn_grads: Gradients<f64, Device> = self.molecule_wat_nn.alloc_grads();

        // Pass the gradients to some temporary tensors
        let element_h_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_h_nn_grads);
        let element_b_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_b_nn_grads);
        let element_c_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_c_nn_grads);
        let element_n_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_n_nn_grads);
        let element_o_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_o_nn_grads);
        let element_f_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_f_nn_grads);
        let element_na_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_na_nn_grads);
        let element_mg_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_mg_nn_grads);
        let element_si_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_si_nn_grads);
        let element_p_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_p_nn_grads);
        let element_s_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_s_nn_grads);
        let element_cl_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_cl_nn_grads);
        let element_k_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_k_nn_grads);
        let element_ca_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_ca_nn_grads);
        let element_v_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_v_nn_grads);
        let element_cr_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_cr_nn_grads);
        let element_mn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_mn_nn_grads);
        let element_fe_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_fe_nn_grads);
        let element_co_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_co_nn_grads);
        let element_ni_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_ni_nn_grads);
        let element_cu_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_cu_nn_grads);
        let element_zn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_zn_nn_grads);
        let element_as_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_as_nn_grads);
        let element_se_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_se_nn_grads);
        let element_br_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_br_nn_grads);
        let element_mo_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_mo_nn_grads);
        let element_cd_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_cd_nn_grads);
        let element_sn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_sn_nn_grads);
        let element_i_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(element_i_nn_grads);

        let amino_acid_gly_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_gly_nn_grads);
        let amino_acid_ala_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_ala_nn_grads);
        let amino_acid_val_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_val_nn_grads);
        let amino_acid_leu_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_leu_nn_grads);
        let amino_acid_ile_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_ile_nn_grads);
        let amino_acid_ser_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_ser_nn_grads);
        let amino_acid_thr_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_thr_nn_grads);
        let amino_acid_asp_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_asp_nn_grads);
        let amino_acid_ash_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_ash_nn_grads);
        let amino_acid_asn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_asn_nn_grads);
        let amino_acid_glu_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_glu_nn_grads);
        let amino_acid_glh_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_glh_nn_grads);
        let amino_acid_gln_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_gln_nn_grads);
        let amino_acid_lys_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_lys_nn_grads);
        let amino_acid_lyn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_lyn_nn_grads);
        let amino_acid_arg_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_arg_nn_grads);
        let amino_acid_arn_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_arn_nn_grads);
        let amino_acid_cys_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_cys_nn_grads);
        let amino_acid_cyx_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_cyx_nn_grads);
        let amino_acid_met_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_met_nn_grads);
        let amino_acid_hid_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_hid_nn_grads);
        let amino_acid_hie_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_hie_nn_grads);
        let amino_acid_hip_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_hip_nn_grads);
        let amino_acid_phe_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_phe_nn_grads);
        let amino_acid_tyr_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_tyr_nn_grads);
        let amino_acid_trp_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_trp_nn_grads);
        let amino_acid_pro_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(amino_acid_pro_nn_grads);

        let molecule_wat_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = dev.zeros().traced(molecule_wat_nn_grads);

        // Combine the gradients and return
        let global_temporary_tensor: Tensor<Rank0, f64, Device, OwnedTape<f64, Device>> = element_h_temporary_tensor + element_b_temporary_tensor + element_c_temporary_tensor + element_n_temporary_tensor + element_o_temporary_tensor + element_f_temporary_tensor + element_na_temporary_tensor + element_mg_temporary_tensor + element_si_temporary_tensor + element_p_temporary_tensor + element_s_temporary_tensor + element_cl_temporary_tensor + element_k_temporary_tensor + element_ca_temporary_tensor + element_v_temporary_tensor + element_cr_temporary_tensor + element_mn_temporary_tensor + element_fe_temporary_tensor + element_co_temporary_tensor + element_ni_temporary_tensor + element_cu_temporary_tensor + element_zn_temporary_tensor + element_as_temporary_tensor + element_se_temporary_tensor + element_br_temporary_tensor + element_mo_temporary_tensor + element_cd_temporary_tensor + element_sn_temporary_tensor + element_i_temporary_tensor + amino_acid_gly_temporary_tensor + amino_acid_ala_temporary_tensor + amino_acid_val_temporary_tensor + amino_acid_leu_temporary_tensor + amino_acid_ile_temporary_tensor + amino_acid_ser_temporary_tensor + amino_acid_thr_temporary_tensor + amino_acid_asp_temporary_tensor + amino_acid_ash_temporary_tensor + amino_acid_asn_temporary_tensor + amino_acid_glu_temporary_tensor + amino_acid_glh_temporary_tensor + amino_acid_gln_temporary_tensor + amino_acid_lys_temporary_tensor + amino_acid_lyn_temporary_tensor + amino_acid_arg_temporary_tensor + amino_acid_arn_temporary_tensor + amino_acid_cys_temporary_tensor + amino_acid_cyx_temporary_tensor + amino_acid_met_temporary_tensor + amino_acid_hid_temporary_tensor + amino_acid_hie_temporary_tensor + amino_acid_hip_temporary_tensor + amino_acid_phe_temporary_tensor + amino_acid_tyr_temporary_tensor + amino_acid_trp_temporary_tensor + amino_acid_pro_temporary_tensor + molecule_wat_temporary_tensor;

        global_temporary_tensor.backward()
    }





    /// Zero all the gradients associated with global NN
    ///
    /// # Parameters
    /// ```
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn zero_grads(&self, global_grads: &mut Gradients<f64, Device>)
    {
        self.element_h_nn.zero_grads(global_grads);
        self.element_b_nn.zero_grads(global_grads);
        self.element_c_nn.zero_grads(global_grads);
        self.element_n_nn.zero_grads(global_grads);
        self.element_o_nn.zero_grads(global_grads);
        self.element_f_nn.zero_grads(global_grads);
        self.element_na_nn.zero_grads(global_grads);
        self.element_mg_nn.zero_grads(global_grads);
        self.element_si_nn.zero_grads(global_grads);
        self.element_p_nn.zero_grads(global_grads);
        self.element_s_nn.zero_grads(global_grads);
        self.element_cl_nn.zero_grads(global_grads);
        self.element_k_nn.zero_grads(global_grads);
        self.element_ca_nn.zero_grads(global_grads);
        self.element_v_nn.zero_grads(global_grads);
        self.element_cr_nn.zero_grads(global_grads);
        self.element_mn_nn.zero_grads(global_grads);
        self.element_fe_nn.zero_grads(global_grads);
        self.element_co_nn.zero_grads(global_grads);
        self.element_ni_nn.zero_grads(global_grads);
        self.element_cu_nn.zero_grads(global_grads);
        self.element_zn_nn.zero_grads(global_grads);
        self.element_as_nn.zero_grads(global_grads);
        self.element_se_nn.zero_grads(global_grads);
        self.element_br_nn.zero_grads(global_grads);
        self.element_mo_nn.zero_grads(global_grads);
        self.element_cd_nn.zero_grads(global_grads);
        self.element_sn_nn.zero_grads(global_grads);
        self.element_i_nn.zero_grads(global_grads);

        self.amino_acid_gly_nn.zero_grads(global_grads);
        self.amino_acid_ala_nn.zero_grads(global_grads);
        self.amino_acid_val_nn.zero_grads(global_grads);
        self.amino_acid_leu_nn.zero_grads(global_grads);
        self.amino_acid_ile_nn.zero_grads(global_grads);
        self.amino_acid_ser_nn.zero_grads(global_grads);
        self.amino_acid_thr_nn.zero_grads(global_grads);
        self.amino_acid_asp_nn.zero_grads(global_grads);
        self.amino_acid_ash_nn.zero_grads(global_grads);
        self.amino_acid_asn_nn.zero_grads(global_grads);
        self.amino_acid_glu_nn.zero_grads(global_grads);
        self.amino_acid_glh_nn.zero_grads(global_grads);
        self.amino_acid_gln_nn.zero_grads(global_grads);
        self.amino_acid_lys_nn.zero_grads(global_grads);
        self.amino_acid_lyn_nn.zero_grads(global_grads);
        self.amino_acid_arg_nn.zero_grads(global_grads);
        self.amino_acid_arn_nn.zero_grads(global_grads);
        self.amino_acid_cys_nn.zero_grads(global_grads);
        self.amino_acid_cyx_nn.zero_grads(global_grads);
        self.amino_acid_met_nn.zero_grads(global_grads);
        self.amino_acid_hid_nn.zero_grads(global_grads);
        self.amino_acid_hie_nn.zero_grads(global_grads);
        self.amino_acid_hip_nn.zero_grads(global_grads);
        self.amino_acid_phe_nn.zero_grads(global_grads);
        self.amino_acid_tyr_nn.zero_grads(global_grads);
        self.amino_acid_trp_nn.zero_grads(global_grads);
        self.amino_acid_pro_nn.zero_grads(global_grads);

        self.molecule_wat_nn.zero_grads(global_grads);
    }
}










impl GlobalAdam
{
    /// Construct a new global Adam optimizer for the global NN
    ///
    /// # Parameters
    /// ```
    /// global_nn: the input global NN of the protein system
    /// adam_config: the input configuration for the global Adam optimizer
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn new(global_nn: &GlobalNN, adam_config: AdamConfig) -> Self
    {
        GlobalAdam
        {
            element_h_adam: Adam::new(&global_nn.element_h_nn, adam_config),
            element_b_adam: Adam::new(&global_nn.element_b_nn, adam_config),
            element_c_adam: Adam::new(&global_nn.element_c_nn, adam_config),
            element_n_adam: Adam::new(&global_nn.element_n_nn, adam_config),
            element_o_adam: Adam::new(&global_nn.element_o_nn, adam_config),
            element_f_adam: Adam::new(&global_nn.element_f_nn, adam_config),
            element_na_adam: Adam::new(&global_nn.element_na_nn, adam_config),
            element_mg_adam: Adam::new(&global_nn.element_mg_nn, adam_config),
            element_si_adam: Adam::new(&global_nn.element_si_nn, adam_config),
            element_p_adam: Adam::new(&global_nn.element_p_nn, adam_config),
            element_s_adam: Adam::new(&global_nn.element_s_nn, adam_config),
            element_cl_adam: Adam::new(&global_nn.element_cl_nn, adam_config),
            element_k_adam: Adam::new(&global_nn.element_k_nn, adam_config),
            element_ca_adam: Adam::new(&global_nn.element_ca_nn, adam_config),
            element_v_adam: Adam::new(&global_nn.element_v_nn, adam_config),
            element_cr_adam: Adam::new(&global_nn.element_cr_nn, adam_config),
            element_mn_adam: Adam::new(&global_nn.element_mn_nn, adam_config),
            element_fe_adam: Adam::new(&global_nn.element_fe_nn, adam_config),
            element_co_adam: Adam::new(&global_nn.element_co_nn, adam_config),
            element_ni_adam: Adam::new(&global_nn.element_ni_nn, adam_config),
            element_cu_adam: Adam::new(&global_nn.element_cu_nn, adam_config),
            element_zn_adam: Adam::new(&global_nn.element_zn_nn, adam_config),
            element_as_adam: Adam::new(&global_nn.element_as_nn, adam_config),
            element_se_adam: Adam::new(&global_nn.element_se_nn, adam_config),
            element_br_adam: Adam::new(&global_nn.element_br_nn, adam_config),
            element_mo_adam: Adam::new(&global_nn.element_mo_nn, adam_config),
            element_cd_adam: Adam::new(&global_nn.element_cd_nn, adam_config),
            element_sn_adam: Adam::new(&global_nn.element_sn_nn, adam_config),
            element_i_adam: Adam::new(&global_nn.element_i_nn, adam_config),

            amino_acid_gly_adam: Adam::new(&global_nn.amino_acid_gly_nn, adam_config),
            amino_acid_ala_adam: Adam::new(&global_nn.amino_acid_ala_nn, adam_config),
            amino_acid_val_adam: Adam::new(&global_nn.amino_acid_val_nn, adam_config),
            amino_acid_leu_adam: Adam::new(&global_nn.amino_acid_leu_nn, adam_config),
            amino_acid_ile_adam: Adam::new(&global_nn.amino_acid_ile_nn, adam_config),
            amino_acid_ser_adam: Adam::new(&global_nn.amino_acid_ser_nn, adam_config),
            amino_acid_thr_adam: Adam::new(&global_nn.amino_acid_thr_nn, adam_config),
            amino_acid_asp_adam: Adam::new(&global_nn.amino_acid_asp_nn, adam_config),
            amino_acid_ash_adam: Adam::new(&global_nn.amino_acid_ash_nn, adam_config),
            amino_acid_asn_adam: Adam::new(&global_nn.amino_acid_asn_nn, adam_config),
            amino_acid_glu_adam: Adam::new(&global_nn.amino_acid_glu_nn, adam_config),
            amino_acid_glh_adam: Adam::new(&global_nn.amino_acid_glh_nn, adam_config),
            amino_acid_gln_adam: Adam::new(&global_nn.amino_acid_gln_nn, adam_config),
            amino_acid_lys_adam: Adam::new(&global_nn.amino_acid_lys_nn, adam_config),
            amino_acid_lyn_adam: Adam::new(&global_nn.amino_acid_lyn_nn, adam_config),
            amino_acid_arg_adam: Adam::new(&global_nn.amino_acid_arg_nn, adam_config),
            amino_acid_arn_adam: Adam::new(&global_nn.amino_acid_arn_nn, adam_config),
            amino_acid_cys_adam: Adam::new(&global_nn.amino_acid_cys_nn, adam_config),
            amino_acid_cyx_adam: Adam::new(&global_nn.amino_acid_cyx_nn, adam_config),
            amino_acid_met_adam: Adam::new(&global_nn.amino_acid_met_nn, adam_config),
            amino_acid_hid_adam: Adam::new(&global_nn.amino_acid_hid_nn, adam_config),
            amino_acid_hie_adam: Adam::new(&global_nn.amino_acid_hie_nn, adam_config),
            amino_acid_hip_adam: Adam::new(&global_nn.amino_acid_hip_nn, adam_config),
            amino_acid_phe_adam: Adam::new(&global_nn.amino_acid_phe_nn, adam_config),
            amino_acid_tyr_adam: Adam::new(&global_nn.amino_acid_tyr_nn, adam_config),
            amino_acid_trp_adam: Adam::new(&global_nn.amino_acid_trp_nn, adam_config),
            amino_acid_pro_adam: Adam::new(&global_nn.amino_acid_pro_nn, adam_config),

            molecule_wat_adam: Adam::new(&global_nn.molecule_wat_nn, adam_config),
        }
    }





    /// Update all the parameters for the input global NN using the input global gradients
    ///
    /// # Parameters
    /// ```
    /// global_nn: the input global NN of the protein system
    /// global_grads: the input global gradients of the global NN
    /// ```
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn update(&mut self, global_nn: &mut GlobalNN, global_grads: &Gradients<f64, Device>)
    {
        self.element_h_adam.update(&mut global_nn.element_h_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element H"));
        self.element_b_adam.update(&mut global_nn.element_b_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element B"));
        self.element_c_adam.update(&mut global_nn.element_c_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element C"));
        self.element_n_adam.update(&mut global_nn.element_n_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element N"));
        self.element_o_adam.update(&mut global_nn.element_o_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element O"));
        self.element_f_adam.update(&mut global_nn.element_f_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element F"));
        self.element_na_adam.update(&mut global_nn.element_na_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Na"));
        self.element_mg_adam.update(&mut global_nn.element_mg_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Mg"));
        self.element_si_adam.update(&mut global_nn.element_si_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Si"));
        self.element_p_adam.update(&mut global_nn.element_p_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element P"));
        self.element_s_adam.update(&mut global_nn.element_s_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element S"));
        self.element_cl_adam.update(&mut global_nn.element_cl_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Cl"));
        self.element_k_adam.update(&mut global_nn.element_k_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element K"));
        self.element_ca_adam.update(&mut global_nn.element_ca_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Ca"));
        self.element_v_adam.update(&mut global_nn.element_v_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element V"));
        self.element_cr_adam.update(&mut global_nn.element_cr_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Cr"));
        self.element_mn_adam.update(&mut global_nn.element_mn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Mn"));
        self.element_fe_adam.update(&mut global_nn.element_fe_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Fe"));
        self.element_co_adam.update(&mut global_nn.element_co_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Co"));
        self.element_ni_adam.update(&mut global_nn.element_ni_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Ni"));
        self.element_cu_adam.update(&mut global_nn.element_cu_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Cu"));
        self.element_zn_adam.update(&mut global_nn.element_zn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Zn"));
        self.element_as_adam.update(&mut global_nn.element_as_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element As"));
        self.element_se_adam.update(&mut global_nn.element_se_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Se"));
        self.element_br_adam.update(&mut global_nn.element_br_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Br"));
        self.element_mo_adam.update(&mut global_nn.element_mo_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Mo"));
        self.element_cd_adam.update(&mut global_nn.element_cd_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Cd"));
        self.element_sn_adam.update(&mut global_nn.element_sn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element Sn"));
        self.element_i_adam.update(&mut global_nn.element_i_nn, &global_grads).expect(&error_nn_para_update("Adam", "Element I"));

        self.amino_acid_gly_adam.update(&mut global_nn.amino_acid_gly_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Gly"));
        self.amino_acid_ala_adam.update(&mut global_nn.amino_acid_ala_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Ala"));
        self.amino_acid_val_adam.update(&mut global_nn.amino_acid_val_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Val"));
        self.amino_acid_leu_adam.update(&mut global_nn.amino_acid_leu_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Leu"));
        self.amino_acid_ile_adam.update(&mut global_nn.amino_acid_ile_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Ile"));
        self.amino_acid_ser_adam.update(&mut global_nn.amino_acid_ser_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Ser"));
        self.amino_acid_thr_adam.update(&mut global_nn.amino_acid_thr_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Thr"));
        self.amino_acid_asp_adam.update(&mut global_nn.amino_acid_asp_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Asp"));
        self.amino_acid_ash_adam.update(&mut global_nn.amino_acid_ash_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Ash"));
        self.amino_acid_asn_adam.update(&mut global_nn.amino_acid_asn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Asn"));
        self.amino_acid_glu_adam.update(&mut global_nn.amino_acid_glu_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Glu"));
        self.amino_acid_glh_adam.update(&mut global_nn.amino_acid_glh_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Glh"));
        self.amino_acid_gln_adam.update(&mut global_nn.amino_acid_gln_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Gln"));
        self.amino_acid_lys_adam.update(&mut global_nn.amino_acid_lys_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Lys"));
        self.amino_acid_lyn_adam.update(&mut global_nn.amino_acid_lyn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Lyn"));
        self.amino_acid_arg_adam.update(&mut global_nn.amino_acid_arg_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Arg"));
        self.amino_acid_arn_adam.update(&mut global_nn.amino_acid_arn_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Arn"));
        self.amino_acid_cys_adam.update(&mut global_nn.amino_acid_cys_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Cys"));
        self.amino_acid_cyx_adam.update(&mut global_nn.amino_acid_cyx_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Cyx"));
        self.amino_acid_met_adam.update(&mut global_nn.amino_acid_met_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Met"));
        self.amino_acid_hid_adam.update(&mut global_nn.amino_acid_hid_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Hid"));
        self.amino_acid_hie_adam.update(&mut global_nn.amino_acid_hie_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Hie"));
        self.amino_acid_hip_adam.update(&mut global_nn.amino_acid_hip_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Hip"));
        self.amino_acid_phe_adam.update(&mut global_nn.amino_acid_phe_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Phe"));
        self.amino_acid_tyr_adam.update(&mut global_nn.amino_acid_tyr_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Tyr"));
        self.amino_acid_trp_adam.update(&mut global_nn.amino_acid_trp_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Trp"));
        self.amino_acid_pro_adam.update(&mut global_nn.amino_acid_pro_nn, &global_grads).expect(&error_nn_para_update("Adam", "Amino Acid Pro"));

        self.molecule_wat_adam.update(&mut global_nn.molecule_wat_nn, &global_grads).expect(&error_nn_para_update("Adam", "Molecule Wat"));
    }
}










