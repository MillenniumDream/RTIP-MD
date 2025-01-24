//! Training of the global neural network
use crate::common::constants::{Device, Element, AminoAcid, Molecule, FragmentType};
use crate::common::error::*;
use crate::io::input::Para;
use crate::nn::global_nn::{GlobalNN, GlobalAdam};
use crate::nn::training_data::{EPS, ProteinSystemDescriptor};
use std::fs;
use std::fs::File;
use std::io::Write;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use dfdx::shapes::{Const, Rank1};
use dfdx::tensor::{Tensor, ZerosTensor, Gradients, Trace, OwnedTape, NoneTape, AsArray};
use dfdx::tensor_ops::{RealizeTo, BroadcastTo, Backward, AdamConfig};
use dfdx::nn::{Module, ModuleMut};
use dfdx::losses::rmse_loss;





const MULTIPLE: f64 = 1.0 / EPS;





/// Train the global neural network for the protein system
///
/// # Parameters
/// ```
/// input_nn_sub_dir: specify the sub-directory (the path with respect to directory 'nn') where to load the global neural network
/// output_nn_sub_dir: specify the sub-directory (the path with respect to directory 'nn') where to save the global neural network after training
/// input_data_sub_dir: specify the sub-directory (the path with respect to directory 'data') where to load the training DFT data
/// ```
///
/// # Examples
/// ```
/// ```
pub fn train(para: &Para)
{
    // Load the global neural network, and allocate gradients for it. Load the global data
    let mut global_nn: GlobalNN = GlobalNN::load(&para.nn_train_para.input_nn_sub_dir);
    let mut global_grads: Gradients<f64, Device> = global_nn.alloc_grads();
    let global_data_set: Vec<ProteinSystemDescriptor> = ProteinSystemDescriptor::from_data(&para.nn_train_para.input_data_sub_dir);



    // Define the Adam optimizer
    let adam_config: AdamConfig = AdamConfig
    {
        lr: 0.001,
        betas: [0.9, 0.999],
        eps: 0.00000001,
        weight_decay: None,
    };
    let mut global_adam: GlobalAdam = GlobalAdam::new(&global_nn, adam_config);



    // If directory 'nn/para.nn_train_para.output_nn_sub_dir' already exist, do nothing; otherwise, create the directory
    let output_nn_dir: String = format!("nn/{}", para.nn_train_para.output_nn_sub_dir);
    let output_nn_dir_exist = fs::metadata(&output_nn_dir);
    match output_nn_dir_exist
    {
        Ok(_) => (),
        Err(_) => fs::create_dir(&output_nn_dir).expect(&error_dir("creating", &output_nn_dir)),
    }
    // Specify the nn training output file and output the header into it
    let nn_training_output_file: String = format!("{}/nn_training.out", &output_nn_dir);
    let mut nn_training_output = File::create(&nn_training_output_file).expect(&error_file("creating", &nn_training_output_file));
    nn_training_output.write_all(b"  step       training_pot_loss         training_f_loss     validation_pot_loss       validation_f_loss\n").expect(&error_file("writing", &nn_training_output_file));



    // Data partition
    let global_data_size: usize = global_data_set.len();
    // there should have at least two data, one for training set, the other for validation set
    if global_data_size < 2
    {
        panic!("{}", error_global_data_size());
    }
    let mut training_data_set: Vec<usize>;
    let mut validation_data_set: Vec<usize>;
    loop
    {
        training_data_set = Vec::with_capacity(global_data_size);
        validation_data_set = Vec::with_capacity(global_data_size/5);
        let random_sampling: Array1<f64> = Array1::random(global_data_size, Uniform::new(0.0, 1.0));
        // Assign each data to training set or validation set with probability of 9:1
        for i in 0..global_data_size
        {
            if random_sampling[i] < 0.9
            {
                training_data_set.push(i);
            }
            else
            {
                validation_data_set.push(i);
            }
        }
        // If both training set and validation set have at least a data, jump out of the loop
        if (training_data_set.len() > 0) && (validation_data_set.len() > 0)
        {
            break
        }
    }
    let training_data_size: usize = training_data_set.len();
    let validation_data_size: usize = validation_data_set.len();



    // Train the NN iteratively
    let mut training_loss_pot: f64;
    let mut training_loss_pot_diff: f64;
    let mut validation_loss_pot: f64;
    let mut validation_loss_pot_diff: f64;
    let mut loss_pot: f64;
    let mut loss_pot_diff: f64;
    let mut random_index: usize;
    for i in 1..(para.nn_train_para.max_step+1)
    {
        // Calculate the training loss and update the global gradients for the global NN
        training_loss_pot = 0.0;
        training_loss_pot_diff = 0.0;
        for _j in 0..para.nn_train_para.training_batch_size
        {
            random_index = Array1::random(1, Uniform::new(0, training_data_size))[0];
            (loss_pot, loss_pot_diff) = one_point_rmse_mut(&mut global_nn, &mut global_grads, &global_data_set[ training_data_set[random_index] ]);
            training_loss_pot += loss_pot;
            training_loss_pot_diff += loss_pot_diff;
        }
        training_loss_pot /= para.nn_train_para.training_batch_size as f64;
        training_loss_pot_diff /= para.nn_train_para.training_batch_size as f64;

        // Update the parameters for the global NN, and zero the global gradients
        global_adam.update(&mut global_nn, &global_grads);
        global_nn.zero_grads(&mut global_grads);

        // Calculate the validation loss
        validation_loss_pot = 0.0;
        validation_loss_pot_diff = 0.0;
        for _j in 0..para.nn_train_para.validation_batch_size
        {
            random_index = Array1::random(1, Uniform::new(0, validation_data_size))[0];
            (loss_pot, loss_pot_diff) = one_point_rmse(&global_nn, &global_data_set[ validation_data_set[random_index] ]);
            validation_loss_pot += loss_pot;
            validation_loss_pot_diff += loss_pot_diff;
        }
        validation_loss_pot /= para.nn_train_para.validation_batch_size as f64;
        validation_loss_pot_diff /= para.nn_train_para.validation_batch_size as f64;

        // Output the losses in the current step, and save the new NN in the print step
        nn_training_output.write_all(format!("{:6} {:23.8} {:23.8} {:23.8} {:23.8}\n", i, training_loss_pot, training_loss_pot_diff, validation_loss_pot, validation_loss_pot_diff).as_bytes()).expect(&error_file("writing", &nn_training_output_file));
        if (i % para.nn_train_para.print_step) == 0
        {
            global_nn.save(&format!("{}/{}", para.nn_train_para.output_nn_sub_dir, i));
        }
    }
}










/// Obtain the root mean square error between predicted values and target values (i.e. pot and pot_diff) for a specific structure, and update the gradients for the global NN
///
/// # Parameters
/// ```
/// global_nn: the input global neural network
/// global_grads: the input mutable gradients of the global neural network
/// one_point_training_data: the input training data for a specific structure
/// rmse_loss_pot: the output root mean square error of pot for the specific structure
/// rmse_loss_pot_diff: the output root mean square error of pot_diff for the specific structure
/// ```
///
/// # Examples
/// ```
/// ```
fn one_point_rmse_mut(global_nn: &mut GlobalNN, global_grads: &mut Gradients<f64, Device>, one_point_training_data: &ProteinSystemDescriptor) -> (f64, f64)
{
    // Define a Device (CPU or Cuda) to build tensors
    let dev: Device = Device::seed_from_u64(1314);

    // Accumulation variables initialization
    let mut predicted_pot: Tensor<Rank1<1>, f64, Device, OwnedTape<f64, Device>> = dev.zeros().trace(global_grads.clone());
    let mut predicted_pot_1: Tensor<Rank1<1>, f64, Device, OwnedTape<f64, Device>> = dev.zeros().trace(global_grads.clone());
    let mut predicted_pot_diff: Tensor<(usize, Const<1>), f64, Device, OwnedTape<f64, Device>> = dev.zeros_like(&(one_point_training_data.n_diff, Const)).trace(global_grads.clone());

    // For each fragment in the input protein system
    for i in 0..one_point_training_data.fragment_descriptor.len()
    {
        let (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i) = match &one_point_training_data.fragment_descriptor[i].fragment_type
        {
            // If the fragment is an atom, predict its pot and pot_diff by the global NN
            FragmentType::Atom(element) =>
            {
                match element
                {
                    Element::H =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_h_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_h_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_h_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::C =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_c_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_c_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_c_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::O =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_o_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_o_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_o_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::N =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_n_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_n_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_n_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::S =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_s_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_s_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_s_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::P =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_p_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_p_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_p_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Na =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_na_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_na_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_na_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Cl =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_cl_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_cl_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_cl_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::K =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_k_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_k_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_k_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Ca =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_ca_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_ca_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_ca_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Mg =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_mg_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_mg_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_mg_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::F =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_f_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_f_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_f_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Fe =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_fe_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_fe_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_fe_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Cu =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_cu_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_cu_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_cu_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Zn =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_zn_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_zn_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_zn_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Mn =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_mn_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_mn_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_mn_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Mo =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_mo_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_mo_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_mo_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Co =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_co_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_co_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_co_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Cr =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_cr_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_cr_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_cr_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::V =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_v_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_v_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_v_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Sn =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_sn_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_sn_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_sn_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Ni =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_ni_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_ni_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_ni_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Si =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_si_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_si_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_si_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Se =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_se_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_se_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_se_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::I =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_i_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_i_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_i_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Br =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_br_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_br_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_br_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::As =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_as_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_as_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_as_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::B =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_b_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_b_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_b_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Cd =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_cd_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_cd_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_cd_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },

                    _ => panic!("{}", error_non_bioelement_nn(element)),
                }
            },

            // If the fragment is an amino acid residue, predict its pot and pot_diff by the global NN
            FragmentType::Residue(amino_acid) =>
            {
                match amino_acid
                {
                    AminoAcid::GLY =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<15>>().expect(&error_intrafragment_descriptor(amino_acid, 15)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<15>>().expect(&error_intrafragment_descriptor(amino_acid, 15)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<15>)>().expect(&error_intrafragment_descriptor(amino_acid, 15)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_gly_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_gly_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_gly_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ALA =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<24>>().expect(&error_intrafragment_descriptor(amino_acid, 24)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<24>>().expect(&error_intrafragment_descriptor(amino_acid, 24)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<24>)>().expect(&error_intrafragment_descriptor(amino_acid, 24)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_ala_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_ala_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_ala_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::VAL =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<42>>().expect(&error_intrafragment_descriptor(amino_acid, 42)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<42>>().expect(&error_intrafragment_descriptor(amino_acid, 42)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<42>)>().expect(&error_intrafragment_descriptor(amino_acid, 42)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_val_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_val_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_val_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::LEU =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<51>>().expect(&error_intrafragment_descriptor(amino_acid, 51)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<51>>().expect(&error_intrafragment_descriptor(amino_acid, 51)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<51>)>().expect(&error_intrafragment_descriptor(amino_acid, 51)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_leu_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_leu_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_leu_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ILE =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<51>>().expect(&error_intrafragment_descriptor(amino_acid, 51)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<51>>().expect(&error_intrafragment_descriptor(amino_acid, 51)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<51>)>().expect(&error_intrafragment_descriptor(amino_acid, 51)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_ile_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_ile_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_ile_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::SER =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<27>>().expect(&error_intrafragment_descriptor(amino_acid, 27)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<27>>().expect(&error_intrafragment_descriptor(amino_acid, 27)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<27>)>().expect(&error_intrafragment_descriptor(amino_acid, 27)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_ser_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_ser_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_ser_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::THR =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<36>)>().expect(&error_intrafragment_descriptor(amino_acid, 36)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_thr_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_thr_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_thr_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ASP =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<30>>().expect(&error_intrafragment_descriptor(amino_acid, 30)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<30>>().expect(&error_intrafragment_descriptor(amino_acid, 30)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<30>)>().expect(&error_intrafragment_descriptor(amino_acid, 30)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_asp_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_asp_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_asp_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ASH =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<33>>().expect(&error_intrafragment_descriptor(amino_acid, 33)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<33>>().expect(&error_intrafragment_descriptor(amino_acid, 33)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<33>)>().expect(&error_intrafragment_descriptor(amino_acid, 33)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_ash_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_ash_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_ash_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ASN =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<36>)>().expect(&error_intrafragment_descriptor(amino_acid, 36)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_asn_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_asn_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_asn_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::GLU =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<39>>().expect(&error_intrafragment_descriptor(amino_acid, 39)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<39>>().expect(&error_intrafragment_descriptor(amino_acid, 39)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<39>)>().expect(&error_intrafragment_descriptor(amino_acid, 39)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_glu_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_glu_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_glu_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::GLH =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<42>>().expect(&error_intrafragment_descriptor(amino_acid, 42)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<42>>().expect(&error_intrafragment_descriptor(amino_acid, 42)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<42>)>().expect(&error_intrafragment_descriptor(amino_acid, 42)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_glh_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_glh_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_glh_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::GLN =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<45>)>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_gln_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_gln_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_gln_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::LYS =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<60>>().expect(&error_intrafragment_descriptor(amino_acid, 60)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<60>>().expect(&error_intrafragment_descriptor(amino_acid, 60)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<60>)>().expect(&error_intrafragment_descriptor(amino_acid, 60)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_lys_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_lys_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_lys_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::LYN =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<57>>().expect(&error_intrafragment_descriptor(amino_acid, 57)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<57>>().expect(&error_intrafragment_descriptor(amino_acid, 57)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<57>)>().expect(&error_intrafragment_descriptor(amino_acid, 57)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_lyn_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_lyn_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_lyn_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ARG =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<66>>().expect(&error_intrafragment_descriptor(amino_acid, 66)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<66>>().expect(&error_intrafragment_descriptor(amino_acid, 66)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<66>)>().expect(&error_intrafragment_descriptor(amino_acid, 66)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_arg_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_arg_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_arg_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ARN =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<63>>().expect(&error_intrafragment_descriptor(amino_acid, 63)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<63>>().expect(&error_intrafragment_descriptor(amino_acid, 63)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<63>)>().expect(&error_intrafragment_descriptor(amino_acid, 63)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_arn_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_arn_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_arn_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::CYS =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<27>>().expect(&error_intrafragment_descriptor(amino_acid, 27)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<27>>().expect(&error_intrafragment_descriptor(amino_acid, 27)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<27>)>().expect(&error_intrafragment_descriptor(amino_acid, 27)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_cys_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_cys_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_cys_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::CYX =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<24>>().expect(&error_intrafragment_descriptor(amino_acid, 24)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<24>>().expect(&error_intrafragment_descriptor(amino_acid, 24)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<24>)>().expect(&error_intrafragment_descriptor(amino_acid, 24)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_cyx_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_cyx_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_cyx_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::MET =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<45>)>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_met_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_met_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_met_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::HID =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<45>)>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_hid_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_hid_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_hid_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::HIE =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<45>)>().expect(&error_intrafragment_descriptor(amino_acid, 45)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_hie_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_hie_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_hie_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::HIP =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<48>>().expect(&error_intrafragment_descriptor(amino_acid, 48)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<48>>().expect(&error_intrafragment_descriptor(amino_acid, 48)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<48>)>().expect(&error_intrafragment_descriptor(amino_acid, 48)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_hip_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_hip_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_hip_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::PHE =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<54>>().expect(&error_intrafragment_descriptor(amino_acid, 54)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<54>>().expect(&error_intrafragment_descriptor(amino_acid, 54)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<54>)>().expect(&error_intrafragment_descriptor(amino_acid, 54)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_phe_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_phe_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_phe_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::TYR =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<57>>().expect(&error_intrafragment_descriptor(amino_acid, 57)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<57>>().expect(&error_intrafragment_descriptor(amino_acid, 57)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<57>)>().expect(&error_intrafragment_descriptor(amino_acid, 57)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_tyr_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_tyr_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_tyr_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::TRP =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<66>>().expect(&error_intrafragment_descriptor(amino_acid, 66)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<66>>().expect(&error_intrafragment_descriptor(amino_acid, 66)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<66>)>().expect(&error_intrafragment_descriptor(amino_acid, 66)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_trp_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_trp_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_trp_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::PRO =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36)).traced(global_grads.clone());
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36)).traced(global_grads.clone());
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<36>)>().expect(&error_intrafragment_descriptor(amino_acid, 36)).traced(global_grads.clone());
                        let predicted_pot_i = global_nn.amino_acid_pro_nn.forward_mut((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_pro_nn.forward_mut((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_pro_nn.forward_mut((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                }
            },

            // If the fragment is a head atom, predict its pot and pot_diff by the global NN
            FragmentType::Head(element) =>
            {
                match element
                {
                    Element::H =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_h_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_h_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_h_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },

                    _ => panic!("{}", error_type("Head", &format!("{:?}", element))),
                }
            },

            // If the fragment is a tail atom, predict its pot and pot_diff by the global NN
            FragmentType::Tail(element) =>
            {
                match element
                {
                    Element::O =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_o_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_o_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_o_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::H =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.element_h_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_h_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_h_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },

                    _ => panic!("{}", error_type("Tail", &format!("{:?}", element))),
                }
            },

            // If the fragment is a molecule, predict its pot and pot_diff by the global NN
            FragmentType::Molecule(molecule) =>
            {
                match molecule
                {
                    Molecule::WAT =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.trace(global_grads.clone());
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.trace(global_grads.clone());
                        let predicted_pot_i = global_nn.molecule_wat_nn.forward_mut(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.molecule_wat_nn.forward_mut(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.molecule_wat_nn.forward_mut(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                }
            },
        };

        // Accumulate the potentials for all the fragments in the specific protein system
        predicted_pot = predicted_pot + predicted_pot_i;
        predicted_pot_1 = predicted_pot_1 + predicted_pot_1_i;
        predicted_pot_diff = predicted_pot_diff + predicted_pot_diff_i;
    }

    // Calculate the RMSE loss and update the gradients for the global NN
    predicted_pot_diff = predicted_pot_diff - predicted_pot_1.broadcast_like(&(one_point_training_data.n_diff, Const));
    let loss_pot = rmse_loss(predicted_pot, one_point_training_data.pot.clone());
    let loss_pot_diff = rmse_loss(predicted_pot_diff, one_point_training_data.pot_diff.clone()) * MULTIPLE;
    let rmse_loss_pot: f64 = loss_pot.array();
    let rmse_loss_pot_diff: f64 = loss_pot_diff.array();
    *global_grads = (loss_pot + loss_pot_diff).backward();

    (rmse_loss_pot, rmse_loss_pot_diff)
}










/// Obtain the root mean square error between predicted values and target values (i.e. pot and pot_diff) for a specific structure
///
/// # Parameters
/// ```
/// global_nn: the input global neural network
/// one_point_training_data: the input training data for a specific structure
/// rmse_loss_pot: the output root mean square error of pot for the specific structure
/// rmse_loss_pot_diff: the output root mean square error of pot_diff for the specific structure
/// ```
///
/// # Examples
/// ```
/// ```
fn one_point_rmse(global_nn: &GlobalNN, one_point_training_data: &ProteinSystemDescriptor) -> (f64, f64)
{
    // Define a Device (CPU or Cuda) to build tensors
    let dev: Device = Device::seed_from_u64(1314);

    // Accumulation variables initialization
    let mut predicted_pot: Tensor<Rank1<1>, f64, Device, NoneTape> = dev.zeros();
    let mut predicted_pot_1: Tensor<Rank1<1>, f64, Device, NoneTape> = dev.zeros();
    let mut predicted_pot_diff: Tensor<(usize, Const<1>), f64, Device, NoneTape> = dev.zeros_like(&(one_point_training_data.n_diff, Const));

    // For each fragment in the input protein system
    for i in 0..one_point_training_data.fragment_descriptor.len()
    {
        let (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i) = match &one_point_training_data.fragment_descriptor[i].fragment_type
        {
            // If the fragment is an atom, predict its pot and pot_diff by the global NN
            FragmentType::Atom(element) =>
            {
                match element
                {
                    Element::H =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_h_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_h_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_h_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::C =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_c_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_c_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_c_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::O =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_o_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_o_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_o_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::N =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_n_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_n_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_n_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::S =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_s_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_s_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_s_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::P =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_p_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_p_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_p_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Na =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_na_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_na_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_na_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Cl =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_cl_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_cl_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_cl_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::K =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_k_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_k_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_k_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Ca =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_ca_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_ca_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_ca_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Mg =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_mg_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_mg_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_mg_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::F =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_f_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_f_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_f_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Fe =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_fe_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_fe_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_fe_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Cu =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_cu_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_cu_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_cu_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Zn =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_zn_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_zn_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_zn_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Mn =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_mn_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_mn_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_mn_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Mo =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_mo_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_mo_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_mo_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Co =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_co_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_co_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_co_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Cr =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_cr_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_cr_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_cr_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::V =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_v_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_v_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_v_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Sn =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_sn_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_sn_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_sn_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Ni =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_ni_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_ni_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_ni_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Si =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_si_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_si_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_si_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Se =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_se_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_se_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_se_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::I =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_i_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_i_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_i_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Br =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_br_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_br_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_br_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::As =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_as_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_as_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_as_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::B =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_b_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_b_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_b_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::Cd =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_cd_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_cd_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_cd_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },

                    _ => panic!("{}", error_non_bioelement_nn(element)),
                }
            },

            // If the fragment is an amino acid residue, predict its pot and pot_diff by the global NN
            FragmentType::Residue(amino_acid) =>
            {
                match amino_acid
                {
                    AminoAcid::GLY =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<15>>().expect(&error_intrafragment_descriptor(amino_acid, 15));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<15>>().expect(&error_intrafragment_descriptor(amino_acid, 15));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<15>)>().expect(&error_intrafragment_descriptor(amino_acid, 15));
                        let predicted_pot_i = global_nn.amino_acid_gly_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_gly_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_gly_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ALA =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<24>>().expect(&error_intrafragment_descriptor(amino_acid, 24));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<24>>().expect(&error_intrafragment_descriptor(amino_acid, 24));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<24>)>().expect(&error_intrafragment_descriptor(amino_acid, 24));
                        let predicted_pot_i = global_nn.amino_acid_ala_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_ala_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_ala_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::VAL =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<42>>().expect(&error_intrafragment_descriptor(amino_acid, 42));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<42>>().expect(&error_intrafragment_descriptor(amino_acid, 42));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<42>)>().expect(&error_intrafragment_descriptor(amino_acid, 42));
                        let predicted_pot_i = global_nn.amino_acid_val_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_val_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_val_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::LEU =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<51>>().expect(&error_intrafragment_descriptor(amino_acid, 51));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<51>>().expect(&error_intrafragment_descriptor(amino_acid, 51));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<51>)>().expect(&error_intrafragment_descriptor(amino_acid, 51));
                        let predicted_pot_i = global_nn.amino_acid_leu_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_leu_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_leu_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ILE =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<51>>().expect(&error_intrafragment_descriptor(amino_acid, 51));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<51>>().expect(&error_intrafragment_descriptor(amino_acid, 51));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<51>)>().expect(&error_intrafragment_descriptor(amino_acid, 51));
                        let predicted_pot_i = global_nn.amino_acid_ile_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_ile_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_ile_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::SER =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<27>>().expect(&error_intrafragment_descriptor(amino_acid, 27));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<27>>().expect(&error_intrafragment_descriptor(amino_acid, 27));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<27>)>().expect(&error_intrafragment_descriptor(amino_acid, 27));
                        let predicted_pot_i = global_nn.amino_acid_ser_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_ser_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_ser_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::THR =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<36>)>().expect(&error_intrafragment_descriptor(amino_acid, 36));
                        let predicted_pot_i = global_nn.amino_acid_thr_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_thr_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_thr_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ASP =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<30>>().expect(&error_intrafragment_descriptor(amino_acid, 30));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<30>>().expect(&error_intrafragment_descriptor(amino_acid, 30));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<30>)>().expect(&error_intrafragment_descriptor(amino_acid, 30));
                        let predicted_pot_i = global_nn.amino_acid_asp_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_asp_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_asp_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ASH =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<33>>().expect(&error_intrafragment_descriptor(amino_acid, 33));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<33>>().expect(&error_intrafragment_descriptor(amino_acid, 33));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<33>)>().expect(&error_intrafragment_descriptor(amino_acid, 33));
                        let predicted_pot_i = global_nn.amino_acid_ash_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_ash_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_ash_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ASN =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<36>)>().expect(&error_intrafragment_descriptor(amino_acid, 36));
                        let predicted_pot_i = global_nn.amino_acid_asn_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_asn_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_asn_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::GLU =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<39>>().expect(&error_intrafragment_descriptor(amino_acid, 39));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<39>>().expect(&error_intrafragment_descriptor(amino_acid, 39));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<39>)>().expect(&error_intrafragment_descriptor(amino_acid, 39));
                        let predicted_pot_i = global_nn.amino_acid_glu_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_glu_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_glu_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::GLH =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<42>>().expect(&error_intrafragment_descriptor(amino_acid, 42));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<42>>().expect(&error_intrafragment_descriptor(amino_acid, 42));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<42>)>().expect(&error_intrafragment_descriptor(amino_acid, 42));
                        let predicted_pot_i = global_nn.amino_acid_glh_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_glh_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_glh_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::GLN =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<45>)>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let predicted_pot_i = global_nn.amino_acid_gln_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_gln_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_gln_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::LYS =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<60>>().expect(&error_intrafragment_descriptor(amino_acid, 60));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<60>>().expect(&error_intrafragment_descriptor(amino_acid, 60));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<60>)>().expect(&error_intrafragment_descriptor(amino_acid, 60));
                        let predicted_pot_i = global_nn.amino_acid_lys_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_lys_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_lys_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::LYN =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<57>>().expect(&error_intrafragment_descriptor(amino_acid, 57));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<57>>().expect(&error_intrafragment_descriptor(amino_acid, 57));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<57>)>().expect(&error_intrafragment_descriptor(amino_acid, 57));
                        let predicted_pot_i = global_nn.amino_acid_lyn_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_lyn_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_lyn_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ARG =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<66>>().expect(&error_intrafragment_descriptor(amino_acid, 66));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<66>>().expect(&error_intrafragment_descriptor(amino_acid, 66));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<66>)>().expect(&error_intrafragment_descriptor(amino_acid, 66));
                        let predicted_pot_i = global_nn.amino_acid_arg_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_arg_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_arg_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::ARN =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<63>>().expect(&error_intrafragment_descriptor(amino_acid, 63));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<63>>().expect(&error_intrafragment_descriptor(amino_acid, 63));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<63>)>().expect(&error_intrafragment_descriptor(amino_acid, 63));
                        let predicted_pot_i = global_nn.amino_acid_arn_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_arn_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_arn_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::CYS =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<27>>().expect(&error_intrafragment_descriptor(amino_acid, 27));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<27>>().expect(&error_intrafragment_descriptor(amino_acid, 27));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<27>)>().expect(&error_intrafragment_descriptor(amino_acid, 27));
                        let predicted_pot_i = global_nn.amino_acid_cys_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_cys_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_cys_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::CYX =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<24>>().expect(&error_intrafragment_descriptor(amino_acid, 24));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<24>>().expect(&error_intrafragment_descriptor(amino_acid, 24));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<24>)>().expect(&error_intrafragment_descriptor(amino_acid, 24));
                        let predicted_pot_i = global_nn.amino_acid_cyx_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_cyx_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_cyx_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::MET =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<45>)>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let predicted_pot_i = global_nn.amino_acid_met_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_met_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_met_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::HID =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<45>)>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let predicted_pot_i = global_nn.amino_acid_hid_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_hid_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_hid_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::HIE =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<45>>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<45>)>().expect(&error_intrafragment_descriptor(amino_acid, 45));
                        let predicted_pot_i = global_nn.amino_acid_hie_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_hie_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_hie_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::HIP =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<48>>().expect(&error_intrafragment_descriptor(amino_acid, 48));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<48>>().expect(&error_intrafragment_descriptor(amino_acid, 48));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<48>)>().expect(&error_intrafragment_descriptor(amino_acid, 48));
                        let predicted_pot_i = global_nn.amino_acid_hip_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_hip_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_hip_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::PHE =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<54>>().expect(&error_intrafragment_descriptor(amino_acid, 54));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<54>>().expect(&error_intrafragment_descriptor(amino_acid, 54));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<54>)>().expect(&error_intrafragment_descriptor(amino_acid, 54));
                        let predicted_pot_i = global_nn.amino_acid_phe_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_phe_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_phe_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::TYR =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<57>>().expect(&error_intrafragment_descriptor(amino_acid, 57));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<57>>().expect(&error_intrafragment_descriptor(amino_acid, 57));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<57>)>().expect(&error_intrafragment_descriptor(amino_acid, 57));
                        let predicted_pot_i = global_nn.amino_acid_tyr_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_tyr_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_tyr_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::TRP =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<66>>().expect(&error_intrafragment_descriptor(amino_acid, 66));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<66>>().expect(&error_intrafragment_descriptor(amino_acid, 66));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<66>)>().expect(&error_intrafragment_descriptor(amino_acid, 66));
                        let predicted_pot_i = global_nn.amino_acid_trp_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_trp_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_trp_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    AminoAcid::PRO =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let intrafragment_descriptor = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36));
                        let intrafragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor.clone().expect(&error_none_value("intrafragment_descriptor")).try_realize::<Rank1<36>>().expect(&error_intrafragment_descriptor(amino_acid, 36));
                        let intrafragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].intrafragment_descriptor_diff.clone().expect(&error_none_value("intrafragment_descriptor_diff")).try_realize::<(usize, Const<36>)>().expect(&error_intrafragment_descriptor(amino_acid, 36));
                        let predicted_pot_i = global_nn.amino_acid_pro_nn.forward((interfragment_descriptor, intrafragment_descriptor));
                        let predicted_pot_1_i = global_nn.amino_acid_pro_nn.forward((interfragment_descriptor_1, intrafragment_descriptor_1));
                        let predicted_pot_diff_i = global_nn.amino_acid_pro_nn.forward((interfragment_descriptor_diff, intrafragment_descriptor_diff));
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                }
            },

            // If the fragment is a head atom, predict its pot and pot_diff by the global NN
            FragmentType::Head(element) =>
            {
                match element
                {
                    Element::H =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_h_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_h_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_h_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },

                    _ => panic!("{}", error_type("Head", &format!("{:?}", element))),
                }
            },

            // If the fragment is a tail atom, predict its pot and pot_diff by the global NN
            FragmentType::Tail(element) =>
            {
                match element
                {
                    Element::O =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_o_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_o_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_o_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                    Element::H =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.element_h_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.element_h_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.element_h_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },

                    _ => panic!("{}", error_type("Tail", &format!("{:?}", element))),
                }
            },

            // If the fragment is a molecule, predict its pot and pot_diff by the global NN
            FragmentType::Molecule(molecule) =>
            {
                match molecule
                {
                    Molecule::WAT =>
                    {
                        let interfragment_descriptor = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_1 = one_point_training_data.fragment_descriptor[i].interfragment_descriptor.clone();
                        let interfragment_descriptor_diff = one_point_training_data.fragment_descriptor[i].interfragment_descriptor_diff.clone();
                        let predicted_pot_i = global_nn.molecule_wat_nn.forward(interfragment_descriptor);
                        let predicted_pot_1_i = global_nn.molecule_wat_nn.forward(interfragment_descriptor_1);
                        let predicted_pot_diff_i = global_nn.molecule_wat_nn.forward(interfragment_descriptor_diff);
                        (predicted_pot_i, predicted_pot_1_i, predicted_pot_diff_i)
                    },
                }
            },
        };

        // Accumulate the potentials for all the fragments in the specific protein system
        predicted_pot = predicted_pot + predicted_pot_i;
        predicted_pot_1 = predicted_pot_1 + predicted_pot_1_i;
        predicted_pot_diff = predicted_pot_diff + predicted_pot_diff_i;
    }

    // Calculate the RMSE loss and update the gradients for the global NN
    predicted_pot_diff = predicted_pot_diff - predicted_pot_1.broadcast_like(&(one_point_training_data.n_diff, Const));
    let loss_pot = rmse_loss(predicted_pot, one_point_training_data.pot.clone());
    let loss_pot_diff = rmse_loss(predicted_pot_diff, one_point_training_data.pot_diff.clone()) * MULTIPLE;
    let rmse_loss_pot: f64 = loss_pot.array();
    let rmse_loss_pot_diff: f64 = loss_pot_diff.array();

    (rmse_loss_pot, rmse_loss_pot_diff)
}










