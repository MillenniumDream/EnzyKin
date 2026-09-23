//! Training of the global neural network
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use crate::common::constants::Device;
use crate::common::error::*;
use crate::io::input::Para;
use crate::nn::descriptor::N_DES;
use crate::nn::global_nn::{GlobalNN, GlobalAdam, NN, NNMut};
use crate::nn::training_data::{EPS, ProteinSystemDescriptor};
use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use dfdx::shapes::{Const, Axis};
use dfdx::tensor::{Tensor, ZerosTensor, TensorFromVec, Gradients, Trace, OwnedTape, NoneTape, AsArray};
use dfdx::tensor_ops::{BroadcastTo, Backward, AdamConfig};
use dfdx::nn::{Module, ModuleMut};
use dfdx::losses::rmse_loss;










/// Train the global neural network on a single data set
///
/// # Parameters
/// ```
/// para: the input global parameters
/// ```
///
/// # Examples
/// ```
/// ```
pub fn training_on_single_data(para: &Para)
{
    // Define a Device (CPU or Cuda) to build tensors
    let dev: Device = Device::seed_from_u64(1314);
    // Load the global neural network, and allocate gradients for it
    let mut global_nn: GlobalNN = GlobalNN::load(&para.nn_train_para.input_nn_dir);
    let mut global_grads: Gradients<f64, Device> = global_nn.alloc_grads(&para.nn_train_para.nn_update);
    // Load the global data
    let global_data_set: Vec<ProteinSystemDescriptor> = ProteinSystemDescriptor::from_single_data(&para.nn_train_para.input_data_dir[0], &para.nn_train_para.norm_dir, para.nn_train_para.long_ranged_dir.clone());



    // Define the Adam optimizer
    let adam_config: AdamConfig = AdamConfig
    {
        lr: para.nn_train_para.lr,
        betas: [0.9, 0.999],
        eps: 0.00000001,
        weight_decay: None,
    };
    let mut global_adam: GlobalAdam = GlobalAdam::new(&global_nn, adam_config);



    // If directory 'para.nn_train_para.output_nn_dir' already exist, do nothing; otherwise, create the directory
    let output_nn_dir_exist = fs::metadata(&para.nn_train_para.output_nn_dir);
    match output_nn_dir_exist
    {
        Ok(_) => (),
        Err(_) => fs::create_dir_all(&para.nn_train_para.output_nn_dir).expect(&error_dir("creating", &para.nn_train_para.output_nn_dir)),
    }
    // Specify the nn training output file and output the header into it
    let nn_training_output_file: PathBuf = para.nn_train_para.output_nn_dir.join("nn_training.out");
    let mut nn_training_output = File::create(&nn_training_output_file).expect(&error_file("creating", &nn_training_output_file));
    nn_training_output.write_all(b"    step       training_pot_loss         training_f_loss     validation_pot_loss       validation_f_loss\n").expect(&error_file("writing", &nn_training_output_file));



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



    // Train the NN iteratively
    let mut training_loss_pot: f64;
    let mut training_loss_pot_diff: f64;
    let mut validation_loss_pot: f64;
    let mut validation_loss_pot_diff: f64;
    for i in 1..(para.nn_train_para.max_step+1)
    {
        // Calculate the training loss and update the global gradients for the global NN
        (training_loss_pot, training_loss_pot_diff) = random_batch_rmse_mut(&dev, &mut global_nn, &mut global_grads, &global_data_set, &training_data_set, para.nn_train_para.training_batch_size);

        // Update the parameters for the global NN, and zero the global gradients
        global_adam.update(&mut global_nn, &global_grads, &para.nn_train_para.nn_update);
        global_nn.zero_grads(&mut global_grads, &para.nn_train_para.nn_update);

        // Calculate the validation loss
        (validation_loss_pot, validation_loss_pot_diff) = random_batch_rmse(&dev, &global_nn, &global_data_set, &validation_data_set, para.nn_train_para.validation_batch_size);

        // Output the losses in the current step, and save the new NN in the print step
        nn_training_output.write_all(format!("{:8} {:23.8} {:23.8} {:23.8} {:23.8}\n", i, training_loss_pot, training_loss_pot_diff, validation_loss_pot, validation_loss_pot_diff).as_bytes()).expect(&error_file("writing", &nn_training_output_file));
        if (i % para.nn_train_para.print_step) == 0
        {
            global_nn.save(&para.nn_train_para.output_nn_dir.join(format!("{i}")));
        }
    }
}










/// Train the global neural network on multiple data sets
///
/// # Parameters
/// ```
/// para: the input global parameters
/// ```
///
/// # Examples
/// ```
/// ```
pub fn training_on_multiple_data(para: &Para)
{
    // Define a Device (CPU or Cuda) to build tensors
    let dev: Device = Device::seed_from_u64(1314);
    // Load the global neural network, and allocate gradients for it
    let mut global_nn: GlobalNN = GlobalNN::load(&para.nn_train_para.input_nn_dir);
    let mut global_grads: Gradients<f64, Device> = global_nn.alloc_grads(&para.nn_train_para.nn_update);



    // Define the Adam optimizer
    let adam_config: AdamConfig = AdamConfig
    {
        lr: para.nn_train_para.lr,
        betas: [0.9, 0.999],
        eps: 0.00000001,
        weight_decay: None,
    };
    let mut global_adam: GlobalAdam = GlobalAdam::new(&global_nn, adam_config);



    // If directory 'para.nn_train_para.output_nn_dir' already exist, do nothing; otherwise, create the directory
    let output_nn_dir_exist = fs::metadata(&para.nn_train_para.output_nn_dir);
    match output_nn_dir_exist
    {
        Ok(_) => (),
        Err(_) => fs::create_dir_all(&para.nn_train_para.output_nn_dir).expect(&error_dir("creating", &para.nn_train_para.output_nn_dir)),
    }
    // Specify the nn training output file and output the header into it
    let nn_training_output_file: PathBuf = para.nn_train_para.output_nn_dir.join("nn_training.out");
    let mut nn_training_output = File::create(&nn_training_output_file).expect(&error_file("creating", &nn_training_output_file));
    nn_training_output.write_all(b"    step       training_pot_loss         training_f_loss\n").expect(&error_file("writing", &nn_training_output_file));



    // Train the NN iteratively
    let mut training_loss_pot: f64;
    let mut training_loss_pot_diff: f64;
    for i in 1..(para.nn_train_para.max_step+1)
    {
        // Randomly select a data set and calculate the descriptors for a batch size of structures
        let batch_data_set: Vec<ProteinSystemDescriptor> = ProteinSystemDescriptor::from_random_data(&para.nn_train_para.input_data_dir, &para.nn_train_para.norm_dir, para.nn_train_para.long_ranged_dir.clone(), para.nn_train_para.training_batch_size);

        // Calculate the training loss and update the global gradients for the global NN
        (training_loss_pot, training_loss_pot_diff) = single_batch_rmse_mut(&dev, &mut global_nn, &mut global_grads, batch_data_set);

        // Update the parameters for the global NN, and zero the global gradients
        global_adam.update(&mut global_nn, &global_grads, &para.nn_train_para.nn_update);
        global_nn.zero_grads(&mut global_grads, &para.nn_train_para.nn_update);

        // Output the losses in the current step, and save the new NN in the print step
        nn_training_output.write_all(format!("{:8} {:23.8} {:23.8}\n", i, training_loss_pot, training_loss_pot_diff).as_bytes()).expect(&error_file("writing", &nn_training_output_file));
        if (i % para.nn_train_para.print_step) == 0
        {
            global_nn.save(&para.nn_train_para.output_nn_dir.join(format!("{i}")));
        }
    }
}










/// Randomly select a series of structures (training_batch_size) from the training set, calculate their predicted values according to their descriptors,
/// return the root mean square error between predicted values and target values (i.e. pot and pot_diff), and update the gradients for the global NN
///
/// # Parameters
/// ```
/// dev: the device for tensor construction
/// global_nn: the input global neural network
/// global_grads: the input mutable gradients of the global neural network
/// global_data_set: the input global data set
/// training_data_set: the indices of the data for training
/// training_batch_size: the batch size for training
/// training_loss_pot: the output root mean square error of pot for the randomly selected structures from training set
/// training_loss_pot_diff: the output root mean square error of pot_diff for the randomly selected structures from training set
/// ```
///
/// # Examples
/// ```
/// ```
fn random_batch_rmse_mut(dev: &Device, global_nn: &mut GlobalNN, global_grads: &mut Gradients<f64, Device>, global_data_set: &Vec<ProteinSystemDescriptor>, training_data_set: &Vec<usize>, training_batch_size: usize) -> (f64, f64)
{
    // Randomly select a series of structures (training_batch_size) from the training set
    let random_index: Array1<usize> = Array1::random(training_batch_size, Uniform::new(0, training_data_set.len()));
    // Obtain the number of finite difference perturbation structures for the input protein system 
    let n_diff: usize = global_data_set[0].n_diff;



    // Accumulation variables initialization
    let mut predicted_pot: Tensor<(usize, Const<1>), f64, Device, OwnedTape<f64, Device>> = dev.zeros_like(&(training_batch_size, Const)).traced(global_grads.clone());
    let mut predicted_pot1: Tensor<(usize, Const<1>), f64, Device, OwnedTape<f64, Device>> = dev.zeros_like(&(training_batch_size, Const)).traced(global_grads.clone());
    let mut predicted_pot_diff: Tensor<(usize, usize, Const<1>), f64, Device, OwnedTape<f64, Device>> = dev.zeros_like(&(training_batch_size, n_diff, Const)).traced(global_grads.clone());



    // For each fragment in the input protein system
    for i in 0..global_data_set[0].fragment_descriptor.len()
    {
        // Assemble the descriptors
        let mut batch_descriptor_vec: Vec<f64> = Vec::with_capacity(training_batch_size*N_DES);
        let mut batch_descriptor_diff_vec: Vec<f64> = Vec::with_capacity(training_batch_size*n_diff*N_DES);
        // For each randomly selected structure from the training set
        for j in 0..training_batch_size
        {
            batch_descriptor_vec.append(&mut global_data_set[ training_data_set[ random_index[j] ] ].fragment_descriptor[i].descriptor.clone());
            batch_descriptor_diff_vec.append(&mut global_data_set[ training_data_set[ random_index[j] ] ].fragment_descriptor[i].descriptor_diff.clone());
        }

        // Convert the Vecs to Tensors
        let batch_descriptor: Tensor<(usize, Const<N_DES>), f64, Device, OwnedTape<f64, Device>> = dev.tensor_from_vec(batch_descriptor_vec.clone(), (training_batch_size, Const)).traced(global_grads.clone());
        let batch_descriptor1: Tensor<(usize, Const<N_DES>), f64, Device, OwnedTape<f64, Device>> = dev.tensor_from_vec(batch_descriptor_vec, (training_batch_size, Const)).traced(global_grads.clone());
        let batch_descriptor_diff: Tensor<(usize, usize, Const<N_DES>), f64, Device, OwnedTape<f64, Device>> = dev.tensor_from_vec(batch_descriptor_diff_vec, (training_batch_size, n_diff, Const)).traced(global_grads.clone());

        // Calculate the predicted pot and pot_diff for Fragment i
        let (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i) = match global_nn.get_fragment_nn_mut(&global_data_set[0].fragment_descriptor[i].fragment_type)
        {
            NNMut::Small(nn_small) =>
            {
                let predicted_pot_i = nn_small.forward_mut(batch_descriptor);
                let predicted_pot1_i = nn_small.forward_mut(batch_descriptor1);
                let predicted_pot_diff_i = nn_small.forward_mut(batch_descriptor_diff);
                (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i)
            },

            NNMut::Middle(nn_middle) =>
            {
                let predicted_pot_i = nn_middle.forward_mut(batch_descriptor);
                let predicted_pot1_i = nn_middle.forward_mut(batch_descriptor1);
                let predicted_pot_diff_i = nn_middle.forward_mut(batch_descriptor_diff);
                (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i)
            },

            NNMut::Large(nn_large) =>
            {
                let predicted_pot_i = nn_large.forward_mut(batch_descriptor);
                let predicted_pot1_i = nn_large.forward_mut(batch_descriptor1);
                let predicted_pot_diff_i = nn_large.forward_mut(batch_descriptor_diff);
                (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i)
            },
        };

        // Accumulate the predicted pot and pot_diff of Fragment i
        predicted_pot = predicted_pot + predicted_pot_i;
        predicted_pot1 = predicted_pot1 + predicted_pot1_i;
        predicted_pot_diff = predicted_pot_diff + predicted_pot_diff_i;
    }



    // Assemble the potentials
    let mut pot: Vec<f64> = Vec::with_capacity(training_batch_size);
    let mut pot_diff: Vec<f64> = Vec::with_capacity(training_batch_size*n_diff);
    // For each randomly selected structure from the training set
    for j in 0..training_batch_size
    {
        pot.push(global_data_set[ training_data_set[ random_index[j] ] ].pot);
        pot_diff.append(&mut global_data_set[ training_data_set[ random_index[j] ] ].pot_diff.clone());
    }
    // Convert the Vecs to Tensors
    let pot: Tensor<(usize, Const<1>), f64, Device, NoneTape> = dev.tensor_from_vec(pot, (training_batch_size, Const));
    let pot_diff: Tensor<(usize, usize, Const<1>), f64, Device, NoneTape> = dev.tensor_from_vec(pot_diff, (training_batch_size, n_diff, Const));



    // Calculate the RMSE loss and update the gradients for the global NN
    predicted_pot_diff = predicted_pot_diff - predicted_pot1.broadcast_like::<_, Axis<1>>(&(training_batch_size, n_diff, Const));
    let loss_pot = rmse_loss(predicted_pot, pot);
    let loss_pot_diff = rmse_loss(predicted_pot_diff, pot_diff) * (n_diff as f64 / EPS);
    let rmse_loss_pot: f64 = loss_pot.array();
    let rmse_loss_pot_diff: f64 = loss_pot_diff.array() / (n_diff as f64);
    *global_grads = (loss_pot + loss_pot_diff).backward();



    (rmse_loss_pot, rmse_loss_pot_diff)
}










/// Randomly select a series of structures (validation_batch_size) from the validation set, calculate their predicted values according to their descriptors,
/// return the root mean square error between predicted values and target values (i.e. pot and pot_diff)
///
/// # Parameters
/// ```
/// global_nn: the input global neural network
/// global_data_set: the input global data set
/// validation_data_set: the indices of the data for validation
/// validation_batch_size: the batch size for validation
/// validation_loss_pot: the output root mean square error of pot for the randomly selected structures from validation set
/// validation_loss_pot_diff: the output root mean square error of pot_diff for the randomly selected structures from validation set
/// ```
///
/// # Examples
/// ```
/// ```
fn random_batch_rmse(dev: &Device, global_nn: &GlobalNN, global_data_set: &Vec<ProteinSystemDescriptor>, validation_data_set: &Vec<usize>, validation_batch_size: usize) -> (f64, f64)
{
    // Randomly select a series of structures (validation_batch_size) from the validation set
    let random_index: Array1<usize> = Array1::random(validation_batch_size, Uniform::new(0, validation_data_set.len()));
    // Obtain the number of finite difference perturbation structures for the input protein system
    let n_diff: usize = global_data_set[0].n_diff;



    // Accumulation variables initialization
    let mut predicted_pot: Tensor<(usize, Const<1>), f64, Device, NoneTape> = dev.zeros_like(&(validation_batch_size, Const));
    let mut predicted_pot1: Tensor<(usize, Const<1>), f64, Device, NoneTape> = dev.zeros_like(&(validation_batch_size, Const));
    let mut predicted_pot_diff: Tensor<(usize, usize, Const<1>), f64, Device, NoneTape> = dev.zeros_like(&(validation_batch_size, n_diff, Const));



    // For each fragment in the input protein system
    for i in 0..global_data_set[0].fragment_descriptor.len()
    {
        // Assemble the descriptors
        let mut batch_descriptor_vec: Vec<f64> = Vec::with_capacity(validation_batch_size*N_DES);
        let mut batch_descriptor_diff_vec: Vec<f64> = Vec::with_capacity(validation_batch_size*n_diff*N_DES);
        // For each randomly selected structure from the validation set
        for j in 0..validation_batch_size
        {
            batch_descriptor_vec.append(&mut global_data_set[ validation_data_set[ random_index[j] ] ].fragment_descriptor[i].descriptor.clone());
            batch_descriptor_diff_vec.append(&mut global_data_set[ validation_data_set[ random_index[j] ] ].fragment_descriptor[i].descriptor_diff.clone());
        }

        // Convert the Vecs to Tensors
        let batch_descriptor: Tensor<(usize, Const<N_DES>), f64, Device, NoneTape> = dev.tensor_from_vec(batch_descriptor_vec.clone(), (validation_batch_size, Const));
        let batch_descriptor1: Tensor<(usize, Const<N_DES>), f64, Device, NoneTape> = dev.tensor_from_vec(batch_descriptor_vec, (validation_batch_size, Const));
        let batch_descriptor_diff: Tensor<(usize, usize, Const<N_DES>), f64, Device, NoneTape> = dev.tensor_from_vec(batch_descriptor_diff_vec, (validation_batch_size, n_diff, Const));

        // Calculate the predicted pot and pot_diff for Fragment i
        let (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i) = match global_nn.get_fragment_nn(&global_data_set[0].fragment_descriptor[i].fragment_type)
        {
            NN::Small(nn_small) =>
            {
                let predicted_pot_i = nn_small.forward(batch_descriptor);
                let predicted_pot1_i = nn_small.forward(batch_descriptor1);
                let predicted_pot_diff_i = nn_small.forward(batch_descriptor_diff);
                (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i)
            },

            NN::Middle(nn_middle) =>
            {
                let predicted_pot_i = nn_middle.forward(batch_descriptor);
                let predicted_pot1_i = nn_middle.forward(batch_descriptor1);
                let predicted_pot_diff_i = nn_middle.forward(batch_descriptor_diff);
                (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i)
            },

            NN::Large(nn_large) =>
            {
                let predicted_pot_i = nn_large.forward(batch_descriptor);
                let predicted_pot1_i = nn_large.forward(batch_descriptor1);
                let predicted_pot_diff_i = nn_large.forward(batch_descriptor_diff);
                (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i)
            },
        };

        // Accumulate the predicted pot and pot_diff of Fragment i
        predicted_pot = predicted_pot + predicted_pot_i;
        predicted_pot1 = predicted_pot1 + predicted_pot1_i;
        predicted_pot_diff = predicted_pot_diff + predicted_pot_diff_i;
    }



    // Assemble the potentials
    let mut pot: Vec<f64> = Vec::with_capacity(validation_batch_size);
    let mut pot_diff: Vec<f64> = Vec::with_capacity(validation_batch_size*n_diff);
    // For each randomly selected structure from the validation set
    for j in 0..validation_batch_size
    {
        pot.push(global_data_set[ validation_data_set[ random_index[j] ] ].pot);
        pot_diff.append(&mut global_data_set[ validation_data_set[ random_index[j] ] ].pot_diff.clone());
    }
    // Convert the Vecs to Tensors
    let pot: Tensor<(usize, Const<1>), f64, Device, NoneTape> = dev.tensor_from_vec(pot, (validation_batch_size, Const));
    let pot_diff: Tensor<(usize, usize, Const<1>), f64, Device, NoneTape> = dev.tensor_from_vec(pot_diff, (validation_batch_size, n_diff, Const));



    // Calculate the RMSE loss
    predicted_pot_diff = predicted_pot_diff - predicted_pot1.broadcast_like::<_, Axis<1>>(&(validation_batch_size, n_diff, Const));
    let loss_pot = rmse_loss(predicted_pot, pot);
    let loss_pot_diff = rmse_loss(predicted_pot_diff, pot_diff) * (1.0 / EPS);
    let rmse_loss_pot: f64 = loss_pot.array();
    let rmse_loss_pot_diff: f64 = loss_pot_diff.array();



    (rmse_loss_pot, rmse_loss_pot_diff)
}










/// Input a batch size of training data, calculate their predicted values according to their descriptors,
/// return the root mean square error between predicted values and target values (i.e. pot and pot_diff), and update the gradients for the global NN
///
/// # Parameters
/// ```
/// dev: the device for tensor construction
/// global_nn: the input global neural network
/// global_grads: the input mutable gradients of the global neural network
/// batch_data_set: the input batch data set
/// training_loss_pot: the output root mean square error of pot for the input batch data set
/// training_loss_pot_diff: the output root mean square error of pot_diff for the input batch data set
/// ```
///
/// # Examples
/// ```
/// ```
fn single_batch_rmse_mut(dev: &Device, global_nn: &mut GlobalNN, global_grads: &mut Gradients<f64, Device>, batch_data_set: Vec<ProteinSystemDescriptor>) -> (f64, f64)
{
    // Obtain the number of finite difference perturbation structures for the input protein system
    let n_diff: usize = batch_data_set[0].n_diff;
    let training_batch_size: usize = batch_data_set.len();



    // Accumulation variables initialization
    let mut predicted_pot: Tensor<(usize, Const<1>), f64, Device, OwnedTape<f64, Device>> = dev.zeros_like(&(training_batch_size, Const)).traced(global_grads.clone());
    let mut predicted_pot1: Tensor<(usize, Const<1>), f64, Device, OwnedTape<f64, Device>> = dev.zeros_like(&(training_batch_size, Const)).traced(global_grads.clone());
    let mut predicted_pot_diff: Tensor<(usize, usize, Const<1>), f64, Device, OwnedTape<f64, Device>> = dev.zeros_like(&(training_batch_size, n_diff, Const)).traced(global_grads.clone());



    // For each fragment in the input protein system
    for i in 0..batch_data_set[0].fragment_descriptor.len()
    {
        // Assemble the descriptors
        let mut batch_descriptor_vec: Vec<f64> = Vec::with_capacity(training_batch_size*N_DES);
        let mut batch_descriptor_diff_vec: Vec<f64> = Vec::with_capacity(training_batch_size*n_diff*N_DES);
        // For each structure in the batch data set
        for j in 0..training_batch_size
        {
            batch_descriptor_vec.append(&mut batch_data_set[j].fragment_descriptor[i].descriptor.clone());
            batch_descriptor_diff_vec.append(&mut batch_data_set[j].fragment_descriptor[i].descriptor_diff.clone());
        }

        // Convert the Vecs to Tensors
        let batch_descriptor: Tensor<(usize, Const<N_DES>), f64, Device, OwnedTape<f64, Device>> = dev.tensor_from_vec(batch_descriptor_vec.clone(), (training_batch_size, Const)).traced(global_grads.clone());
        let batch_descriptor1: Tensor<(usize, Const<N_DES>), f64, Device, OwnedTape<f64, Device>> = dev.tensor_from_vec(batch_descriptor_vec, (training_batch_size, Const)).traced(global_grads.clone());
        let batch_descriptor_diff: Tensor<(usize, usize, Const<N_DES>), f64, Device, OwnedTape<f64, Device>> = dev.tensor_from_vec(batch_descriptor_diff_vec, (training_batch_size, n_diff, Const)).traced(global_grads.clone());

        // Calculate the predicted pot and pot_diff for Fragment i
        let (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i) = match global_nn.get_fragment_nn_mut(&batch_data_set[0].fragment_descriptor[i].fragment_type)
        {
            NNMut::Small(nn_small) =>
            {
                let predicted_pot_i = nn_small.forward_mut(batch_descriptor);
                let predicted_pot1_i = nn_small.forward_mut(batch_descriptor1);
                let predicted_pot_diff_i = nn_small.forward_mut(batch_descriptor_diff);
                (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i)
            },

            NNMut::Middle(nn_middle) =>
            {
                let predicted_pot_i = nn_middle.forward_mut(batch_descriptor);
                let predicted_pot1_i = nn_middle.forward_mut(batch_descriptor1);
                let predicted_pot_diff_i = nn_middle.forward_mut(batch_descriptor_diff);
                (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i)
            },

            NNMut::Large(nn_large) =>
            {
                let predicted_pot_i = nn_large.forward_mut(batch_descriptor);
                let predicted_pot1_i = nn_large.forward_mut(batch_descriptor1);
                let predicted_pot_diff_i = nn_large.forward_mut(batch_descriptor_diff);
                (predicted_pot_i, predicted_pot1_i, predicted_pot_diff_i)
            },
        };

        // Accumulate the predicted pot and pot_diff of Fragment i
        predicted_pot = predicted_pot + predicted_pot_i;
        predicted_pot1 = predicted_pot1 + predicted_pot1_i;
        predicted_pot_diff = predicted_pot_diff + predicted_pot_diff_i;
    }



    // Assemble the potentials
    let mut pot: Vec<f64> = Vec::with_capacity(training_batch_size);
    let mut pot_diff: Vec<f64> = Vec::with_capacity(training_batch_size*n_diff);
    // For each structure in the batch data set
    for j in 0..training_batch_size
    {
        pot.push(batch_data_set[j].pot);
        pot_diff.append(&mut batch_data_set[j].pot_diff.clone());
    }
    // Convert the Vecs to Tensors
    let pot: Tensor<(usize, Const<1>), f64, Device, NoneTape> = dev.tensor_from_vec(pot, (training_batch_size, Const));
    let pot_diff: Tensor<(usize, usize, Const<1>), f64, Device, NoneTape> = dev.tensor_from_vec(pot_diff, (training_batch_size, n_diff, Const));



    // Calculate the RMSE loss and update the gradients for the global NN
    predicted_pot_diff = predicted_pot_diff - predicted_pot1.broadcast_like::<_, Axis<1>>(&(training_batch_size, n_diff, Const));
    let loss_pot = rmse_loss(predicted_pot, pot);
    let loss_pot_diff = rmse_loss(predicted_pot_diff, pot_diff) * (n_diff as f64 / EPS);
    let rmse_loss_pot: f64 = loss_pot.array();
    let rmse_loss_pot_diff: f64 = loss_pot_diff.array() / (n_diff as f64);
    *global_grads = (loss_pot + loss_pot_diff).backward();



    (rmse_loss_pot, rmse_loss_pot_diff)
}










