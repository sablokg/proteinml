use crate::bceloss::bceloss;
use crate::classifytensor::maketensor;
use burn::Tensor;
use burn::backend::{Autodiff, NdArray};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::tensor::TensorData;
use std::default::Default;
use std::error::Error;

/*
Gaurav Sablok
gsablok@proton.me
 */

#[derive(Module, Debug)]
pub struct LinearRegression<B: Backend> {
    pub linear: Linear<B>,
}

impl<B: Backend> LinearRegression<B> {
    pub fn new(inputsize: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(inputsize, 1).init(device);
        Self { linear }
    }
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let logits = self.linear.forward(x);
        burn::tensor::activation::sigmoid(logits)
    }
}

pub fn burnneural<B: Backend>(
    pathnamefile: &str,
    device: &B::Device,
) -> Result<(Tensor<B, 2>, Tensor<B, 2>), Box<dyn Error>> {
    let tensor = maketensor(pathnamefile).unwrap();

    let x_tensor: Vec<f32> = tensor.0.iter().flatten().map(|x| *x as f32).collect();
    let x_shape = [tensor.0.len(), tensor.0[0].len()];
    let x_tensor_data = TensorData::new(x_tensor, x_shape);
    let tensor_x_burn = Tensor::<B, 2>::from_data(x_tensor_data, device);

    let y_flat: Vec<f32> = tensor.1.iter().map(|x| x.clone()).flatten().collect();
    let y_len = y_flat.len();
    let y_tensor_data = TensorData::new(y_flat, [y_len, 1]);
    let tensor_y_burn = Tensor::<B, 2>::from_data(y_tensor_data, device);

    Ok((tensor_x_burn, tensor_y_burn))
}

use burn::tensor::backend::AutodiffBackend;

pub fn runmodel<B: AutodiffBackend>(pathfile: &str, device: &B::Device) {
    let unpacktensor = burnneural::<B>(pathfile, device).unwrap();
    let tensor_x_burn = unpacktensor.0;
    let tensor_y_burn = unpacktensor.1;

    let mut model = LinearRegression::<B>::new(2, device);
    let mut optim = SgdConfig::new().init();

    for epoch in 0..100 {
        let preds = model.forward(tensor_x_burn.clone());
        let loss = bceloss(preds, tensor_y_burn.clone());
        let loss_value = loss.clone().into_scalar();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(0.5, model, grads);

        if epoch % 50 == 0 {
            println!("Epoch: {epoch} and the loss is {:.4}", loss_value);
        }
    }
}

pub fn burnarray(pathname: &str) -> Result<String, Box<dyn Error>> {
    type MyBackend = Autodiff<NdArray>;
    let device = Default::default();
    runmodel::<MyBackend>(pathname, &device);
    Ok("The model has run the neural network".to_string())
}
