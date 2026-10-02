use burn::Tensor;
use burn::tensor::backend::Backend;

/*
Gaurav Sablok
gsablok@proton.me
 */

#[inline(always)]
pub fn bceloss<B: Backend>(probabilities: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
    let epsilon_val = 1e-7;
    let preds = probabilities.clamp(epsilon_val, 1.0 - epsilon_val);
    let loss =
        targets.clone() * preds.clone().log() + (targets.neg() + 1.0) * (preds.neg() + 1.0).log();
    loss.mean().neg()
}
