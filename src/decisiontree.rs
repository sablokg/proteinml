use crate::classifytensor::maketensor;
use ndarray::Array2;
use smelt_ml::prelude::*;

/*
Gaurav Sablok
gsablok@proton.me
*/

#[tokio::main]
pub async fn decisiontree(pathname: &str) -> Result<String> {
    let tensorunpack = maketensor(pathname).unwrap();

    let nrows = tensorunpack.0.len();
    let ncols = tensorunpack.0[0].len();

    let inputvec: Vec<f64> = tensorunpack
        .0
        .iter()
        .flatten()
        .map(|x| x.to_string().parse::<f64>().unwrap())
        .collect();

    // NOTE: shape must be a tuple `(rows, cols)`, not an array `[rows, cols]`.
    // `[usize; 2]` doesn't implement `ShapeBuilder`, which is why this failed
    // to compile.
    let arraybasevec = Array2::from_shape_vec((nrows, ncols), inputvec)
        .map_err(|e| format!("failed to build feature matrix: {e}"))
        .unwrap();

    let inputlabels: Vec<usize> = tensorunpack
        .1
        .iter()
        .flatten()
        .map(|x| x.to_string().parse::<usize>().unwrap())
        .collect();

    let task = ClassificationTask::new("proteinmal", arraybasevec, inputlabels)
        .map_err(|e| format!("failed to build task: {e}"))
        .unwrap();

    let mut tree = DecisionTree::default();
    let model = tree
        .train_classif(&task)
        .map_err(|e| format!("training failed: {e}"))
        .unwrap();

    let pred = model
        .predict(task.features())
        .map_err(|e| format!("prediction failed: {e}"))
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy
        .score(&pred)
        .map_err(|e| format!("scoring failed: {e}"))
        .unwrap();

    Ok(format!(
        "The decision tree classifier has been trained (accuracy: {acc:.4})"
    ))
}
