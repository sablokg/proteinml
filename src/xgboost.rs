use crate::classifytensor::maketensor;
use ndarray::Array2;
use smelt_ml::prelude::*;
use smelt_ml::tuning::{BayesianOptimizer, ParamDistribution, ParamSpace, ParamValue};

/*
Gaurav Sablok
gsablok@proton.me
*/

#[tokio::main]
pub async fn boost(pathname: &str) -> Result<String> {
    let tensorunpack = maketensor(pathname).unwrap();

    let inputvec = tensorunpack
        .0
        .iter()
        .map(|x| {
            x.iter()
                .map(|x| x.to_string().parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .iter()
        .flatten()
        .collect::<Vec<_>>()
        .iter()
        .map(|x| **x)
        .collect::<Vec<_>>();
    let arraybasevec =
        Array2::from_shape_vec([tensorunpack.0.len(), tensorunpack.0[0].len()], inputvec).unwrap();
    let inputlabels = tensorunpack
        .1
        .iter()
        .flatten()
        .collect::<Vec<_>>()
        .iter()
        .map(|x| x.to_string().parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let task = ClassificationTask::new("proteinml", arraybasevec, inputlabels).unwrap();
    let cv = CrossValidation::new(4).with_seed(42);
    let mut space = ParamSpace::new();
    space.insert("max_depth".into(), ParamDistribution::Uniform(2.0, 8.0));
    space.insert(
        "learning_rate".into(),
        ParamDistribution::LogUniform(0.01, 0.5),
    );
    space.insert(
        "n_estimators".into(),
        ParamDistribution::Choice(vec![
            ParamValue::Int(25),
            ParamValue::Int(50),
            ParamValue::Int(100),
        ]),
    );

    println!("=== Bayesian Optimization (TPE) ===");
    let bo = BayesianOptimizer::new(
        |_params| {
            Box::new(
                XGBoost::new()
                    .with_n_estimators(100)
                    .with_max_depth(6)
                    .with_learning_rate(0.3)
                    .with_lambda(1.0) // L2 regularization
                    .with_subsample(0.8)
                    .with_colsample_bytree(0.8)
                    .with_early_stopping_rounds(10),
            )
        },
        space.clone(),
    )
    .with_n_iter(20)
    .with_seed(42);

    let bo_result = bo.tune_classif(&task, &cv, &Accuracy).unwrap();
    println!("Best score: {:.4}", bo_result.best_score);
    println!("Best params:");
    for (k, v) in &bo_result.best_params {
        println!("  {:?}: {:.4?}", k, v);
    }
    println!("\n=== Random Search (same budget) ===");
    let rs = smelt_ml::tuning::RandomSearch::new(
        |_params| {
            Box::new(
                XGBoost::new()
                    .with_n_estimators(100)
                    .with_max_depth(6)
                    .with_learning_rate(0.3)
                    .with_lambda(1.0) // L2 regularization
                    .with_subsample(0.8)
                    .with_colsample_bytree(0.8)
                    .with_early_stopping_rounds(10),
            )
        },
        space,
    )
    .with_n_iter(20)
    .with_seed(42);
    let rs_result = rs.tune_classif(&task, &cv, &Accuracy).unwrap();
    println!("Best score: {:.4}", rs_result.best_score);
    println!("\n=== Summary ===");
    println!(
        "Bayesian: {:.4}  |  Random: {:.4}",
        bo_result.best_score, rs_result.best_score
    );
    Ok("The xgboost classifier has been trained".to_string())
}
