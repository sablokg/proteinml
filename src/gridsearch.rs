use crate::classifytensor::maketensor;
use smartcore::api::SupervisedEstimator;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use smartcore::model_selection::{KFold, cross_validate};
use std::error::Error;
/*
Gaurav Sablok
gsablok@proton.me
*/

fn gridrandom(pathname: &str) -> Result<String, Box<dyn Error>> {
    let arrayunrarp = maketensor(pathname).unwrap();

    let vectorconvert = arrayunrarp
        .0
        .iter()
        .map(|x| {
            x.iter()
                .map(|x| x.to_string().parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let labelconvert = arrayunrarp
        .1
        .iter()
        .map(|x| {
            x.iter()
                .map(|x| x.to_string().parse::<i32>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let finallabel = labelconvert
        .iter()
        .map(|x| x.clone())
        .flatten()
        .collect::<Vec<_>>();

    let cv = KFold::default().with_n_splits(5);
    let n_trees = [10, 50, 100];
    let max_tree_option = [Some(3), Some(5), Some(7), None];

    let mut best_score = f64::MIN;
    let mut best_params = None;

    let densematrix = DenseMatrix::from_2d_vec(&vectorconvert).unwrap();

    for &tree in &n_trees {
        for &maxtree in &max_tree_option {
            let params = RandomForestClassifierParameters::default()
                .with_n_trees(tree)
                .with_max_depth(maxtree.unwrap_or(u16::MAX as u16));

            let result = cross_validate(
                RandomForestClassifier::new(),
                &densematrix,
                &finallabel,
                params.clone(),
                &cv,
                &accuracy,
            )
            .unwrap();

            let score = result.mean_test_score();
            println!("n_trees ={tree}, max_depth={maxtree:?} -> acc = {score:.4}");

            if score > best_score {
                best_score = score;
                best_params = Some(params);
            }
        }
    }

    println!("Best score: {best_score:.4}, params: {:?}", best_params);

    Ok("The random grid search has been implemented".to_string())
}
