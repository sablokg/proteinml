use crate::classifytensor::maketensor;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::linear::logistic_regression::LogisticRegressionParameters;
use smartcore::metrics::accuracy;
use smartcore::model_selection::KFold;
use smartcore::model_selection::cross_validate;
use smartcore::model_selection::train_test_split;
use std::error::Error;

/*
Gaurav Sablok
gsablok@proton.me
 */

#[tokio::main]
pub async fn crossvalidate(pathname: &str) -> Result<String, Box<dyn Error>> {
    let maketensor = maketensor(pathname).unwrap();

    let vectorconvert = maketensor
        .0
        .iter()
        .map(|x| {
            x.iter()
                .map(|x| x.to_string().parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let labelconvert = maketensor
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

    let vectordense = DenseMatrix::from_2d_vec(&vectorconvert).unwrap();

    let trainsplit = train_test_split(&vectordense, &finallabel, 0.2, true, Some(2811));

    let trainsplit_1 = trainsplit.0;
    let trainsplit_2 = trainsplit.1;
    let trainsplit_3 = trainsplit.2;
    let trainsplit_4 = trainsplit.3;

    let parameters_log = KFold::default().with_n_splits(5).with_shuffle(true);
    let results = cross_validate(
        LogisticRegression::fit(
            &trainsplit_1,
            &trainsplit_3,
            LogisticRegressionParameters::default(),
        )
        .unwrap(),
        &trainsplit_1,
        &trainsplit_3,
        LogisticRegressionParameters::<f64>::default(),
        &parameters_log,
        &accuracy,
    )
    .unwrap();

    let mean_score: f64 = results.test_score.iter().sum::<f64>() / results.test_score.len() as f64;

    println!(
        "Mean score of cross validation model with the trainsplit train  dataset used is {}",
        mean_score
    );

    let results_test = cross_validate(
        LogisticRegression::fit(
            &trainsplit_2,
            &trainsplit_4,
            LogisticRegressionParameters::default(),
        )
        .unwrap(),
        &trainsplit_2,
        &trainsplit_4,
        LogisticRegressionParameters::<f64>::default(),
        &parameters_log,
        &accuracy,
    )
    .unwrap();

    let mean_test: f64 =
        results_test.test_score.iter().sum::<f64>() / results_test.test_score.len() as f64;

    println!(
        "The cross validate mean score on the test data split are as follows:{}",
        mean_test
    );

    Ok("The cross validate has been run".to_string())
}
