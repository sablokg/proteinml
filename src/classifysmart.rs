use crate::classifytensor::maketensor;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::model_selection::train_test_split;
use smelt_ml::prelude::*;

/*
Gaurav Sablok
gsablok@proton.me
*/

#[tokio::main]
pub async fn tensor(pathname: &str) -> Result<(DenseMatrix<f64>, Vec<i32>)> {
    let tensorunpack = maketensor(pathname).unwrap();

    let inputvec = tensorunpack.0;
    let inputlabel = tensorunpack.1;

    let vectorconvert = inputvec
        .iter()
        .map(|x| {
            x.iter()
                .map(|x| x.to_string().parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let labelconvert = inputlabel
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

    let tensorinput = DenseMatrix::from_2d_vec(&vectorconvert).expect("vector not found");

    Ok((tensorinput, finallabel))
}

#[tokio::main]
pub async fn logclassify(pathname: &str) -> Result<String> {
    let unpack = tensor(pathname).unwrap();

    let trainsplit = train_test_split(&unpack.0, &unpack.1, 0.2, true, Some(2811));

    let trainsplit_0 = trainsplit.0;
    let trainsplit_1 = trainsplit.1;
    let trainsplit_2 = trainsplit.2;
    let trainsplit_3 = trainsplit.3;

    let logistic =
        LogisticRegression::fit(&trainsplit_0, &trainsplit_2, Default::default()).unwrap();

    let logisiticpredict = logistic.predict(&trainsplit_1).unwrap();
    let log_accuracy = smartcore::metrics::accuracy(&trainsplit_3, &logisiticpredict);
    println!("The logisitic accuracy is:{}", log_accuracy);

    Ok("The classifer has been trained".to_string())
}
