use crate::classifytensor::maketensor;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use std::error::Error;

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn randomclassify(pathname: &str) -> Result<String, Box<dyn Error>> {
    let knntensor = maketensor(pathname).unwrap();

    let inputvalues = knntensor.0;
    let inputlabels = knntensor
        .1
        .iter()
        .flatten()
        .collect::<Vec<_>>()
        .iter()
        .map(|x| x.to_string().parse::<i32>().unwrap())
        .collect::<Vec<_>>();

    let inputmatrix = DenseMatrix::from_2d_vec(&inputvalues).expect("matrix cant be formed");

    let trainsplit = train_test_split(&inputmatrix, &inputlabels, 0.2, true, Some(2849));

    let trainsplit_0 = trainsplit.0;
    let trainsplit_1 = trainsplit.1;
    let trainsplit_2 = trainsplit.2;
    let trainsplit_3 = trainsplit.3;

    let randomclassify =
        RandomForestClassifier::fit(&trainsplit_0, &trainsplit_2, Default::default()).unwrap();

    let randompredict = randomclassify.predict(&trainsplit_1).unwrap();

    let accuracy_knn = accuracy(&trainsplit_3, &randompredict);

    println!("The accuracy of the model is {}", accuracy_knn);

    Ok("String has been parsed".to_string())
}
