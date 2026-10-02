use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/

type TENSOR = (Vec<Vec<f32>>, Vec<Vec<f32>>);

#[tokio::main]
pub async fn maketensor(pathfile: &str) -> Result<TENSOR, Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not found");
    let fileread = BufReader::new(fileopen);
    let mut annotatedpeak: Vec<_> = Vec::new();
    let mut labels: Vec<_> = Vec::new();

    for i in fileread.lines() {
        let line = i.expect("line not present");
        let linevec = line
            .split(",")
            .map(|x| x.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        let lineannoation = linevec.clone()[0..linevec.clone().len() - 1]
            .to_vec()
            .iter()
            .map(|x| x.to_string().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        let label = linevec.clone()[linevec.len() - 1..linevec.clone().len()]
            .to_vec()
            .iter()
            .map(|x| x.to_string().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        annotatedpeak.push(lineannoation);
        labels.push(label);
    }

    let finalannotated = confirmpeaks(annotatedpeak).unwrap();
    let finallabels = labels.clone();
    Ok((finalannotated, finallabels))
}

#[tokio::main]
pub async fn confirmpeaks(pathvec: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let maxvalue = pathvec.iter().map(|x| x.len()).collect::<Vec<_>>();
    let maxvec = maxvalue.iter().max().unwrap();
    let mut returnvec: Vec<Vec<f32>> = Vec::new();

    for i in pathvec.iter() {
        let value = i.len();
        let mut vec = i.clone();
        if value == *maxvec {
            returnvec.push(vec);
        } else if value <= *maxvec {
            for _ in 0..*maxvec - value {
                vec.push(0f32);
            }
            returnvec.push(vec);
        }
    }

    Ok(returnvec)
}
