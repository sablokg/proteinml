mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
mod classifysmart;
mod classifytensor;
mod decisiontree;
mod xgboost;
use crate::classifysmart::logclassify;
use crate::decisiontree::decisiontree;
use crate::xgboost::boost;
mod knn;
mod random;
use crate::knn::knnclassify;
use crate::random::randomclassify;
mod crossvalidate;
mod randomcross;
use crate::crossvalidate::crossvalidate;
use crate::randomcross::randomcrossprotein;
use figlet_rs::FIGfont;
mod bceloss;
mod burnneural;
use crate::burnconvd::train;
mod burnconvd;
use self::burnneural::burnarray;
use burn::tensor::backend::AutodiffBackend;
mod gridsearch;

/*
Gaurav Sablok
gsablok@proton.me
*/

fn main() {
    let standard_font = FIGfont::standard().unwrap();
    let figure = standard_font.convert("proteinML");
    assert!(figure.is_some());
    println!("{}", figure.unwrap());
    let argparse = CommandParse::parse();
    match &argparse.command {
        Commands::Logistic { peaks, thread } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .unwrap();
            pool.install(|| {
                let value = logclassify(peaks).unwrap();
                print!("The command has finished:{:?}", value);
            })
        }
        Commands::DecisionTree { peaks, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed");
            value.install(|| {
                let command = decisiontree(peaks).unwrap();
                println!("The command has finished:{}", command);
            });
        }
        Commands::XgBoost { peaks, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed");
            value.install(|| {
                let command = boost(peaks).unwrap();
                println!("The command has finished:{}", command);
            });
        }
        Commands::KNN { peaks, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed");
            value.install(|| {
                let command = knnclassify(peaks).unwrap();
                println!("The command has finished:{}", command);
            });
        }
        Commands::RandomForest { peaks, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed");
            value.install(|| {
                let commands = randomclassify(peaks).unwrap();
                println!("The command has finished:{}", commands);
            });
        }
        Commands::CrossValidate { peaks, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed");
            value.install(|| {
                let command = crossvalidate(peaks).unwrap();
                println!("The cross validate has been finished:{}", command);
            });
        }
        Commands::RandomCross { peaks, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed");
            value.install(|| {
                let command = randomcrossprotein(peaks).unwrap();
                println!("The commad has finished:{}", command);
            });
        }
        Commands::BurnClass {
            pathname,
            seqlength,
        } => {
            let threads = 4.to_string();
            let threadslaunch = threads.parse::<usize>().unwrap();
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(threadslaunch)
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                train::<B: AutodiffBackend>(device, pathname, *seqlength);
            });
        }
        Commands::LogisticNeural { peaks, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed");
            value.install(|| {
                let command = burnarray(peaks).unwrap();
                println!("The command has finished:{}", command);
            })
        }
    }
}
