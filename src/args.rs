use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "proteoseek",
    version = "1.0",
    about = "deep learning classification based on the annotated peak
       ************************************************
       Author Gaurav Sablok
       Email gsablok@proton.me
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Generate logistic neural
    Logistic {
        /// path to the annotated peaks
        peaks: String,
        /// threads for the analysis
        thread: String,
    },
    /// Decision Tree
    DecisionTree {
        /// path to the annotated peaks
        peaks: String,
        /// threads for the analysis
        thread: String,
    },
    /// XgBoost
    XgBoost {
        /// path to the annotated peaks
        peaks: String,
        /// threads for the analysis
        thread: String,
    },
    /// RandomForest
    RandomForest {
        /// path to the annotated peaks
        peaks: String,
        /// threads for the analysis
        thread: String,
    },
    /// KNNClassifier
    KNN {
        /// path to the annotated peaks
        peaks: String,
        /// threads for the analysis
        thread: String,
    },
    /// Crossvalidate
    CrossValidate {
        /// path to the annotated peaks
        peaks: String,
        /// threads for the analysis
        thread: String,
    },
    /// RandomCross
    RandomCross {
        /// path to the annotated peaks
        peaks: String,
        /// threads for the analysis
        thread: String,
    },
    /// Neural Burn classification
    BurnClass {
        /// path to the filename
        pathname: String,
        /// sequence length
        seqlength: usize,
    },
    /// Logistic Neural
    LogisticNeural {
        /// path to the annotated peaks
        peaks: String,
        /// threads for the analysis
        thread: String,
    },
}
