# proteinml

- Entire protein machine learning in one crate using the annotated peaks from metabolomics and the sample labels.
- Checks that the vectors forms are equal for the metabolomics tensor.

-Completed
- Logistic regression 
- Random Forest 
- KNN Classifier 
- Decision Tree 
- XgBoost with Gradient Boosting and transformer and L2 regularization
- Cross Validation 
- Standard Scaler
- Logistic Neural 
- Conv1D using Burn 

```
cargo build
```

```
                         _            _           __  __   _     
  _ __    _ __    ___   | |_    ___  (_)  _ __   |  \/  | | |    
 | '_ \  | '__|  / _ \  | __|  / _ \ | | | '_ \  | |\/| | | |    
 | |_) | | |    | (_) | | |_  |  __/ | | | | | | | |  | | | |___ 
 | .__/  |_|     \___/   \__|  \___| |_| |_| |_| |_|  |_| |_____|
 |_|                                                             

deep learning classification based on the annotated peak
       ************************************************
       Author Gaurav Sablok
       Email gsablok@proton.me
      ************************************************

Usage: proteinml <COMMAND>

Commands:
  logistic        Generate logistic neural
  decision-tree   Decision Tree
  xg-boost        XgBoost
  random-forest   RandomForest
  knn             KNNClassifier
  cross-validate  Crossvalidate
  random-cross    RandomCross
  burn-class      Neural Burn classification
  help            Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version

```
