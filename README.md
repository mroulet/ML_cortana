# Projet1 ML EPFL

## Learning to discover: the Higgs boson machine learning challenge

### Team member
* Item Nino Hervé, nino.herve@epfl.ch
* Item Florent Jeanpetit, florent.jeanpetit@epfl.ch
* Item Margaux Roulet, margaux.roulet@epfl.ch

### Project Description
Binary classifier of Higgs Boson from CERN raw data.

### Report
The report contains a methodic search for an optimized implementation including processing of data, feature engineering and model selection.

### Folders and Files
#### data
The pathway to open the csv datafile are referenced is ../data/"train.csv" or ../data/"test.csv". We invite you to upload your csv file into this folder.

#### plot
Includes png plots generated with our console. They illustrate our results.

#### script
* implementations.py: includes the six regressions methods
* run.py : Includes our best model submission. You can run this file from raw data and it will create a submission file in csv format containing prediction and their respective indexes.
* helpers.py: Includes small algorithms as computation functions, cross validation,...
* proj1_helpers.py: Provided with the project. They contain loading data functions and creating submission function
* plot.py: Include plot functions generated with our consoles to visualize our results.
* test_runners.py: Includes our main tests. Iteration functions that allows to run through all our methods with different parameters from a single notebook.

* main_notebook.pypnb: Includes our Machine Learning procedure:
  * test basic regression methods implementation
  * find optimal hyperparameters for ridge regression
  * test the accuracy score from train.csv for the best model chosen
  * run the code for our final submission
