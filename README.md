# Projet1 ML EPFL

## Learning to discover: the Higgs boson machine learning challenge

### Team member
Nino Hervé, nino.herve@epfl.ch   
Florent Jeanpetit, florent.jeanpetit@epfl.ch   
Margaux Roulet, margaux.roulet@epfl.ch  

**Team name: Cortana**

### Project Description
Binary classifier of Higgs Boson from CERN raw data.

### Report
The report contains a methodic search for an optimized implementation of machine learning regression methods. It includes processing of data, feature engineering and model selection.

### Folders and Files
#### data
The pathway to open the csv datafile are referenced as ../data/"train.csv" or ../data/"test.csv". We invite you to upload your csv files into this folder.  
The submission file is saved in this folder. The name of the file is submissioncortana.csv

#### scripts
* implementations.py: includes the six regressions methods
* run.py : includes our optimized machine learning model. You can run this file from raw data and it will create a submission file in csv format containing prediction and their respective indexes.
* helpers.py: includes small algorithms as computation functions, cross validation,...
* data_manager.py: includes all data processing functions
* proj1_helpers.py: provided with the project. They contain loading data functions and creating submission function
* plot.py: include plot functions generated with our consoles to visualize our results.
* test_runners.py: iteration functions that allows to run through all our methods with different parameters from a single notebook.

* main_notebook.pypnb: Includes our Machine Learning procedure:
  * test consol to visualize radom parameters and plot 2D graphs
  * test basic regression methods implementation
  * find optimal hyperparameters for ridge regression
  * test the accuracy score from train.csv for the best model chosen
  * run the code for our final submission

### instruction to run
* add the 2 .csv files from the CERN in the data folder
* run run.py on your favorite interface
