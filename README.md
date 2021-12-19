# README for G04

Our project relies mainly on sklearn, pandas, PyTorch, and matplotlib.
Running the code requires the following files:

&emsp;&emsp; ./requirements.txt &emsp;&nbsp; <-- python environment requirements.txt file <br />
&emsp;&emsp; ./main.py &emsp;&emsp;&emsp;&emsp;&emsp; <-- To run the entire project 


  
## Utilities
Our code utilities for classification, regression, and novelty component require the following files:

&emsp;&emsp; ./Models/import_data.py &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; <-- script to read raw data files <br />
&emsp;&emsp; ./Models/modelling.py &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp; <-- script to train and evaluate models <br />
&emsp;&emsp; ./Models/plotting.py &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp; <-- script to create plots <br />
&emsp;&emsp; ./Models/export_data.py &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; <-- script to export plots and model evaluations <br />
&emsp;&emsp; ./Models/Training_parameters/*.py &nbsp;&nbsp;&nbsp; <-- model parameters for each dataset <br />

Our code utilities for classifier interpretability require the following files:

&emsp;&emsp; ./Models/import_batches.py &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; <-- script to import batches <br />
&emsp;&emsp; ./Models/Classifier_interpretability/classifier_interpretability.py &nbsp;&nbsp; <-- script to run models <br />
&emsp;&emsp; ./Models/Classifier_interpretability/*.pkl &nbsp;&nbsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<-- script to store trained models <br />
  
* The files should be run in the order:
   main.py
* GPU is not required.
* The main.py script creates a "Results" directory (for CL, REGR) and an "out_img" directory (for DTC, CNN) and saves results there.
* Training takes ~1 day (4 cores 3.5Ghz).
* To lower training time to 2 min:
  * Access ./Models/import_data.py, forward to the read_files_param_grid variable at the bottom, comment out every dataset (and their parameters) except one. remember the dataset.
  * Access ./Models/Training_parameters/{dataset_in_read_files_param_grid}.py and comment out most models
  * Access ./main.py and comment out the last line: classifier_interpretability.initialize_ci()
  * execute ./main.py