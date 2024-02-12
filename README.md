---------------------------------------------------------------------------------------------------
### Introduction
---------------------------------------------------------------------------------------------------

This README explains how to use the `Weak-PDE-LEARN` library. If you have additional questions (or 
want to report a bug), please email me (Robert) at `rrs254@cornell.edu`.

The `Weak-PDE-LEARN` library consists of several `.py` and `.txt` files. In general, you will only
only need to interact with two of them: `Settings.txt` and `Library.txt`. The former controls all
hyper-parameters and settings while the latter specifies the library (left and right-hand side 
terms) we use to identify the hidden PDE.



---------------------------------------------------------------------------------------------------
### Dependencies 
---------------------------------------------------------------------------------------------------

Our library uses several external packages. We recommend setting up a virtual environment to do this. This should ensure that you have the correct version of each external package/won't run into any package conflicts. 

We include a guide on how to set up the virtual environment using `conda`. Alternatively, you can manually install the packages listed below.

First, open a command line in the `Weak-PDE-LEARN` library and make a new virtual environment using 
`python 3.10`:

`conda create --name WeakPDE python=3.10`

Say yes if conda prompts you. Once you've built the virtual environment, activate it:

`conda activate WeakPDE`

Now, let's add the packages we ned to run the `WeakPDE` code (note: the versions are optional; the versions I list are the ones I used when developing this library. If you encounter any packages errors while running the code, try re-installing the packages with the specified versions):

`conda install numpy=1.26.4`
`conda install torch=2.2.0`
`conda install matplotlib=3.8.2`
`conda install seaborn=0.13.2`
`conda install scipy=1.12.0`
`conda install pandas=2.2.0`

If you can't install any of these packages using `conda`, try using `pip`. Simply replace the word `conda` with `pip`. The virtual environment is now built and you're ready to start using `weak-PDE-LEARN`! 

In the future, to activate the environment (and thus, gain access to all the packages we need to run the `WeakPDE` library), simply activate the environment with `conda activate WeakPDE`.


---------------------------------------------------------------------------------------------------
### Making a dataset
---------------------------------------------------------------------------------------------------

`Weak-PDE-LEARN` comes with a few benchmark datasets pre-installed (see `./Data/DataSets`). Here, we describe how to build a new DataSet that we can use to train `weak-PDE-LEARN`.

A "DataSet" is a `.npz` file that contains a dictionary with six keys: "Training Inputs," "Training Targets," "Testing Inputs," "Testing Targets," "Bounds," and "Number of Dimensions." Each of these keys refers to a `numpy ndarray` object (except "Number of Dimensions," which is an integer that specifies the number of spatial dimensions in the inputs within the data set).

While training, `Weak-PDE-LEARN` will update the system response functions using only the training 
set data. However, `Weak-PDE-LEARN` will report the loss on both the training and test sets. You can use this to determine if the solution networks are over-fitting the training set. 

If you want to use `PDE-LEARN` on your data, you must write a program that calls the `Create_Data_Set` function (in `./Data/Create_Data_Set.py`) with the appropriate arguments. See that function's doc-string for details. 

Alternatively, you can create a DataSet using one of our `MATLAB` data sets by running `Python ./From_MATLAB.py` when your current working directory is `Data.` The `From_MATLAB` file contains five settings: "Data_File_Name," "Num_Spatial_Dimensions," "Noise_Proportion," "Num_Train_Examples," and "Num_Test_Examples." "Data_File_Name" should refer to one of the `.mat` files in the `MATLAB/Data` directory "Num_Spatial_Dimensions" specifies the number of spatial dimensions in the inputs stored in the `.mat` file. "Noise_Proportion," "Num_Train_Examples," and "Num_Test_Examples" control the level of noise in the data, the number of training data points, and the number of testing data points, respectively.

Once you've built your DataSet, you need to tell `Weak-PDE-LEARN` to train on it. Recall that `Weak-PDE-LEARN` trains a neural network to match the data in a dataset. At the bottom of `Settings.txt`, you'll see a setting titled `DataSet Names`. You should set this to be a list of strings. Each entry should be the name of a DataSet in `./Data/DataSet`. Note that `weak-PDE-LEARN` assumes that the SAME PDE governs the system response function in EACH dataset. If one dataset comes from Burgers' equation, and another comes from the KdV equation, `weak-PDE-LEARN` will do strange things. Don't make it do that!



---------------------------------------------------------------------------------------------------
### Running the Code 
---------------------------------------------------------------------------------------------------

Once you have selected your settings you, can start training by running `Code/main.py`. To do this, run the following command (the `Weak-PDE-LEARN` environment should be activated at this stage):

`python ./Code/main.py`

Remember that there are often several rounds of training. See the paper for details. 

**What to do if you get nan:** `Weak-PDE-LEARN` can use the `LBFGS` optimizer. Unfortunately, `PyTorch`'s `LBFGS` optimizer is known to yield nan (see <https://github.com/pytorch/pytorch/issues/5953>). Using the `LBFGS` optimizer occasionally causes `Weak-PDE-LEARN` to break down and start reporting `nan.` If this happens, you should kill `Weak-PDE-LEARN` (in the terminal window, press `Ctrl + C`) and then re-run `Weak-PDE-LEARN.` Since `Weak-PDE-LEARN` randomly samples the weight functions, no two runs of `Weak-PDE-LEARN` are identical. Thus, even if you keep the settings the same, re-running `Weak-PDE-LEARN` may avoid the `nan` issue. If you encounter `nan` on several successive runs of `Weak-PDE-LEARN,` reduce the learning rate by a factor of $10$ and try again. If all else fails, consider training using another optimizer.