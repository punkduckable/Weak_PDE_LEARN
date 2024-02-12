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

Our library uses several external packages. We recommend setting up a virtual environment to do 
this. This should ensure that you have the correct version of each external package/won't run into
any package conflicts. 

We include a guide on how to set up the virtual environment using `conda`. Alternatively, you can 
manually install the packages listed below.

First, open a command line in the `Weak-PDE-LEARN` library and make a new virtual environment using 
`python 3.10`:

`conda create --name WeakPDE python=3.10`

Say yes if conda prompts you. Once you've built the virtual environment, activate it:

`conda activate WeakPDE`

Now, let's add the packages we ned to run the `WeakPDE` code (note: the versions are optional; the 
versions I list are the ones I used when developing this library. If you encounter any packages 
errors while running the code, try re-installing the packages with the specified versions):

`conda install numpy=1.26.4`
`conda install torch=2.2.0`
`conda install matplotlib=3.8.2`
`conda install seaborn=0.13.2`
`conda install scipy=1.12.0`
`conda install pandas=2.2.0`

If you can't install any of these packages using `conda`, try using `pip`. Simply replace the 
word `conda` with `pip`. The virtual environment is now built and you're ready to start using 
`weak-PDE-LEARN`! 

In the future, to activate the environment (and thus, gain access to all the packages we need to
run the `WeakPDE` library), simply activate the environment with `conda activate WeakPDE`.



---------------------------------------------------------------------------------------------------
### Running the code 
---------------------------------------------------------------------------------------------------

Once you have selected your settings you, can start training by running `Code/main.py`. To do this, 
run the following command (the `Weak-PDE-LEARN` environment should be activated at this stage):

`python ./Code/main.py`

Remember that there are often several rounds of training. See the paper for details. 

