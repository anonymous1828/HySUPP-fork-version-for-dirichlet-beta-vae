# HySUPP version for submission of paper "Exploring Dirichlet priors in $\beta$-VAE Inverse Problem" 

This is a patched version of the open-source **Hy**per**S**pectral **U**nmixing **P**ython **P**ackage. 
The original Github repository can be found at https://github.com/BehnoodRasti/HySUPP.git

---

## Introduction

The goal of this repository is to provide an integration of the Dirichlet $\beta$-VAE to the open-source HySUPP package, a recent toolbox for hyperspectral unmixing practitioners. 
Modifications, as minimal as possible, were made to integrate the Dirichlet beta-VAE as well as the Samson and Urban real datasets with 4, 5 and 6 ground-truth endmembers. They are well documented in the commit and summarized in the section below. 

## Getting started with Dirichlet $\beta$-VAE 

### Installation of the environment

#### Using `conda`

We recommend using a `conda` virtual Python environment to install HySUPP.

In the following steps we will use `conda` to handle the Python distribution and `pip` to install the required Python packages.
If you do not have `conda`, please install it using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```
conda create --name hysupp python=3.10
```

Activate the new `conda` environment to install the Python packages.

```
conda activate hysupp
```

Clone the Github repository.

```
git clone https://github.com/anonymous1828/HySUPP-fork-version-for-dirichlet-beta-vae.git
```

Change directory and install the required Python packages.

```
cd HySUPP && pip install -r requirements.txt
```

If you encounter any issue when installing `spams`, we recommend reading the Installation section [here](https://pypi.org/project/spams/).
For windows users, we suggest removing line 10 in the requirements.txt (spams==2.6.5.4), and after installing the requirements, install spams using pip install spams-bin.

Another simple workaround consists in running the following commands:
```
git clone https://github.com/getspams/spams-python
cd spams-python
pip install -e .
```

### Download Datasets

For this submission, we worked with Samson and Urban datasets. Samson is available directly and can be used under the name of "adapted_for_hysupp_Samson" as shown previously. 
Urban is too heavy to be stored on Github. To obtain it follow those steps
1) download the dataset 
   - observations https://rslab.ut.ac.ir/documents/437291/1493656/Urban_R162.mat/24b3640f-ea17-a8cc-6e09-9b2ff22bf8c3?t=1710110006859&download=true
   - 4 endmembers ground truth : https://rslab.ut.ac.ir/documents/437291/1493656/groundTruth_4_end.zip/c03d6f3a-cb26-865d-6d4b-04c4480bdc57?t=1710109903885&download=true
   - 5 endmembers ground truth : https://rslab.ut.ac.ir/documents/437291/1493656/groundTruth_Urban_end5.zip/fe72c2db-d724-b848-92f7-c5fb0d6d1e79?t=1710109956268&download=true
   - 6 endmembers : https://rslab.ut.ac.ir/documents/437291/1493656/groundTruth_Urban_end6.zip/4ae2518a-298b-bdd2-0e28-8ee60c8e3ef1?t=1710109871882&download=true
2) Create a directory in ./data/ named Urban*k* where *k* is the number of endmembers (Urban4 for instance) and place the Urban_R162.mat and the groundTruth_Urban_end*k*.zip (for Urban4, your directory must contain Urban_R162.mat and groundTruth_Urban_end4.zip)
3) unzip groundTruth_Urban_end*k*.zip and drag the content in ./data/Urban*k* (for instance your Urban4 must now contain Urban_R162.mat, groundTruth_Urban_end4.zip and end4_groundTruth.mat)
4) in the terminal, place yourself in the project root directory and run the following command ``` python utils/adapt_data.py --name_origin_data Urban*k* ``` (for instance for Urban4 run ``` python utils/adapt_data.py --name_origin_data Urban4 ```)

If it executes well, a new file should appear in the data directory named adapted_for_hysupp_Urban*k*.mat (for Urban4 you should now see adapted_for_hysupp_Urban4.mat)


### Run the Dirichlet $\beta$-VAE 

This toolbox uses [MLXP](https://inria-thoth.github.io/mlxp/) to manage multiple experiments built on top of [hydra](https://hydra.cc/).

There are a few required parameters to define in order to run an experiment:
* `data`: hyperspectral unmixing dataset
* `model`: unmixing model
* `SNR`: input SNR (*optional*)
* `mode`: unmixing mode (no longer needed!)

The Dirichlet $\beta$-VAE. To run it, you can use the following line or modify the config file DirVAE.yaml to set default hyperparameters

```shell
python unmixing.py data=adapted_for_hysupp_Samson model=DirVAE model.epochs=200 model.lr=0.001 model.reg_factor=0.001
```
The results of the experiment can be found in the logs directory located at ```./logs\```. To facilitate its access, we propose a demo.ipynb file where one can access the selected log_id and display the metrics.


## Modifications
- Model files:
  - src/model/blind/DirVAE.py file is the implementation of the model  
  - src/model/blind/ADMMNet.py didn't integrate the random seed in the VCA, this has been fixed  
  - src/model/blind/CNNAEU.py has been modified to call the implementation of Tensorflow RMSprop in Pytorch and recover the performances of the original paper which was implemented in Tensorflow
  - src/model/blind/\__init__.py integrates DirVAE declaration
- Dataset files:
  - utils/adapt_data_for_hysupp.py is a file to extract and convert Samson et Urban data in the layout of HySUPP
  - data/base.py have been modified to integrate Urban and Samson (class RealHSI)
- Utils:
  - src/utils/tf_rms_prop.py contains the RMSpropTF class, an implementation of tensorflow RMSprop into Pytorch. The difference lies in the intervertion of operation in the last update of the weights: compute the sum of epsilon in gradient before square rooting it (see Torch RMS prop documentation for further details)
  - src/utils/constraint.py clean implementation of nonnegativty constraint to ensure nonnegativity in the weights of the decoder
  - src/utils/metrics.py has been modified to add the DKL between 2 Dirichlet distributions class refered to as GammaKL
- Main files:
  - src/blind.py didn't integrate the random seed in the model.compute_endmembers_and_abundances method of the model, this has been fixed + an If condition has been added line 36 to avoid apply noise when Samson or Urban datasets are selected
- Config files:
  - config/data/adapted_for_hysupp_*.yaml with * = Samson, Urban4, Urban5, Urban6 have been added as part of the intregration
  - config/model/DirVAE.yaml has also been added for same reasons

## Data format

Datasets consist in a dedicated `.mat` file containing the following keys:

* `Y`: original hyperspectral image (dimension `L` x `N`)
* `E`: ground truth endmembers (dimension `L` x `p`)
* `A`: ground truth abundances (dimension `p` x `N`)
* `H`: HSI number of rows
* `W`: HSI number of columns
* `p`: number of endmembers
* `L`: number of channels
* `N`: number of pixels (`N` == `H`*`W`)






