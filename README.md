
<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://github.com/user-attachments/assets/1ec9e217-508d-4b10-a41a-08dface269c7" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

# Welcome to views-models repository! 

This repository contains all of the necesary components for creating new models which are compatible with the VIEWS pipeline. The views-models repository also contains all of the already implemented VIEWS models (with the exception of [HydraNet](https://github.com/views-platform/views-hydranet)), at both PRIO-GRID-month and country-month levels of analysis, along with information about prediction targets, input data and model algorithms. 

---

## .env Template

```
# .env
cm_path=""
pgm_path=""
month_to_update=[]

APPWRITE_ENDPOINT=""
APPWRITE_DATASTORE_PROJECT_ID=""
```

---

## Table of contents

<!-- toc -->

- [Key Terms and Definitions](#key-terms-and-definitions)
- [Time Partitioning](#time-partitioning)
- [Model Naming Conventions](#model-naming-conventions) 
- [Creating New Models](#creating-new-models)
- [Model scripts](#model-scripts)
- [Model filesystems](#model-filesystems)
- [Running a single model](#running-a-single-model)
- [Ensembles](#ensembles)
- [Creating a new ensemble](#creating-a-new-ensemble)
- [Ensemble scripts](#ensemble-scripts)
- [Ensemble filesystem](#ensemble-filesystem)
- [Running an ensemble](#running-an-ensemble)
- [Integration Testing](#integration-testing)
- [Implemented Models](#implemented-models)
- [Model Catalogs](#catalogs)
    - [Country-Month Models](#country-month-model-catalog)
    - [PRIO-GRID-Month Model](#prio-grid-month-model-catalog)
    - [Ensembles](#ensemble-catalog)

<!-- tocstop -->

---

## Key Terms and Definitions 
<a name="key-terms-and-definitions"></a>

In VIEWS terminology a **model** is defined as: 
1. A specific instantiation of a machine learning algorithm,
2. Trained using a predetermined and unique set of hyperparameters,
3. On a well-defined set of input features,
    - The specific input features for every model are referred to as [querysets](https://github.com/prio-data/viewser?tab=readme-ov-file#via-api). 
4. A model predicts specific outcome target or targets.
5. In the case of [stepshift models](https://github.com/views-platform/views-stepshifter/blob/main/README.md), a model is understood as all code and all artifacts necessary to generate a comprehensive 36 month forecast for the specified target.
6. Note that, two models, identical in all other aspects, will be deemed distinct if varying post-processing techniques are applied to their generated predictions. For instance, if one model's predictions undergo calibration or normalization while the other's do not. Similarly, two models identical in all aspects are considered distinct if they utilize different input features (querysets).

---
<a name="time-partitioning"></a>

## Time partitioning

VIEWS models all currently use the same time partitioning model to divide the time-axis of the input dataset up into three segments. The boundaries of these partitions are currently fixed in the `views-pipeline-core` package, but they will be made user-configurable in the future. Partitions are labelled by their VIEWS `month_id`, where month 001 is January 1980, month 121 is January 1990, and so on. The partitions are as follows:
- **calibration**: training interval: 121 - 396, test interval: 397 - 444
- **validation**: training interval: 121 - 444, test interval: 445 - 456
- **forecasting**: training interval: 121 - (current VIEWS month -2)

---

## Model Naming Conventions

The models belonging to the VIEWS pipeline follow a predetermined naming standard. Models no longer carry descriptive titles (e.g., transform_log_clf_name_LGBMClassifier_reg_name_LGBMRegressor). Although such titles provided some information about the models, as  models are developed over time, this type of naming could cause confusion and ultimately small differences could not be communicated properly through the model titles. Instead, we rely on the metadata of the model for model specifications and being able to substantively differentiate them between each other.

Additionally, the new naming convention for models in the pipeline takes the form of adjective_noun, adding more models alphabetically. For example, the first model to be added can be named amazing_apple, the second model bad_bunny, etc. This is a popular practice, and Weights & Biases implements this naming convention automatically.

---

## Creating New Models 
<a name="creating-new-models"></a>

The views-models repository contains the tools for creating new models, as well as creating new model ensembles. All of the necessary components are found in the `tools/scaffold/build_model_scaffold.py` and `tools/scaffold/build_ensemble_scaffold.py` files. The goal of this part of the VIEWS pipeline is the ability to simply create models which have the right structure and fit into the VIEWS directory structure. This makes the models uniform, consistent, and allows for easier replicability. 

As with other parts of the VIEWS pipeline, we aim to make interactions with our pipeline as simple and straightforward as possible. In the context of the views-models, when creating a new model or ensemble, the user is closely guided through the steps which are needed, in an intuitive manner. This allows for the model creation processes to be consistent no matter how experienced the creator is. After providing a name for the model or ensemble, guided to be in the form adjective_noun, the user can specify the desired model algorithm and the model architecture package. Currently, only [stepshift models](https://github.com/views-platform/views-stepshifter/blob/main/README.md) are supported, however, we work on expanding the list of supported algorithms and model architectures. Then, the scaffold builders create all of the model files and model directories, uniformly structured. This instantly removes possibilities of error, increases efficiency and effectiveness as it decreases manual inputs of code. Finally, this allows all of our users, no matter their level of proficiency, to seamlessly interact with out pipeline in no time.  

To run the model scaffold builder, execute

`python tools/scaffold/build_model_scaffold.py`

You will be asked to enter a name for your model in lowercase `adjective_noun` form. If the scaffolder is happy with your proposed model name, it will create a new directory with your chosen name. This directory in turn contains the scripts and folders needed to run your model and store intermediate data belonging to it. It is the responsibility of the model creator to make changes to the newly created scripts where appropriate - see below for further information on which scripts need to be updated. The scripts created are as follows (see further down for a description of the filesystem):

---
<a name="model-scripts"></a>
# MODEL SCRIPTS

## `README.md`
It is the responsibility of the model creator to write a README file for their model. This should give a concise, human-readable description of the model:
- what it forecasts
- what algorithm(s) it uses
- what hyperparameters it relies on and whether these have been or can be optimised (e.g. in a sweep)
- a brief description of what input data it requires
- how it is or should be evaluated 
- (preferably) some notes on performance.

## `run.sh`
This shell script is the principal means by which a model should be run (e.g. by executing `source run.sh arg1 arg2...` at a terminal prompt - see 'Running a single model' below). You probably will not need to modify it, but it is important to understand what it is for.

The VIEWS platform is designed to support models of arbitrary form. A model may need to import many external libraries or modules and the set of modules required by one model are quite likely to be incompatible with those of another (a 'dependency conflict').

The VIEWS platform solves this problem by building a custom Python **environment** for every model. A Python environment is an isolated sandbox into which a particular set of modules can be installed, and it does not matter if the modules installed on one environment are incompatible with those installed in another. Code execution can be quickly switched between environments, so that models with dependency conflicts can be easily executed in series.

The `run.sh` script first builds the environment required to run a model, specified in the `requirements.txt` file - see below), and then executes the model inside that environment by passing its `main.py` file (see below) to the Python interpreter.

## `requirements.txt`
The purpose of this file is to specify which modules (probably including their versions or an acceptable range thereof) need to be installed in the model-specific environment built by `run.sh`. 

**It is the model creator's responsibility to ensure that this file is correctly populated.** Only modules named in this file (and their dependencies) will be installed in the model env. If your model needs `numpy` and it is not installed by any other dependencies, it needs to be specified here.

It is strongly advised to specify a range of acceptable versions for each installed module using the standard notation, e.g. `views-stepshifter>=1.0.0,<2.0.0`.

## `main.py`
Once the `run.sh` script has created the model's environment, it activates the environment and executes the `main.py` file inside it. The `main.py` has several tasks:

- it uses the `ModelPathManager` from `views-pipeline-core` to establish where it is on the host machine's filesystem so that other scripts and modules can be found by the Python interpreter at runtime
- it logs into `weights-and-biases` - all runs executed in the VIEWS platform are automatically externally logged to the weights-and-biases web platform - URLs are printed to the terminal during model/ensemble execution, which will take users to webpages showing live logging and analytics
- it parses command line arguments (forwarded by `run.sh`) which specify whether the model is to be trained, whether a sweep over hyperparameters should be performed, etc.
- it then calls the relevant `Manager` from `views-pipeline-core` which superintends the execution of the model. Every class of models has its own custom manager (e.g. `StepShifterManager` looks after stepshifted regression models). **If you are introducing a new class of model to VIEWS, you will need to create a new Manager class for it.**

**Make sure to import your model manager class and include it in the appropriate sections in the `main.py` script!**   

<a name="model-filesystems"></a>

# MODEL FILESYSTEM
As well as understanding the function of the model scripts, users and developers need to have a grasp of the structure of the model filesystem. A description of each of the directories follows below:

## `artifacts`
An artifact is the result of training a model on a particular set of input data. For example, if a regression model is trained on a particular input, the set of regression coefficients calculated by the model constitute an artifact. The artifact can be stored and later used to make predictions from new data without needing to train the model again. 

The VIEWS platform allows users to store model-specific artifacts locally. If you have never trained a particular model, this directory will be empty.

## `configs`
This directory contains Python scripts used to control model configuration. **Model creators need to ensure that all settings needed to configure a model or a model sweep are contained in these scripts and correctly defined.**

- `config_deployment.py`: The VIEWS platform is designed to permit new models to be tested and developed in parallel with established (i.e. 'production') models which are used to generate our publicly-disseminated forecasts. A model's `deployment_status` must be specified in this script and must be one of `shadow`, `deployed`, `baseline`, or `deprecated` to indicate its stage of development. An under-development model which should not be used in production should have status `shadow`. Fully developed production models have status `deployed`. Simple models used as references or yardsticks are `baseline`. If a production model is superseded, it can be retired from the production system by setting its status to `deprecated`. **A model MUST NOT be given `deployed` status without discussion with the modelling team**.


- `config_hyperparameters.py`: Most models will rely on algorithms for which hyperparameters need to be specified (even if invisibly by default). This script contains dictionary specifying any required model-specific hyperparameters to be read at runtime.


- `config_meta.py`: This script specifies the most basic model parameters, e.g. the model's name, the name of its forecasting algorithm, the dependent variable it forecasts, the name of its input data queryset (see below), its creator. **This dictionary must be populated correctly**, since it controls important aspects of model execution further down the pipeline. 


- `config_queryset.py`: Most VIEWS models are anticipated to need to fetch data from the central VIEWS database via the `viewser` client. This is done by specifying a `queryset`. A queryset is a representation of a data table. It consists of a name, a target level-of-analysis (into which all data is automatically transformed) and one or more Columns. A Column, in turn, has a name, a source level-of-analysis, the name of a raw feature from the VIEWS database and zero or more transforms from the `views-transformation-library`. The queryset is passed via the viewser client to a server which executes the required database fetches and transformations and returns the dataset as a single dataframe (or, in the future, a tensor). The `config_queryset.py` specifies the queryset, and **it is the model creator's responsibility to ensure that the specification is correct**.


- `config_sweep.py`: During model development, developers will often wish to perform sweeps over ranges of model hyperparameters for optimisation purposes (hyperparameter tuning). This script allows such sweeps to be configured, specifying which parameters ranges are to explored and what is to be optimised.


## `data`
The VIEWS platform allows local storage of data for convenience, both raw data (i.e. input data from a queryset fetch) and generated data (e.g. forecasts), all of which is stored in this directory.

## `logs`
The platform produces detailed logs during execution which are printed to the terminal, exported to weights-and-biases and also saved locally in this directory.

## `notebooks`
While the use of Jupyter notebooks is generally discouraged on the grounds of stability and git interoperability, this directory is provided for those who wish to use them during model development. 

Users should note, however, that **Jupyter notebooks MUST NOT be used to run production models**.

## `reports`
Convenience directory where figures, papers or slides relating to particular models can be stored.

## `wandb`
Logs shipped to weights-and-biases are also stored locally here for convenience

---

<a name="running-a-single-model"></a>

## Running a single model
A model is run by executing the `run.sh` script in its root directory, which checks to see if an appropriate environment for the model exists, creates one if not, activates the environment, and executes the model's `main.py` inside it. The model can be run by executing the `main.py` directly, but it is then up to the user to ensure that the model's environment is correctly built and activated. However, if the environment is setup once e.g. by executing the `run.sh` script, it can be activated at a later point in time and the model can be run by by executing the `main.py` directly.

The `run.sh` and `main.py` both require command line arguments to control their behaviour (command line arguments submitted to `run.sh` are simply passed on to `main.py`). A description of these arguments follows:

- `-r` or `--run_type` followed by one of [`calibration`, `validation`, `forecasting`]:  choose the run type


- `-s` or `--sweep`: perform a sweep run (run type must be `calibration`)


- `-t` or `--train`: flag indicating whether a new model artifact should be trained


- `-e` or `--evaluate`: flag indicating if model should be evaluated


- `-f` or `--forecast`: flag indicating if forecasts are to be generated


- `-a` or `--artifact_name`: flag allowing the name of the artifact to be evaluated to be supplied


- `-sa` or `--saved`: flag to indicate that saved data/artifacts should be used


- `-o` or `--override_month`: flag allowing one to specify a month other than the most recent month with data from which to forecast
  

- `-dd` or `--drift_self_test`: flag enabling drift-detection self_test at data-fetch


- `-et` or `--eval_type`: flag allowing type of evaluation to be performed to be specified

Consequently, in order to train and evaluate a model it is either possible to execute `python main.py -run_type calibration -t -e` or `run.sh -run_type calibration -t -e`. The first command runs the script directly, while the second one also handles environment setup before execution. Of course, these commands can be used to run already existing models (see the [Catalogs](#Catalogs) for a list of already existing models). Consult the [Glossary](https://github.com/views-platform/docs/blob/main/FAQ%20%26%20Glossary/glossary.md) and the Model Documentation Series to learn more about different run types.

# Ensembles

An ensemble is a combination of models which has greater predictive power than any of the models does singly. Ensemble forecasts can be simple averages over the forecasts of their constituent models, or a more sophisticated weighted average, where the weights are computed by optimising the ensemble's predictive performance over a specially reserved data partition, using, for example, an evolutionary algorithm. (The latter is not yet implemented in the VIEWS pipeline).

It is also possible to reconcile one ensemble with another (usually at a different spatial resolution) to, for example, force the forecasts to agree over well-defined spatial areas such as countries. The VIEWS pipeline allows point priogrid-level forecasts to be reconciled with country-level forecasts on a month-by-month basis (accounting for the fact that countries change size or appear/disappear altogether).

<a name="creating-a-new-ensemble"></a>

## Creating New Ensembles 

The procedure for creating a new ensemble is much the same as that for creating a new model. The `tools/scaffold/build_ensemble_scaffold.py` script is run and, once it is supplied with a legal lower case `adjective_noun` ensemble name, a filesystem very similar to that created for a new model is built. As in the case of creating new models, make sure to update the appropriate model scripts (indicated below).

<a name="ensemble-scripts"></a>

# ENSEMBLE SCRIPTS

## `README.md`
It is the responsibility of the ensemble creator to write a README file for their ensemble. This should give a concise, human-readable description of the ensemble:
- what it forecasts
- which constituent models it ensembles over
- how the ensembling is done
- how it is or should be evaluated 
- (preferably) some notes on performance.

## `run.sh`
This shell script is the principal means by which an ensemble should be run (e.g. by executing `source run.sh arg1 arg2...` at a terminal prompt - see 'Running an ensemble' below). You probably will not need to modify it, but it is important to understand what it is for.

The `run.sh` script first builds the environment required to run the ensemble, specified in the `requirements.txt` file - see below), and then executes the ensemble inside that environment by passing its `main.py` file (see below) to the Python interpreter.

## `requirements.txt`
The purpose of this file is to specify which modules (probably including their versions or an acceptable range thereof) need to be installed in the ensemble-specific environment built by `run.sh`.

**It is the ensemble creator's responsibility to ensure that this file is correctly populated.** Only modules named in this file (and their dependencies) will be installed in the ensemble env. If your ensemble needs `numpy` and it is not installed by any other dependencies, it needs to be specified here.

It is strongly advised to specify a range of acceptable versions for each installed module using the standard notation, e.g. `views-stepshifter>=1.0.0,<2.0.0`.

## `main.py`
Once the `run.sh` script has created the ensemble's environment, it activates the environment and executes the `main.py` file inside it. The `main.py` has several tasks:

- it uses the `EnsemblePathManager` from `views-pipeline-core` to establish where it is on the host machine's filesystem so that other scripts and modules can be found by the Python interpreter at runtime
- it logs into `weights-and-biases` - all runs executed in the VIEWS platform are automatically externally logged to the weights-and-biases web platform - URLs are printed to the terminal during model/ensemble execution, which will take users to webpages showing live logging and analytics
- it parses command line arguments (forwarded by `run.sh`) which specify whether the ensemble is to be trained, whether forecasts are to be generated, etc.
- it then calls the `EnsembleManager` from `views-pipeline-core` which superintends the execution of the ensemble.

<a name="ensemble-filesystem"></a>

# ENSEMBLE FILESYSTEM
As well as understanding the function of the ensemble scripts, users and developers need to have a grasp of the structure of the ensemble filesystem. A description of each of the directories follows below:

## `artifacts`
Currently not used by ensembles.

## `configs`
This directory contains Python scripts used to control model configuration. **Model creators need to ensure that all settings needed to configure a model or a model sweep are contained in these scripts and correctly defined.**

- `config_deployment.py`: An ensemble's `deployment_status` must be specified in this script and must be one of `shadow`, `deployed`, `baseline`, or `deprecated` to indicate its stage of development. An under-development ensemble which should not be used in production should have status `shadow`. Fully developed production ensembles have status `deployed`. Ensembles used as references or yardsticks are `baseline`. If a production ensemble is superseded, it can be retired from the production system by setting its status to `deprecated`. **An ensemble MUST NOT be given `deployed` status without discussion with the modelling team**.


- `config_hyperparameters.py`: This is currently only used to configure the number of timesteps forward the ensemble forecasts


- `config_meta.py`: This script specifies the most basic ensemble parameters, e.g. the ensemble's name, the models it ensembles over, the dependent variable it forecasts, the aggregation scheme used to perform the ensembling, which reconciliation algorithm it to be applied, which other ensemble it should be reconciled with, and its creator. **This dictionary must be populated correctly**, since it controls important aspects of ensemble execution further down the pipeline.


## `data`
The VIEWS platform allows local storage of data for convenience, in this directory.

## `logs`
The platform produces detailed logs during execution which are printed to the terminal, exported to weights-and-biases and also saved locally in this directory.

## `reports`
Convenience directory where figures, papers or slides relating to particular models can be stored.

## `wandb`
Logs shipped to weights-and-biases are also stored locally here for convenience

<a name="running-an-ensemble"></a>

## Running an ensemble
An ensemble is run by executing the `run.sh` script in its root directory, which checks to see if an appropriate environment for the ensemble exists, creates one if not, activates the environment, and executes the ensemble's `main.py` inside it. The ensemble can be run by executing the `main.py` directly, but it is then up to the user to ensure that the model's environment is correctly built and activated.

The `run.sh` and `main.py` both require command line arguments to control their behaviour (command line arguments submitted to `run.sh` are simply passed on to `main.py`). A description of these arguments follows:

- `-r` or `--run_type` followed by one of [`calibration`, `validation`, `forecasting`]:  choose the run type


- `-t` or `--train`: flag indicating whether new model artifacts should be trained


- `-e` or `--evaluate`: flag indicating if the ensemble should be evaluated


- `-f` or `--forecast`: flag indicating if forecasts are to be generated


- `-sa` or `--saved`: flag to indicate that saved data/artifacts should be used


- `-o` or `--override_month`: flag allowing one to specify a month other than the most recent month with data from which to forecast
  

- `-dd` or `--drift_self_test`: flag enabling drift-detection self_test at data-fetch


- `-et` or `--eval_type`: flag allowing type of evaluation to be performed to be specified

Consequently, in order to train a model and generate predictions, execute either `python main.py -t --run_type forecasting -f` or `run.sh -t --run_type forecasting -f`. Of course, these commands can be used to run already existing ensembles (see the [Catalogs](#Catalogs) for a list of already existing ensembles). Consult the [Glossary](https://github.com/views-platform/docs/blob/main/FAQ%20%26%20Glossary/glossary.md) and the Model Documentation Series to learn more about different run types.


## Implementing Model Architectures

As of now, the only implemented model architecture is the [stepshifter model](https://github.com/views-platform/views-stepshifter/blob/main/README.md). Experienced users have the possibility to develop their own model architecture including their own model class manager. Head over to [views-pipeline-core](https://github.com/views-platform/views-pipeline-core) for further information on the model class manager and on how to develop new model architectures. 


## Integration Testing
<a name="integration-testing"></a>

The repository includes an integration test runner that verifies models haven't been broken by changes in this repo or in upstream/downstream packages. It trains and evaluates every runnable model end-to-end on calibration and validation partitions, running them sequentially in a single shared conda environment, and produces a summary table of `PASS`/`FAIL`/`TIMEOUT`/`DEPRECATED`/`ABORTED` results with per-model logs. A single `Ctrl-C` cleanly aborts a run and prints a partial summary.

```bash
# Run all models (calibration + validation)
bash run_integration_tests.sh

# Run only country-month models
bash run_integration_tests.sh --level cm

# Run only baseline models
bash run_integration_tests.sh --library baseline

# Run specific models with a custom timeout
bash run_integration_tests.sh --models "counting_stars bad_blood" --timeout 3600
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models "m1 m2"` | all models | Run only these models |
| `--level` `cm` or `pgm` | no filter | Run only models at this level of analysis |
| `--library NAME` | no filter | Run only models using this library (baseline/stepshifter/r2darts2/hydranet) |
| `--exclude "m1 m2"` | `"purple_alien"` | Skip these models (replaces the default, does not append) |
| `--partitions "p1 p2"` | `"calibration validation"` | Partitions to test |
| `--timeout SECONDS` | `1800` | Max wall-clock time per model run |
| `--env NAME` | `views_pipeline` | Conda environment to activate |

Logs are written to `logs/integration_test_<timestamp>/` with a `summary.log` and per-model logs under `{partition}/{model}.log`.

For the full guide — including how model discovery works, how to read failure logs, and important caveats — see [docs/run_integration_tests.md](docs/run_integration_tests.md).

---

## Implemented Models

In addition to the possibility of easily creating new models and ensembles, in order to maintain an organized and structured overview over all of the implemented models, the views-models repository also contains model catalogs containing all of the information about individual models. This information is collected from the metadata of each model and entails:
1. Model name 
2. Model algorithm
3. The prediction target
4. Input features/Queryset
5. The non-default hyperparameters
6. Forecasting type
7. Implementation status
8. Implementation date 
9. The creator of the model

The catalogs are automatically filled out and updated, through a GitHub action, with every new model or ensemble which is created.  

## UpdateViewser Class

Currently, the data returned from viewser returns 0 from month id 543 onward. In order to still produce valid predictions, an emergency update solution has been implemented. For our most important datasources - Acled and UCDP - updates are fetched and then integrated into the dataframe returned from viewser. For this emergency solution to work note:

- querysets need to be modified such that they return the raw data for all of the transformed variables. For all transformed variables, the raw base variables have to be added to the queryset and while it is up to the user to choose a name for them, they HAVE to start with "raw_". If the queryset does not contain variables staring with "raw_" and you use the -u flag, an exception is thrown. example queryset line: .with_column(Column("raw_ged_sb_dep", from_loa="country_month", from_column="ged_sb_best_sum_nokgi"))
- UCDP & Acled updates are stored in a folder of your choice. For the update dataframes, contact Sonja. 
- As a user you need to: 1. Place the update dataframes in a folder of your choice. 2. In the dotenv file in views-models add the path to your update dataframes (cm_path='path/to/file' & pgm_path='path/to/file') and month_to_update as a list of ints e.g.: month_to_update= [543,544,545,546].
- Execute the pipeline with your usual flags and add -u or --update_viewser. The default is to NOT update the viewser dataframe if you just use the usual flags. 

---
## Catalogs

The catalogs for all of the existing VIEWS models can be found below. The models catalogs are separated based on the models' level of analysis - country-month models and PRIO-GRID-month models, with the ensamble catalog all the way at the bottom. All of the information about algorithms, input features, hyperparameters and other model specification are included.

### Country-Month Model Catalog

<!-- CM_TABLE_START -->
| Model Name | Algorithm | Targets | Input Features | Hyperparameters | Implementation Status | Implementation Date | Author |
| ---------- | --------- | ------- | -------------- | --------------- | --------------------- | ------------------- | ------ |
| [adolecent_slob](https://github.com/views-platform/views-models/blob/development/models/adolecent_slob) | TCNModel | lr_ged_sb_dep | - [adolecent_slob_features](https://github.com/views-platform/views-models/blob/development/models/adolecent_slob/configs/config_queryset.py) | - [hyperparameters adolecent_slob](https://github.com/views-platform/views-models/blob/development/models/adolecent_slob/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [average_cmbaseline](https://github.com/views-platform/views-models/blob/development/models/average_cmbaseline) | AverageModel | lr_ged_sb | N/A | - [hyperparameters average_cmbaseline](https://github.com/views-platform/views-models/blob/development/models/average_cmbaseline/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Sonja |
| [bad_romance](https://github.com/views-platform/views-models/blob/development/models/bad_romance) | TiDEModel | lr_ged_sb | - [bad_romance_features](https://github.com/views-platform/views-models/blob/development/models/bad_romance/configs/config_queryset.py) | - [hyperparameters bad_romance](https://github.com/views-platform/views-models/blob/development/models/bad_romance/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Dylan |
| [bittersweet_symphony](https://github.com/views-platform/views-models/blob/development/models/bittersweet_symphony) | XGBRegressor | lr_ged_sb | - [bittersweet_symphony_features](https://github.com/views-platform/views-models/blob/development/models/bittersweet_symphony/configs/config_queryset.py) | - [hyperparameters bittersweet_symphony](https://github.com/views-platform/views-models/blob/development/models/bittersweet_symphony/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Marina |
| [bouncy_organ](https://github.com/views-platform/views-models/blob/development/models/bouncy_organ) | TSMixerModel | lr_ged_sb_dep | - [bouncy_organ_features](https://github.com/views-platform/views-models/blob/development/models/bouncy_organ/configs/config_queryset.py) | - [hyperparameters bouncy_organ](https://github.com/views-platform/views-models/blob/development/models/bouncy_organ/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [brown_cheese](https://github.com/views-platform/views-models/blob/development/models/brown_cheese) | XGBRFRegressor | lr_ged_sb | - [brown_cheese_features](https://github.com/views-platform/views-models/blob/development/models/brown_cheese/configs/config_queryset.py) | - [hyperparameters brown_cheese](https://github.com/views-platform/views-models/blob/development/models/brown_cheese/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [car_radio](https://github.com/views-platform/views-models/blob/development/models/car_radio) | XGBRegressor | lr_ged_sb | - [car_radio_features](https://github.com/views-platform/views-models/blob/development/models/car_radio/configs/config_queryset.py) | - [hyperparameters car_radio](https://github.com/views-platform/views-models/blob/development/models/car_radio/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [cheap_thrills](https://github.com/views-platform/views-models/blob/development/models/cheap_thrills) | ShurfModel | lr_sb_best | - [cheap_thrills_features](https://github.com/views-platform/views-models/blob/development/models/cheap_thrills/configs/config_queryset.py) | - [hyperparameters cheap_thrills](https://github.com/views-platform/views-models/blob/development/models/cheap_thrills/configs/config_hyperparameters.py) | shadow | 2025-03-19 | Håvard |
| [cold_heart](https://github.com/views-platform/views-models/blob/development/models/cold_heart) | NBEATSModel | lr_ged_sb | - [cold_heart_features](https://github.com/views-platform/views-models/blob/development/models/cold_heart/configs/config_queryset.py) | - [hyperparameters cold_heart](https://github.com/views-platform/views-models/blob/development/models/cold_heart/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Dylan |
| [counting_stars](https://github.com/views-platform/views-models/blob/development/models/counting_stars) | XGBRegressor | lr_ged_sb | - [counting_stars_features](https://github.com/views-platform/views-models/blob/development/models/counting_stars/configs/config_queryset.py) | - [hyperparameters counting_stars](https://github.com/views-platform/views-models/blob/development/models/counting_stars/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [dancing_queen](https://github.com/views-platform/views-models/blob/development/models/dancing_queen) | BlockRNNModel | lr_ged_sb | - [dancing_queen_features](https://github.com/views-platform/views-models/blob/development/models/dancing_queen/configs/config_queryset.py) | - [hyperparameters dancing_queen](https://github.com/views-platform/views-models/blob/development/models/dancing_queen/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Dylan |
| [demon_days](https://github.com/views-platform/views-models/blob/development/models/demon_days) | XGBRFRegressor | lr_ged_sb | - [demon_days_features](https://github.com/views-platform/views-models/blob/development/models/demon_days/configs/config_queryset.py) | - [hyperparameters demon_days](https://github.com/views-platform/views-models/blob/development/models/demon_days/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Marina |
| [elastic_heart](https://github.com/views-platform/views-models/blob/development/models/elastic_heart) | TSMixerModel | lr_ged_sb | - [elastic_heart_features](https://github.com/views-platform/views-models/blob/development/models/elastic_heart/configs/config_queryset.py) | - [hyperparameters elastic_heart](https://github.com/views-platform/views-models/blob/development/models/elastic_heart/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Dylan |
| [electric_relaxation](https://github.com/views-platform/views-models/blob/development/models/electric_relaxation) | RandomForestRegressor | lr_ged_sb | - [electric_relaxation_features](https://github.com/views-platform/views-models/blob/development/models/electric_relaxation/configs/config_queryset.py) | - [hyperparameters electric_relaxation](https://github.com/views-platform/views-models/blob/development/models/electric_relaxation/configs/config_hyperparameters.py) | deprecated | 2024-11-22 | Sara |
| [emerging_principles](https://github.com/views-platform/views-models/blob/development/models/emerging_principles) | NBEATSModel | lr_ged_sb | - [emerging_principles_features](https://github.com/views-platform/views-models/blob/development/models/emerging_principles/configs/config_queryset.py) | - [hyperparameters emerging_principles](https://github.com/views-platform/views-models/blob/development/models/emerging_principles/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [fancy_feline](https://github.com/views-platform/views-models/blob/development/models/fancy_feline) | TiDEModel | lr_ged_sb_dep | - [fancy_feline_features](https://github.com/views-platform/views-models/blob/development/models/fancy_feline/configs/config_queryset.py) | - [hyperparameters fancy_feline](https://github.com/views-platform/views-models/blob/development/models/fancy_feline/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [fast_car](https://github.com/views-platform/views-models/blob/development/models/fast_car) | HurdleModel | lr_ged_sb | - [fast_car_features](https://github.com/views-platform/views-models/blob/development/models/fast_car/configs/config_queryset.py) | - [hyperparameters fast_car](https://github.com/views-platform/views-models/blob/development/models/fast_car/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [fluorescent_adolescent](https://github.com/views-platform/views-models/blob/development/models/fluorescent_adolescent) | HurdleModel | lr_ged_sb | - [fluorescent_adolescent_features](https://github.com/views-platform/views-models/blob/development/models/fluorescent_adolescent/configs/config_queryset.py) | - [hyperparameters fluorescent_adolescent](https://github.com/views-platform/views-models/blob/development/models/fluorescent_adolescent/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Marina |
| [fourtieth_symphony](https://github.com/views-platform/views-models/blob/development/models/fourtieth_symphony) | ShurfModel | lr_sb_best | - [fourtieth_symphony_features](https://github.com/views-platform/views-models/blob/development/models/fourtieth_symphony/configs/config_queryset.py) | - [hyperparameters fourtieth_symphony](https://github.com/views-platform/views-models/blob/development/models/fourtieth_symphony/configs/config_hyperparameters.py) | shadow | 2025-03-19 | Håvard |
| [free_fallin](https://github.com/views-platform/views-models/blob/development/models/free_fallin) | TSMixerModel | lr_ged_sb | - [free_fallin_features](https://github.com/views-platform/views-models/blob/development/models/free_fallin/configs/config_queryset.py) | - [hyperparameters free_fallin](https://github.com/views-platform/views-models/blob/development/models/free_fallin/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Dylan |
| [good_life](https://github.com/views-platform/views-models/blob/development/models/good_life) | TransformerModel | lr_ged_sb | - [good_life_features](https://github.com/views-platform/views-models/blob/development/models/good_life/configs/config_queryset.py) | - [hyperparameters good_life](https://github.com/views-platform/views-models/blob/development/models/good_life/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Dylan |
| [good_riddance](https://github.com/views-platform/views-models/blob/development/models/good_riddance) | XGBRFRegressor | lr_ged_sb | - [good_riddance_features](https://github.com/views-platform/views-models/blob/development/models/good_riddance/configs/config_queryset.py) | - [hyperparameters good_riddance](https://github.com/views-platform/views-models/blob/development/models/good_riddance/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Marina |
| [green_ranger](https://github.com/views-platform/views-models/blob/development/models/green_ranger) | MixtureBaseline | lr_ns_best | - [green_ranger_features](https://github.com/views-platform/views-models/blob/development/models/green_ranger/configs/config_queryset.py) | - [hyperparameters green_ranger](https://github.com/views-platform/views-models/blob/development/models/green_ranger/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [green_squirrel](https://github.com/views-platform/views-models/blob/development/models/green_squirrel) | HurdleModel | lr_ged_sb | - [green_squirrel_features](https://github.com/views-platform/views-models/blob/development/models/green_squirrel/configs/config_queryset.py) | - [hyperparameters green_squirrel](https://github.com/views-platform/views-models/blob/development/models/green_squirrel/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [heat_waves](https://github.com/views-platform/views-models/blob/development/models/heat_waves) | TFTModel | lr_ged_sb | - [heat_waves_features](https://github.com/views-platform/views-models/blob/development/models/heat_waves/configs/config_queryset.py) | - [hyperparameters heat_waves](https://github.com/views-platform/views-models/blob/development/models/heat_waves/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Dylan |
| [heavy_rotation](https://github.com/views-platform/views-models/blob/development/models/heavy_rotation) | XGBRFRegressor | lr_ged_sb | - [heavy_rotation_features](https://github.com/views-platform/views-models/blob/development/models/heavy_rotation/configs/config_queryset.py) | - [hyperparameters heavy_rotation](https://github.com/views-platform/views-models/blob/development/models/heavy_rotation/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [high_hopes](https://github.com/views-platform/views-models/blob/development/models/high_hopes) | HurdleModel | lr_ged_sb | - [high_hopes_features](https://github.com/views-platform/views-models/blob/development/models/high_hopes/configs/config_queryset.py) | - [hyperparameters high_hopes](https://github.com/views-platform/views-models/blob/development/models/high_hopes/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [hot_stream](https://github.com/views-platform/views-models/blob/development/models/hot_stream) | TFTModel | lr_ged_sb_dep | - [hot_stream_features](https://github.com/views-platform/views-models/blob/development/models/hot_stream/configs/config_queryset.py) | - [hyperparameters hot_stream](https://github.com/views-platform/views-models/blob/development/models/hot_stream/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [little_lies](https://github.com/views-platform/views-models/blob/development/models/little_lies) | HurdleModel | lr_ged_sb | - [little_lies_features](https://github.com/views-platform/views-models/blob/development/models/little_lies/configs/config_queryset.py) | - [hyperparameters little_lies](https://github.com/views-platform/views-models/blob/development/models/little_lies/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Marina |
| [locf_cmbaseline](https://github.com/views-platform/views-models/blob/development/models/locf_cmbaseline) | LocfModel | lr_ged_sb | N/A | - [hyperparameters locf_cmbaseline](https://github.com/views-platform/views-models/blob/development/models/locf_cmbaseline/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Sonja |
| [lovely_creature](https://github.com/views-platform/views-models/blob/development/models/lovely_creature) | ShurfModel | lr_sb_best | - [lovely_creature_features](https://github.com/views-platform/views-models/blob/development/models/lovely_creature/configs/config_queryset.py) | - [hyperparameters lovely_creature](https://github.com/views-platform/views-models/blob/development/models/lovely_creature/configs/config_hyperparameters.py) | shadow | 2025-03-19 | Håvard |
| [national_anthem](https://github.com/views-platform/views-models/blob/development/models/national_anthem) | XGBRFRegressor | lr_ged_sb | - [national_anthem_features](https://github.com/views-platform/views-models/blob/development/models/national_anthem/configs/config_queryset.py) | - [hyperparameters national_anthem](https://github.com/views-platform/views-models/blob/development/models/national_anthem/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [new_rules](https://github.com/views-platform/views-models/blob/development/models/new_rules) | NBEATSModel | lr_ged_sb | - [new_rules_features](https://github.com/views-platform/views-models/blob/development/models/new_rules/configs/config_queryset.py) | - [hyperparameters new_rules](https://github.com/views-platform/views-models/blob/development/models/new_rules/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Dylan |
| [novel_heuristics](https://github.com/views-platform/views-models/blob/development/models/novel_heuristics) | NBEATSModel | lr_ged_sb | - [novel_heuristics_features](https://github.com/views-platform/views-models/blob/development/models/novel_heuristics/configs/config_queryset.py) | - [hyperparameters novel_heuristics](https://github.com/views-platform/views-models/blob/development/models/novel_heuristics/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [ominous_ox](https://github.com/views-platform/views-models/blob/development/models/ominous_ox) | XGBRFRegressor | lr_ged_sb | - [ominous_ox_features](https://github.com/views-platform/views-models/blob/development/models/ominous_ox/configs/config_queryset.py) | - [hyperparameters ominous_ox](https://github.com/views-platform/views-models/blob/development/models/ominous_ox/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [party_princess](https://github.com/views-platform/views-models/blob/development/models/party_princess) | BlockRNNModel | lr_ged_sb_dep | - [party_princess_features](https://github.com/views-platform/views-models/blob/development/models/party_princess/configs/config_queryset.py) | - [hyperparameters party_princess](https://github.com/views-platform/views-models/blob/development/models/party_princess/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [plastic_beach](https://github.com/views-platform/views-models/blob/development/models/plastic_beach) | XGBRFRegressor | lr_ged_sb | - [plastic_beach_features](https://github.com/views-platform/views-models/blob/development/models/plastic_beach/configs/config_queryset.py) | - [hyperparameters plastic_beach](https://github.com/views-platform/views-models/blob/development/models/plastic_beach/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Marina |
| [popular_monster](https://github.com/views-platform/views-models/blob/development/models/popular_monster) | XGBRFRegressor | lr_ged_sb | - [popular_monster_features](https://github.com/views-platform/views-models/blob/development/models/popular_monster/configs/config_queryset.py) | - [hyperparameters popular_monster](https://github.com/views-platform/views-models/blob/development/models/popular_monster/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [preliminary_directives](https://github.com/views-platform/views-models/blob/development/models/preliminary_directives) | NBEATSModel | lr_ged_sb | - [preliminary_directives_features](https://github.com/views-platform/views-models/blob/development/models/preliminary_directives/configs/config_queryset.py) | - [hyperparameters preliminary_directives](https://github.com/views-platform/views-models/blob/development/models/preliminary_directives/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [purple_haze](https://github.com/views-platform/views-models/blob/development/models/purple_haze) | ShurfModel | lr_sb_best | - [purple_haze_features](https://github.com/views-platform/views-models/blob/development/models/purple_haze/configs/config_queryset.py) | - [hyperparameters purple_haze](https://github.com/views-platform/views-models/blob/development/models/purple_haze/configs/config_hyperparameters.py) | shadow | 2025-03-19 | Håvard |
| [red_ranger](https://github.com/views-platform/views-models/blob/development/models/red_ranger) | MixtureBaseline | lr_ged_sb | - [red_ranger_features](https://github.com/views-platform/views-models/blob/development/models/red_ranger/configs/config_queryset.py) | - [hyperparameters red_ranger](https://github.com/views-platform/views-models/blob/development/models/red_ranger/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [revolving_door](https://github.com/views-platform/views-models/blob/development/models/revolving_door) | NHiTSModel | lr_ged_sb | - [revolving_door_features](https://github.com/views-platform/views-models/blob/development/models/revolving_door/configs/config_queryset.py) | - [hyperparameters revolving_door](https://github.com/views-platform/views-models/blob/development/models/revolving_door/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Dylan |
| [shining_codex](https://github.com/views-platform/views-models/blob/development/models/shining_codex) | NBEATSModel | lr_ged_sb | - [shining_codex_features](https://github.com/views-platform/views-models/blob/development/models/shining_codex/configs/config_queryset.py) | - [hyperparameters shining_codex](https://github.com/views-platform/views-models/blob/development/models/shining_codex/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [smol_cat](https://github.com/views-platform/views-models/blob/development/models/smol_cat) | TiDEModel | lr_ged_sb | - [smol_cat_features](https://github.com/views-platform/views-models/blob/development/models/smol_cat/configs/config_queryset.py) | - [hyperparameters smol_cat](https://github.com/views-platform/views-models/blob/development/models/smol_cat/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Dylan |
| [teen_spirit](https://github.com/views-platform/views-models/blob/development/models/teen_spirit) | XGBRFRegressor | lr_ged_sb | - [teen_spirit_features](https://github.com/views-platform/views-models/blob/development/models/teen_spirit/configs/config_queryset.py) | - [hyperparameters teen_spirit](https://github.com/views-platform/views-models/blob/development/models/teen_spirit/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Marina |
| [twin_flame](https://github.com/views-platform/views-models/blob/development/models/twin_flame) | HurdleModel | lr_ged_sb | - [twin_flame_features](https://github.com/views-platform/views-models/blob/development/models/twin_flame/configs/config_queryset.py) | - [hyperparameters twin_flame](https://github.com/views-platform/views-models/blob/development/models/twin_flame/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Borbála |
| [wild_rose](https://github.com/views-platform/views-models/blob/development/models/wild_rose) | ShurfModel | lr_sb_best | - [wild_rose_features](https://github.com/views-platform/views-models/blob/development/models/wild_rose/configs/config_queryset.py) | - [hyperparameters wild_rose](https://github.com/views-platform/views-models/blob/development/models/wild_rose/configs/config_hyperparameters.py) | shadow | 2025-03-19 | Håvard |
| [wuthering_heights](https://github.com/views-platform/views-models/blob/development/models/wuthering_heights) | ShurfModel | lr_sb_best | - [wuthering_heights_features](https://github.com/views-platform/views-models/blob/development/models/wuthering_heights/configs/config_queryset.py) | - [hyperparameters wuthering_heights](https://github.com/views-platform/views-models/blob/development/models/wuthering_heights/configs/config_hyperparameters.py) | shadow | 2025-03-19 | Håvard |
| [yellow_ranger](https://github.com/views-platform/views-models/blob/development/models/yellow_ranger) | MixtureBaseline | lr_os_best | - [yellow_ranger_features](https://github.com/views-platform/views-models/blob/development/models/yellow_ranger/configs/config_queryset.py) | - [hyperparameters yellow_ranger](https://github.com/views-platform/views-models/blob/development/models/yellow_ranger/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [yellow_submarine](https://github.com/views-platform/views-models/blob/development/models/yellow_submarine) | XGBRFRegressor | lr_ged_sb | - [yellow_submarine_features](https://github.com/views-platform/views-models/blob/development/models/yellow_submarine/configs/config_queryset.py) | - [hyperparameters yellow_submarine](https://github.com/views-platform/views-models/blob/development/models/yellow_submarine/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Marina |
| [zero_cmbaseline](https://github.com/views-platform/views-models/blob/development/models/zero_cmbaseline) | ZeroModel | lr_ged_sb | N/A | - [hyperparameters zero_cmbaseline](https://github.com/views-platform/views-models/blob/development/models/zero_cmbaseline/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Sonja |

<!-- CM_TABLE_END -->

---

### PRIO-GRID-Month Model Catalog

<!-- PGM_TABLE_START -->
| Model Name | Algorithm | Targets | Input Features | Hyperparameters | Implementation Status | Implementation Date | Author |
| ---------- | --------- | ------- | -------------- | --------------- | --------------------- | ------------------- | ------ |
| [average_pgmbaseline](https://github.com/views-platform/views-models/blob/development/models/average_pgmbaseline) | AverageModel | lr_ged_sb | N/A | - [hyperparameters average_pgmbaseline](https://github.com/views-platform/views-models/blob/development/models/average_pgmbaseline/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Sonja |
| [bad_blood](https://github.com/views-platform/views-models/blob/development/models/bad_blood) | LGBMRegressor | lr_ged_sb | - [bad_blood_features](https://github.com/views-platform/views-models/blob/development/models/bad_blood/configs/config_queryset.py) | - [hyperparameters bad_blood](https://github.com/views-platform/views-models/blob/development/models/bad_blood/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [black_ranger](https://github.com/views-platform/views-models/blob/development/models/black_ranger) | MixtureBaseline | lr_os_best | - [black_ranger_features](https://github.com/views-platform/views-models/blob/development/models/black_ranger/configs/config_queryset.py) | - [hyperparameters black_ranger](https://github.com/views-platform/views-models/blob/development/models/black_ranger/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [blank_space](https://github.com/views-platform/views-models/blob/development/models/blank_space) | HurdleModel | lr_ged_sb | - [blank_space_features](https://github.com/views-platform/views-models/blob/development/models/blank_space/configs/config_queryset.py) | - [hyperparameters blank_space](https://github.com/views-platform/views-models/blob/development/models/blank_space/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [blazing_meteor](https://github.com/views-platform/views-models/blob/development/models/blazing_meteor) | HydraNet |  | - [blazing_meteor_features](https://github.com/views-platform/views-models/blob/development/models/blazing_meteor/configs/config_queryset.py) | - [hyperparameters blazing_meteor](https://github.com/views-platform/views-models/blob/development/models/blazing_meteor/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [blue_ranger](https://github.com/views-platform/views-models/blob/development/models/blue_ranger) | MixtureBaseline | lr_ged_sb | - [blue_ranger_features](https://github.com/views-platform/views-models/blob/development/models/blue_ranger/configs/config_queryset.py) | - [hyperparameters blue_ranger](https://github.com/views-platform/views-models/blob/development/models/blue_ranger/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [blue_stranger](https://github.com/views-platform/views-models/blob/development/models/blue_stranger) | HydraNet |  | - [blue_stranger_features](https://github.com/views-platform/views-models/blob/development/models/blue_stranger/configs/config_queryset.py) | - [hyperparameters blue_stranger](https://github.com/views-platform/views-models/blob/development/models/blue_stranger/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [bold_comet](https://github.com/views-platform/views-models/blob/development/models/bold_comet) | HydraNet |  | - [bold_comet_features](https://github.com/views-platform/views-models/blob/development/models/bold_comet/configs/config_queryset.py) | - [hyperparameters bold_comet](https://github.com/views-platform/views-models/blob/development/models/bold_comet/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [bright_starship](https://github.com/views-platform/views-models/blob/development/models/bright_starship) | HydraNet |  | - [bright_starship_features](https://github.com/views-platform/views-models/blob/development/models/bright_starship/configs/config_queryset.py) | - [hyperparameters bright_starship](https://github.com/views-platform/views-models/blob/development/models/bright_starship/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [caring_fish](https://github.com/views-platform/views-models/blob/development/models/caring_fish) | XGBRegressor | lr_ged_sb | - [caring_fish_features](https://github.com/views-platform/views-models/blob/development/models/caring_fish/configs/config_queryset.py) | - [hyperparameters caring_fish](https://github.com/views-platform/views-models/blob/development/models/caring_fish/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [chunky_cat](https://github.com/views-platform/views-models/blob/development/models/chunky_cat) | LGBMRegressor | lr_ged_sb | - [chunky_cat_features](https://github.com/views-platform/views-models/blob/development/models/chunky_cat/configs/config_queryset.py) | - [hyperparameters chunky_cat](https://github.com/views-platform/views-models/blob/development/models/chunky_cat/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [dark_paradise](https://github.com/views-platform/views-models/blob/development/models/dark_paradise) | HurdleModel | lr_ged_sb | - [dark_paradise_features](https://github.com/views-platform/views-models/blob/development/models/dark_paradise/configs/config_queryset.py) | - [hyperparameters dark_paradise](https://github.com/views-platform/views-models/blob/development/models/dark_paradise/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [heavy_freighter](https://github.com/views-platform/views-models/blob/development/models/heavy_freighter) | HydraNet |  | - [heavy_freighter_features](https://github.com/views-platform/views-models/blob/development/models/heavy_freighter/configs/config_queryset.py) | - [hyperparameters heavy_freighter](https://github.com/views-platform/views-models/blob/development/models/heavy_freighter/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [heavy_strider](https://github.com/views-platform/views-models/blob/development/models/heavy_strider) | ConflictologyModel |  | - [heavy_strider_features](https://github.com/views-platform/views-models/blob/development/models/heavy_strider/configs/config_queryset.py) | - [hyperparameters heavy_strider](https://github.com/views-platform/views-models/blob/development/models/heavy_strider/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [invisible_string](https://github.com/views-platform/views-models/blob/development/models/invisible_string) | LGBMRegressor | lr_ged_sb | - [invisible_string_features](https://github.com/views-platform/views-models/blob/development/models/invisible_string/configs/config_queryset.py) | - [hyperparameters invisible_string](https://github.com/views-platform/views-models/blob/development/models/invisible_string/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [lavender_haze](https://github.com/views-platform/views-models/blob/development/models/lavender_haze) | HurdleModel | lr_ged_sb | - [lavender_haze_features](https://github.com/views-platform/views-models/blob/development/models/lavender_haze/configs/config_queryset.py) | - [hyperparameters lavender_haze](https://github.com/views-platform/views-models/blob/development/models/lavender_haze/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [light_strider](https://github.com/views-platform/views-models/blob/development/models/light_strider) | ConflictologyModel |  | - [light_strider_features](https://github.com/views-platform/views-models/blob/development/models/light_strider/configs/config_queryset.py) | - [hyperparameters light_strider](https://github.com/views-platform/views-models/blob/development/models/light_strider/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [locf_pgmbaseline](https://github.com/views-platform/views-models/blob/development/models/locf_pgmbaseline) | LocfModel | lr_ged_sb | N/A | - [hyperparameters locf_pgmbaseline](https://github.com/views-platform/views-models/blob/development/models/locf_pgmbaseline/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Sonja |
| [midnight_rain](https://github.com/views-platform/views-models/blob/development/models/midnight_rain) | LGBMRegressor | lr_ged_sb | - [midnight_rain_features](https://github.com/views-platform/views-models/blob/development/models/midnight_rain/configs/config_queryset.py) | - [hyperparameters midnight_rain](https://github.com/views-platform/views-models/blob/development/models/midnight_rain/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [old_money](https://github.com/views-platform/views-models/blob/development/models/old_money) | HurdleModel | lr_ged_sb | - [old_money_features](https://github.com/views-platform/views-models/blob/development/models/old_money/configs/config_queryset.py) | - [hyperparameters old_money](https://github.com/views-platform/views-models/blob/development/models/old_money/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [orange_pasta](https://github.com/views-platform/views-models/blob/development/models/orange_pasta) | LGBMRegressor | lr_ged_sb | - [orange_pasta_features](https://github.com/views-platform/views-models/blob/development/models/orange_pasta/configs/config_queryset.py) | - [hyperparameters orange_pasta](https://github.com/views-platform/views-models/blob/development/models/orange_pasta/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [pink_pirate](https://github.com/views-platform/views-models/blob/development/models/pink_pirate) | HydraNet |  | - [pink_pirate_features](https://github.com/views-platform/views-models/blob/development/models/pink_pirate/configs/config_queryset.py) | - [hyperparameters pink_pirate](https://github.com/views-platform/views-models/blob/development/models/pink_pirate/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [pink_ranger](https://github.com/views-platform/views-models/blob/development/models/pink_ranger) | MixtureBaseline | lr_ns_best | - [pink_ranger_features](https://github.com/views-platform/views-models/blob/development/models/pink_ranger/configs/config_queryset.py) | - [hyperparameters pink_ranger](https://github.com/views-platform/views-models/blob/development/models/pink_ranger/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [purple_alien](https://github.com/views-platform/views-models/blob/development/models/purple_alien) | HydraNet |  | - [purple_alien_features](https://github.com/views-platform/views-models/blob/development/models/purple_alien/configs/config_queryset.py) | - [hyperparameters purple_alien](https://github.com/views-platform/views-models/blob/development/models/purple_alien/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [violet_visitor](https://github.com/views-platform/views-models/blob/development/models/violet_visitor) | HydraNet |  | - [violet_visitor_features](https://github.com/views-platform/views-models/blob/development/models/violet_visitor/configs/config_queryset.py) | - [hyperparameters violet_visitor](https://github.com/views-platform/views-models/blob/development/models/violet_visitor/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Simon |
| [white_ranger](https://github.com/views-platform/views-models/blob/development/models/white_ranger) | ConflictologyModel |  | - [white_ranger_features](https://github.com/views-platform/views-models/blob/development/models/white_ranger/configs/config_queryset.py) | - [hyperparameters white_ranger](https://github.com/views-platform/views-models/blob/development/models/white_ranger/configs/config_hyperparameters.py) | baseline | 2024-11-22 | Simon |
| [wildest_dream](https://github.com/views-platform/views-models/blob/development/models/wildest_dream) | HurdleModel | lr_ged_sb | - [wildest_dream_features](https://github.com/views-platform/views-models/blob/development/models/wildest_dream/configs/config_queryset.py) | - [hyperparameters wildest_dream](https://github.com/views-platform/views-models/blob/development/models/wildest_dream/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [yellow_pikachu](https://github.com/views-platform/views-models/blob/development/models/yellow_pikachu) | HurdleModel | lr_ged_sb | - [yellow_pikachu_features](https://github.com/views-platform/views-models/blob/development/models/yellow_pikachu/configs/config_queryset.py) | - [hyperparameters yellow_pikachu](https://github.com/views-platform/views-models/blob/development/models/yellow_pikachu/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Xiaolong |
| [zero_pgmbaseline](https://github.com/views-platform/views-models/blob/development/models/zero_pgmbaseline) | ZeroModel | lr_ged_sb | N/A | - [hyperparameters zero_pgmbaseline](https://github.com/views-platform/views-models/blob/development/models/zero_pgmbaseline/configs/config_hyperparameters.py) | shadow | 2024-11-22 | Sonja |

<!-- PGM_TABLE_END -->

---

### Ensemble Catalog

<!-- ENSEMBLE_TABLE_START -->
| Ensemble Name | Algorithm | Targets | Constituent Models | Hyperparameters | Implementation Status | Implementation Date | Author |
| ------------- | --------- | ------- | ------------------ | --------------- | --------------------- | ------------------- | ------ |
| [cruel_summer](https://github.com/views-platform/views-models/blob/development/ensembles/cruel_summer) | median | lr_ged_sb | - [cruel_summer_constituent_models](https://github.com/views-platform/views-models/blob/development/ensembles/cruel_summer/configs/config_modelset.py) | - [hyperparameters cruel_summer](https://github.com/views-platform/views-models/blob/development/ensembles/cruel_summer/configs/config_hyperparameters.py) | shadow | 2024-11-27 | Xiaolong |
| [first_love](https://github.com/views-platform/views-models/blob/development/ensembles/first_love) | concat | lr_ged_sb | - [first_love_constituent_models](https://github.com/views-platform/views-models/blob/development/ensembles/first_love/configs/config_modelset.py) | - [hyperparameters first_love](https://github.com/views-platform/views-models/blob/development/ensembles/first_love/configs/config_hyperparameters.py) | shadow | 2024-11-27 | Dylan |
| [golden_hour](https://github.com/views-platform/views-models/blob/development/ensembles/golden_hour) | concat | lr_sb_best, lr_ns_best, lr_os_best | - [golden_hour_constituent_models](https://github.com/views-platform/views-models/blob/development/ensembles/golden_hour/configs/config_modelset.py) | - [hyperparameters golden_hour](https://github.com/views-platform/views-models/blob/development/ensembles/golden_hour/configs/config_hyperparameters.py) | shadow | 2026-05-26 | Simon |
| [pink_ponyclub](https://github.com/views-platform/views-models/blob/development/ensembles/pink_ponyclub) | mean | lr_ged_sb | - [pink_ponyclub_constituent_models](https://github.com/views-platform/views-models/blob/development/ensembles/pink_ponyclub/configs/config_modelset.py) | - [hyperparameters pink_ponyclub](https://github.com/views-platform/views-models/blob/development/ensembles/pink_ponyclub/configs/config_hyperparameters.py) | shadow | 2025-02-20 | Xiaolong |
| [rude_boy](https://github.com/views-platform/views-models/blob/development/ensembles/rude_boy) | mean | lr_ged_sb | - [rude_boy_constituent_models](https://github.com/views-platform/views-models/blob/development/ensembles/rude_boy/configs/config_modelset.py) | - [hyperparameters rude_boy](https://github.com/views-platform/views-models/blob/development/ensembles/rude_boy/configs/config_hyperparameters.py) | shadow | 2024-11-27 | Dylan |
| [skinny_love](https://github.com/views-platform/views-models/blob/development/ensembles/skinny_love) | mean | lr_ged_sb | - [skinny_love_constituent_models](https://github.com/views-platform/views-models/blob/development/ensembles/skinny_love/configs/config_modelset.py) | - [hyperparameters skinny_love](https://github.com/views-platform/views-models/blob/development/ensembles/skinny_love/configs/config_hyperparameters.py) | shadow | 2025-02-20 | Xiaolong |
| [stellar_horizon](https://github.com/views-platform/views-models/blob/development/ensembles/stellar_horizon) | concat | lr_sb_best, lr_ns_best, lr_os_best | - [stellar_horizon_constituent_models](https://github.com/views-platform/views-models/blob/development/ensembles/stellar_horizon/configs/config_modelset.py) | - [hyperparameters stellar_horizon](https://github.com/views-platform/views-models/blob/development/ensembles/stellar_horizon/configs/config_hyperparameters.py) | shadow | 2026-05-26 | Simon |
| [white_mustang](https://github.com/views-platform/views-models/blob/development/ensembles/white_mustang) | mean | lr_ged_sb | - [white_mustang_constituent_models](https://github.com/views-platform/views-models/blob/development/ensembles/white_mustang/configs/config_modelset.py) | - [hyperparameters white_mustang](https://github.com/views-platform/views-models/blob/development/ensembles/white_mustang/configs/config_hyperparameters.py) | deployed | 2024-11-22 | Xiaolong |

<!-- ENSEMBLE_TABLE_END -->

---

## Acknowledgement

<p align="center">
  <img src="https://raw.githubusercontent.com/views-platform/docs/main/images/views_funders.png" alt="Views Funders" width="80%">
</p>

Special thanks to the **VIEWS MD&D Team** for their collaboration and support.  
