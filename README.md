
<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://pbs.twimg.com/profile_banners/1237000633896652800/1717069203/1500x500" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

# Welcome to views-models repository! 

This repository contains all of the necesary components for creating new models which are compatible with the VIEWS pipeline. The views-models repository also contains all of the already implemented VIEWS models (with the exception of [HydraNet](https://github.com/views-platform/views-hydranet)), at both PRIO-GRID-month and country-month levels of analysis, along with information about prediction targets, input data and model algorithms. 

---

## Table of contents

<!-- toc -->

- [Key Terms and Definitions](#key-terms-and-definitions)
- [Model Naming Conventions](#model-naming-conventions) 
- [Creating New Models](#creatingnewmodels)
- [Model scripts](#modelscripts)
- [Model filesystems](#model_filesystems)
- [Running a single model](#running_a_single_model)
- [Ensembles](#ensembles)
- [Creating a new ensemble](#Creating-a-new-ensemble)
- [Ensemble scripts](#ensemble_scripts)
- [Ensemble filesystem](#ensemble_filesystem)
- [Running an ensemble](#running_an_ensemble)
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
<a name="creatingnewmodels"></a>

The views-models repository contains the tools for creating new models, as well as creating new model ensembles. All of the necessary components are found in the `build_model_scaffold.py` and `build_ensemble_scaffold.py` files. The goal of this part of the VIEWS pipeline is the ability to simply create models which have the right structure and fit into the VIEWS directory structure. This makes the models uniform, consistent, and allows for easier replicability. 

As with other parts of the VIEWS pipeline, we aim to make interactions with our pipeline as simple and straightfoward as possible. In the context of the views-models, when creating a new model or ensemble, the user is closely guided through the steps which are needed, in an intuitive manner. This allows for the model creation processes to be consistent no matter how experienced the creator is. After providing a name for the model or ensemble, guided to be in the form adjective_noun, the scaffold builders create all of the model files and model directories, uniformly structured. This instantly removes possibilities of error, increases efficiency and effectiveness as it decreases manual inputs of code. Finally, this allows all of our users, no matter their level of proficiency, to seamlessly interact with out pipeline in no time.  

To run the model scaffold builder, execute

`python build_model_scaffold.py`

You will be asked to enter a name for your model in lowercase `adjective_noun` form. If the scaffolder is happy with your proposed model name, it will create a new directory with your chosen name. This directory in turn contains the scripts and folders needed to run your model and store intermediate data belonging to it. The scripts created are as follows (see further down for a description of the filesystem):

---

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

The VIEWS platform solves this problem by building a custom Python **enviroment** for every model. A Python environment is an isolated sandbox into which a particular set of modules can be installed, and it does not matter if the modules installed on one environment are incompatible with those installed in another. Code execution can be quickly switched between environments, so that models with dependency conflicts can be easily executed in series.

The `run.sh` script first builds the environment required to run a model, specified in the `requirements.txt` file - see below), and then executes the model inside that environment by passing its `main.py` file (see below) to the Python interpreter.

## `requirements.txt`
The purpose of this file is to specify which modules (probably including their versions or an acceptable range thereof) need to be installed in the model-specific environment built by `run.sh`.

**It is the model creator's responsibility to ensure that this file is correctly populated.** Only modules named in this file (and their dependencies) will be installed in the model env. If your model needs `numpy` and it is not installed by any other dependencies, it needs to be specified here.

It is strongly advised to specify a range of acceptable versions for each installed module using the standard notation, e.g. `views-stepshifter>=0.1.2,<1.0.0`.

## `main.py`
Once the `run.sh` script has created the model's environment, it activates the environment and executes the `main.py` file inside it. The `main.py` has several tasks:

- it uses the `ModelPathManager` from `views-pipeline-core` to establish where it is on the host machine's filesystem so that other scripts and modules can be found by the Python interpreter at runtime
- it logs into `weights-and-biases` - all runs executed in the VIEWS platform are automatically externally logged to the weights-and-biases web platform - URLs are printed to the terminal during model/ensemble execution, which will take users to webpages showing live logging and analytics
- it parses command line arguments (forwarded by `run.sh`) which specify whether the model is to be trained, whether a sweep over hyperparameters should be performed, etc.
- it then calls the relevant `Manager` from `views-pipeline-core` which superintends the execution of the model. Every class of models has its own custom manager (e.g. `StepShifterManager` looks after stepshifted regression models). **If you are introducing a new class of model to VIEWS, you will need to create a new Manager class for it.**

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


- `config_sweep.py`: During model development, developers will often wish to perform sweeps over ranges of model hyperparameters for optimisation purposes. This script allows such sweeps to be configured, specifying which parameters ranges are to explored and what is to be optimised.


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

## Running a single model
A model is run by executing the `run.sh` script in its root directory, which checks to see if an appropriate environment for the model exists, creates one if not, activates the environment, and executes the model's `main.py` inside it. The model can be run by executing the `main.py` directly, but it is then up to the user to ensure that the model's environment is correctly built and activated.

The `run.sh` and `main.py` both require command line arguments to control their behaviour (command line arguments submitted to `run.sh` are simply passed on to `main.py`). A description of these arguments follows:

- `-r` or `--run_type` followed by one of [`calibration`, `validation`, `forecasting`]:  choose the run type


- `-s` or `--sweep`: perform a sweep run (run type must be `calibration`)


- `-t` or `--train`: flag indicating whether a new model artifact should be trained


- `-e` or `--evaluate`: flag indicating if model should be evaluated


- `-f` or `--forecast`: flag indicating if forecasts are to be generated


- `-a` or `--artifact_name`: flag allowing the name of the artifact to be evaluated to be supplied


- `-en` or `--ensemble`: flag to indicate that a model is en ensemble


- `-sa` or `--saved`: flag to indicate that saved data/artifacts should be used


- `-o` or `--override_month`: flag allowing one to specify a month other than the most recent month with data from which to forecast


- `-et` or `--eval_type`: flag allowing type of evaluation to be performed to be specified

# Ensembles

An ensemble is a combination of models which has greater predictive power than any of the models does singly. Ensemble forecasts can be simple averages over the forecasts of their constituent models, or a more sophisticated weighted average, where the weights are computed by optimising the ensemble's predictive performance over a specially reserved data partition, using, for example, an evolutionary algorithm. (The latter is not yet implemented in the VIEWS pipeline).

It is also possible to reconcile one ensemble with another (usually at a different spatial resolution) to, for example, force the forecasts to agree over well-defined spatial areas such as countries. The VIEWS pipeline allows point priogrid-level forecasts to be reconciled with country-level forecasts on a month-by-month basis (accounting for the fact that countries change size or appear/disappear altogether).

## Creating New Ensembles 

The procedure for creating a new ensemble is much the same as that for creating a new model. The `build_ensemble_scaffold.py` script is run and, once it is supplied with a legal lower case `adjective_noun` ensemble name, a filesystem very similar to that created for a new model is built.

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

It is strongly advised to specify a range of acceptable versions for each installed module using the standard notation, e.g. `views-stepshifter>=0.1.2,<1.0.0`.

## `main.py`
Once the `run.sh` script has created the ensemble's environment, it activates the environment and executes the `main.py` file inside it. The `main.py` has several tasks:

- it uses the `EnsemblePathManager` from `views-pipeline-core` to establish where it is on the host machine's filesystem so that other scripts and modules can be found by the Python interpreter at runtime
- it logs into `weights-and-biases` - all runs executed in the VIEWS platform are automatically externally logged to the weights-and-biases web platform - URLs are printed to the terminal during model/ensemble execution, which will take users to webpages showing live logging and analytics
- it parses command line arguments (forwarded by `run.sh`) which specify whether the ensemble is to be trained, whether forecasts are to be generated, etc.
- it then calls the `EnsembleManager` from `views-pipeline-core` which superintends the execution of the ensemble.

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

## Running an ensemble
An ensemble is run by executing the `run.sh` script in its root directory, which checks to see if an appropriate environment for the ensemble exists, creates one if not, activates the environment, and executes the ensemble's `main.py` inside it. The ensemble can be run by executing the `main.py` directly, but it is then up to the user to ensure that the model's environment is correctly built and activated.

The `run.sh` and `main.py` both require command line arguments to control their behaviour (command line arguments submitted to `run.sh` are simply passed on to `main.py`). A description of these arguments follows:

- `-r` or `--run_type` followed by one of [`calibration`, `validation`, `forecasting`]:  choose the run type


- `-t` or `--train`: flag indicating whether new model artifacts should be trained


- `-e` or `--evaluate`: flag indicating if the ensemble should be evaluated


- `-f` or `--forecast`: flag indicating if forecasts are to be generated


- `-en` or `--ensemble`: flag to indicate that a model is en ensemble


- `-sa` or `--saved`: flag to indicate that saved data/artifacts should be used


- `-o` or `--override_month`: flag allowing one to specify a month other than the most recent month with data from which to forecast


- `-et` or `--eval_type`: flag allowing type of evaluation to be performed to be specified


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

---
## Catalogs

The catalogs for all of the existing VIEWS models can be found below. The models catalogs are separated based on the models' level of analysis - country-month models and PRIO-GRID-month models, with the ensamble catalog all the way at the bottom. All of the information about algorithms, input features, hyperparameters and other model specification are included.

### Country-Month Model Catalog

<!-- CM_TABLE_START -->
| Model Name | Algorithm | Target | Input Features | Non-default Hyperparameters | Forecasting Type | Implementation Status | Implementation Date | Author |
| ---------- | --------- | ------ | -------------- | --------------------------- | ---------------- | --------------------- | ------------------- | ------ |
| bittersweet_symphony | XGBRegressor | ln_ged_sb_dep | - [ fatalities003_all_features](https://github.com/views-platform/views-models/blob/main/models/bittersweet_symphony/configs/config_queryset.py) | - [hyperparameters bittersweet_symphony](https://github.com/views-platform/views-models/blob/main/models/bittersweet_symphony/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| brown_cheese | XGBRFRegressor | ln_ged_sb_dep | - [fatalities003_baseline](https://github.com/views-platform/views-models/blob/main/models/brown_cheese/configs/config_queryset.py) | - [hyperparameters brown_cheese](https://github.com/views-platform/views-models/blob/main/models/brown_cheese/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| car_radio | XGBRegressor | ln_ged_sb_dep | - [fatalities003_topics](https://github.com/views-platform/views-models/blob/main/models/car_radio/configs/config_queryset.py) | - [hyperparameters car_radio](https://github.com/views-platform/views-models/blob/main/models/car_radio/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| counting_stars | XGBRegressor | ln_ged_sb_dep | - [fatalities003_conflict_history_long](https://github.com/views-platform/views-models/blob/main/models/counting_stars/configs/config_queryset.py) | - [hyperparameters counting_stars](https://github.com/views-platform/views-models/blob/main/models/counting_stars/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| demon_days | XGBRFRegressor | ln_ged_sb_dep | - [fatalities003_faostat](https://github.com/views-platform/views-models/blob/main/models/demon_days/configs/config_queryset.py) | - [hyperparameters demon_days](https://github.com/views-platform/views-models/blob/main/models/demon_days/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| electric_relaxation | RandomForestRegressor | ged_sb_dep | - [escwa001_cflong](https://github.com/views-platform/views-models/blob/main/models/electric_relaxation/configs/config_queryset.py) | - [hyperparameters electric_relaxation](https://github.com/views-platform/views-models/blob/main/models/electric_relaxation/configs/config_hyperparameters.py) | None | shadow | NA | Sara |
| fast_car | HurdleModel | ln_ged_sb_dep | - [fatalities003_vdem_short](https://github.com/views-platform/views-models/blob/main/models/fast_car/configs/config_queryset.py) | - [hyperparameters fast_car](https://github.com/views-platform/views-models/blob/main/models/fast_car/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| fluorescent_adolescent | HurdleModel | ln_ged_sb_dep | - [fatalities003_joint_narrow](https://github.com/views-platform/views-models/blob/main/models/fluorescent_adolescent/configs/config_queryset.py) | - [hyperparameters fluorescent_adolescent](https://github.com/views-platform/views-models/blob/main/models/fluorescent_adolescent/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| good_riddance | XGBRFRegressor | ln_ged_sb_dep | - [fatalities003_joint_narrow](https://github.com/views-platform/views-models/blob/main/models/good_riddance/configs/config_queryset.py) | - [hyperparameters good_riddance](https://github.com/views-platform/views-models/blob/main/models/good_riddance/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| green_squirrel | HurdleModel | ln_ged_sb_dep | - [fatalities003_joint_broad](https://github.com/views-platform/views-models/blob/main/models/green_squirrel/configs/config_queryset.py) | - [hyperparameters green_squirrel](https://github.com/views-platform/views-models/blob/main/models/green_squirrel/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| heavy_rotation | XGBRFRegressor | ln_ged_sb_dep | - [fatalities003_joint_broad](https://github.com/views-platform/views-models/blob/main/models/heavy_rotation/configs/config_queryset.py) | - [hyperparameters heavy_rotation](https://github.com/views-platform/views-models/blob/main/models/heavy_rotation/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| high_hopes | HurdleModel | ln_ged_sb_dep | - [fatalities003_conflict_history](https://github.com/views-platform/views-models/blob/main/models/high_hopes/configs/config_queryset.py) | - [hyperparameters high_hopes](https://github.com/views-platform/views-models/blob/main/models/high_hopes/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| little_lies | HurdleModel | ln_ged_sb_dep | - [fatalities003_joint_narrow](https://github.com/views-platform/views-models/blob/main/models/little_lies/configs/config_queryset.py) | - [hyperparameters little_lies](https://github.com/views-platform/views-models/blob/main/models/little_lies/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| national_anthem | XGBRFRegressor | ln_ged_sb_dep | - [fatalities003_wdi_short](https://github.com/views-platform/views-models/blob/main/models/national_anthem/configs/config_queryset.py) | - [hyperparameters national_anthem](https://github.com/views-platform/views-models/blob/main/models/national_anthem/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| ominous_ox | XGBRFRegressor | ln_ged_sb_dep | - [fatalities003_conflict_history](https://github.com/views-platform/views-models/blob/main/models/ominous_ox/configs/config_queryset.py) | - [hyperparameters ominous_ox](https://github.com/views-platform/views-models/blob/main/models/ominous_ox/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| plastic_beach | XGBRFRegressor | ln_ged_sb_dep | - [fatalities003_aquastat](https://github.com/views-platform/views-models/blob/main/models/plastic_beach/configs/config_queryset.py) | - [hyperparameters plastic_beach](https://github.com/views-platform/views-models/blob/main/models/plastic_beach/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| popular_monster | XGBRFRegressor | ln_ged_sb_dep | - [fatalities003_topics](https://github.com/views-platform/views-models/blob/main/models/popular_monster/configs/config_queryset.py) | - [hyperparameters popular_monster](https://github.com/views-platform/views-models/blob/main/models/popular_monster/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| teen_spirit | XGBRFRegressor | ln_ged_sb_dep | - [fatalities003_faoprices](https://github.com/views-platform/views-models/blob/main/models/teen_spirit/configs/config_queryset.py) | - [hyperparameters teen_spirit](https://github.com/views-platform/views-models/blob/main/models/teen_spirit/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| twin_flame | HurdleModel | ln_ged_sb_dep | - [fatalities003_topics](https://github.com/views-platform/views-models/blob/main/models/twin_flame/configs/config_queryset.py) | - [hyperparameters twin_flame](https://github.com/views-platform/views-models/blob/main/models/twin_flame/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| yellow_submarine | XGBRFRegressor | ln_ged_sb_dep | - [fatalities003_imfweo](https://github.com/views-platform/views-models/blob/main/models/yellow_submarine/configs/config_queryset.py) | - [hyperparameters yellow_submarine](https://github.com/views-platform/views-models/blob/main/models/yellow_submarine/configs/config_hyperparameters.py) | None | shadow | NA | Marina |

<!-- CM_TABLE_END -->

---

### PRIO-GRID-Month Model Catalog

<!-- PGM_TABLE_START -->
| Model Name | Algorithm | Target | Input Features | Non-default Hyperparameters | Forecasting Type | Implementation Status | Implementation Date | Author |
| ---------- | --------- | ------ | -------------- | --------------------------- | ---------------- | --------------------- | ------------------- | ------ |
| bad_blood | LGBMRegressor | ln_ged_sb_dep | - [fatalities003_pgm_natsoc](https://github.com/views-platform/views-models/blob/main/models/bad_blood/configs/config_queryset.py) | - [hyperparameters bad_blood](https://github.com/views-platform/views-models/blob/main/models/bad_blood/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| blank_space | HurdleModel | ln_ged_sb_dep | - [fatalities003_pgm_natsoc](https://github.com/views-platform/views-models/blob/main/models/blank_space/configs/config_queryset.py) | - [hyperparameters blank_space](https://github.com/views-platform/views-models/blob/main/models/blank_space/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| caring_fish | XGBRegressor | ln_ged_sb_dep | - [fatalities003_pgm_conflict_history](https://github.com/views-platform/views-models/blob/main/models/caring_fish/configs/config_queryset.py) | - [hyperparameters caring_fish](https://github.com/views-platform/views-models/blob/main/models/caring_fish/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| chunky_cat | LGBMRegressor | ln_ged_sb_dep | - [fatalities003_pgm_conflictlong](https://github.com/views-platform/views-models/blob/main/models/chunky_cat/configs/config_queryset.py) | - [hyperparameters chunky_cat](https://github.com/views-platform/views-models/blob/main/models/chunky_cat/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| dark_paradise | HurdleModel | ln_ged_sb_dep | - [fatalities003_pgm_conflictlong](https://github.com/views-platform/views-models/blob/main/models/dark_paradise/configs/config_queryset.py) | - [hyperparameters dark_paradise](https://github.com/views-platform/views-models/blob/main/models/dark_paradise/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| invisible_string | LGBMRegressor | ln_ged_sb_dep | - [fatalities003_pgm_broad](https://github.com/views-platform/views-models/blob/main/models/invisible_string/configs/config_queryset.py) | - [hyperparameters invisible_string](https://github.com/views-platform/views-models/blob/main/models/invisible_string/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| lavender_haze | HurdleModel | ln_ged_sb_dep | - [fatalities003_pgm_broad](https://github.com/views-platform/views-models/blob/main/models/lavender_haze/configs/config_queryset.py) | - [hyperparameters lavender_haze](https://github.com/views-platform/views-models/blob/main/models/lavender_haze/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| midnight_rain | LGBMRegressor | ln_ged_sb_dep | - [fatalities003_pgm_escwa_drought](https://github.com/views-platform/views-models/blob/main/models/midnight_rain/configs/config_queryset.py) | - [hyperparameters midnight_rain](https://github.com/views-platform/views-models/blob/main/models/midnight_rain/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| old_money | HurdleModel | ln_ged_sb_dep | - [fatalities003_pgm_escwa_drought](https://github.com/views-platform/views-models/blob/main/models/old_money/configs/config_queryset.py) | - [hyperparameters old_money](https://github.com/views-platform/views-models/blob/main/models/old_money/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| orange_pasta | LGBMRegressor | ln_ged_sb_dep | - [fatalities003_pgm_baseline](https://github.com/views-platform/views-models/blob/main/models/orange_pasta/configs/config_queryset.py) | - [hyperparameters orange_pasta](https://github.com/views-platform/views-models/blob/main/models/orange_pasta/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| purple_alien | HydraNet | ln_sb_best, ln_ns_best, ln_os_best, ln_sb_best_binarized, ln_ns_best_binarized, ln_os_best_binarized | - [escwa001_cflong](https://github.com/views-platform/views-models/blob/main/models/purple_alien/configs/config_queryset.py) | - [hyperparameters purple_alien](https://github.com/views-platform/views-models/blob/main/models/purple_alien/configs/config_hyperparameters.py) | None | shadow | NA | Simon |
| wildest_dream | HurdleModel | ln_ged_sb_dep | - [fatalities003_pgm_conflict_sptime_dist](https://github.com/views-platform/views-models/blob/main/models/wildest_dream/configs/config_queryset.py) | - [hyperparameters wildest_dream](https://github.com/views-platform/views-models/blob/main/models/wildest_dream/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| yellow_pikachu | HurdleModel | ln_ged_sb_dep | - [fatalities003_pgm_conflict_treelag](https://github.com/views-platform/views-models/blob/main/models/yellow_pikachu/configs/config_queryset.py) | - [hyperparameters yellow_pikachu](https://github.com/views-platform/views-models/blob/main/models/yellow_pikachu/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |

<!-- PGM_TABLE_END -->

---

### Ensemble Catalog

<!-- ENSEMBLE_TABLE_START -->
| Model Name | Algorithm | Target | Input Features | Non-default Hyperparameters | Forecasting Type | Implementation Status | Implementation Date | Author |
| ---------- | --------- | ------ | -------------- | --------------------------- | ---------------- | --------------------- | ------------------- | ------ |
| cruel_summer |  | ln_ged_sb_dep | None | - [hyperparameters cruel_summer](https://github.com/views-platform/views-models/blob/main/ensembles/cruel_summer/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| pink_ponyclub |  | ln_ged_sb_dep | None | - [hyperparameters pink_ponyclub](https://github.com/views-platform/views-models/blob/main/ensembles/pink_ponyclub/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| skinny_love |  | ln_ged_sb_dep | None | - [hyperparameters skinny_love](https://github.com/views-platform/views-models/blob/main/ensembles/skinny_love/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| white_mustang |  | ln_ged_sb_dep | None | - [hyperparameters white_mustang](https://github.com/views-platform/views-models/blob/main/ensembles/white_mustang/configs/config_hyperparameters.py) | None | deployed | NA | Xiaolong |

<!-- ENSEMBLE_TABLE_END -->

---

## Acknowledgement

<p align="center">
  <img src="https://raw.githubusercontent.com/views-platform/docs/main/images/views_funders.png" alt="Views Funders" width="80%">
</p>

Special thanks to the **VIEWS MD&D Team** for their collaboration and support.  
