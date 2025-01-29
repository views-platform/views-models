
<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://pbs.twimg.com/profile_banners/1237000633896652800/1717069203/1500x500" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

# Welcome to view-models repository! 

This repository contains all of the necesary components for creating new models which are compatible with the VIEWS pipeline. The views-models repository also contains all of the already implemented VIEWS models (with the exception of [HydraNet](https://github.com/views-platform/views-hydranet)), at both PRIO-GRID-month and country-month levels of analysis, along with information about prediction targets, input data and model algorithms. 

---

## Table of contents

<!-- toc -->

- [Key Terms and Definitions](#key-terms-and-definitions)
- [Model Naming Conventions](#model-naming-conventions) 
- [Creating New Models](#creating-new-models)
- [Implemented Models](#implemented-models)
- [Model Catalogs](#catalogs)
    - [Country-Month Models](#country-month-model-catalog)
    - [PRIO-GRID-Month Model](#prio-grid-month-model-catalog)
    - [Ensambles](#ensemble-catalog)
- [Platform Structure and Contents](#views-platform-organization-structure-and-contents)
- [Further Resources and Documentation](#further-resources-and-documentation)
- [About the VIEWS Project](#about-the-views-project)

<!-- tocstop -->

---

## Key Terms and Definitions 

In VIEWS terminology a **model** is defined as: 
1. A specific instantiation of a machine learning algorithm,
2. Trained using a predetermined and unique set of hyperparameters,
3. On a well-defined set of input features,
    - The specific input features for every model are reffered to as [querysets](https://github.com/prio-data/viewser?tab=readme-ov-file#via-api). 
4. And targeting a specific outcome target.
5. In the case of [stepshift models](https://github.com/views-platform/views-stepshifter/blob/main/README.md), a model is understood as all code and all artifacts necessary to generate a comprehensive 36 month forecast for the specified target.
6. Note that, two models, identical in all other aspects, will be deemed distinct if varying post-processing techniques are applied to their generated predictions. For instance, if one model's predictions undergo calibration or normalization while the other's do not. Similarly, two models identical in all aspects are considered distinct if they utilize different input features (querysets).

---

## Model Naming Conventions

The models belonging to the VIEWS pipeline follow a predetermined naming standard. Models no longer carry descriptive titles (e.g., transform_log_clf_name_LGBMClassifier_reg_name_LGBMRegressor). Although such titles provided some information about the models, as  models are developed over time, this type of naming could cause confusion and ultimately small differences could not be communicated properly through the model titles. Instead, we rely on the metadata of the model for model specifications and being able to substantively differentiate them between each other.

Additionaly, the new naming convention for models in the pipeline takes the form of adjective_noun, adding more models alphabetically. For example, the first model to be added can be named amazing_apple, the second model bad_bunny, etc. This is a popular practice, and Weights & Biases implements this naming convention automatically.

---

## Creating New Models 

The views-models repository contains the tools for creating new models, as well as creating new model ensembles. All of the necessary components are found in the `build_model_scaffold.py` and `build_ensemble_scaffold.py` files. The goal of this part of the VIEWS pipeline is the ability to simply create models which have the right structure and fit into the VIEWS directory structure. This makes the models uniform, consistent, and allows for easier replicability. 

As with other parts of the VIEWS pipeline, we aim to make interactions with our pipeline as simple and straightfoward as possible. In the context of the views-models, when creating a new model or ensemble, the user is closely guided through the steps which are needed, in an intuitive manner. This allows for the model creation processes to be consistent no matter how experienced the creator is. After providing a name for the model or ensemble, guided to be in the form adjective_noun, the scaffold builders create all of the model files and model directories, uniformly structured. This instantly removes possibilities of error, increases efficiency and effectiveness as it decreases manual inputs of code. Finally, this allows all of our users, no matter their level of proficiency, to seamlessly interact with out pipeline in no time.  

---

## Implemented Models

In addition to the possibility of easily creating new models and ensembles, in order to maintain an organized and structured overview over all of th eimplemented models, the views-models repository also contains model catalogs containing all of the information about individual models. This information is collected from the metadata of each model and entails:
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
| bittersweet_symphony | XGBModel | ln_ged_sb_dep | - [ fatalities003_all_features](https://github.com/views-platform/views-models/blob/main/models/bittersweet_symphony/configs/config_queryset.py) | - [hyperparameters bittersweet_symphony](https://github.com/views-platform/views-models/blob/main/models/bittersweet_symphony/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| brown_cheese | RandomForestModel | ln_ged_sb_dep | - [fatalities003_baseline](https://github.com/views-platform/views-models/blob/main/models/brown_cheese/configs/config_queryset.py) | - [hyperparameters brown_cheese](https://github.com/views-platform/views-models/blob/main/models/brown_cheese/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| car_radio | XGBModel | ln_ged_sb_dep | - [fatalities003_topics](https://github.com/views-platform/views-models/blob/main/models/car_radio/configs/config_queryset.py) | - [hyperparameters car_radio](https://github.com/views-platform/views-models/blob/main/models/car_radio/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| counting_stars | XGBModel | ln_ged_sb_dep | - [fatalities003_conflict_history_long](https://github.com/views-platform/views-models/blob/main/models/counting_stars/configs/config_queryset.py) | - [hyperparameters counting_stars](https://github.com/views-platform/views-models/blob/main/models/counting_stars/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| demon_days | RandomForestModel | ln_ged_sb_dep | - [fatalities003_faostat](https://github.com/views-platform/views-models/blob/main/models/demon_days/configs/config_queryset.py) | - [hyperparameters demon_days](https://github.com/views-platform/views-models/blob/main/models/demon_days/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| electric_relaxation | RandomForestModel | ged_sb_dep | - [escwa001_cflong](https://github.com/views-platform/views-models/blob/main/models/electric_relaxation/configs/config_queryset.py) | - [hyperparameters electric_relaxation](https://github.com/views-platform/views-models/blob/main/models/electric_relaxation/configs/config_hyperparameters.py) | None | shadow | NA | Sara |
| fast_car | HurdleModel | ln_ged_sb_dep | - [fatalities003_vdem_short](https://github.com/views-platform/views-models/blob/main/models/fast_car/configs/config_queryset.py) | - [hyperparameters fast_car](https://github.com/views-platform/views-models/blob/main/models/fast_car/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| fluorescent_adolescent | HurdleModel | ln_ged_sb_dep | - [fatalities003_joint_narrow](https://github.com/views-platform/views-models/blob/main/models/fluorescent_adolescent/configs/config_queryset.py) | - [hyperparameters fluorescent_adolescent](https://github.com/views-platform/views-models/blob/main/models/fluorescent_adolescent/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| good_riddance | RandomForestModel | ln_ged_sb_dep | - [fatalities003_joint_narrow](https://github.com/views-platform/views-models/blob/main/models/good_riddance/configs/config_queryset.py) | - [hyperparameters good_riddance](https://github.com/views-platform/views-models/blob/main/models/good_riddance/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| green_squirrel | HurdleModel | ln_ged_sb_dep | - [fatalities003_joint_broad](https://github.com/views-platform/views-models/blob/main/models/green_squirrel/configs/config_queryset.py) | - [hyperparameters green_squirrel](https://github.com/views-platform/views-models/blob/main/models/green_squirrel/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| heavy_rotation | RandomForestModel | ln_ged_sb_dep | - [fatalities003_joint_broad](https://github.com/views-platform/views-models/blob/main/models/heavy_rotation/configs/config_queryset.py) | - [hyperparameters heavy_rotation](https://github.com/views-platform/views-models/blob/main/models/heavy_rotation/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| high_hopes | HurdleModel | ln_ged_sb_dep | - [fatalities003_conflict_history](https://github.com/views-platform/views-models/blob/main/models/high_hopes/configs/config_queryset.py) | - [hyperparameters high_hopes](https://github.com/views-platform/views-models/blob/main/models/high_hopes/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| little_lies | HurdleModel | ln_ged_sb_dep | - [fatalities003_joint_narrow](https://github.com/views-platform/views-models/blob/main/models/little_lies/configs/config_queryset.py) | - [hyperparameters little_lies](https://github.com/views-platform/views-models/blob/main/models/little_lies/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| national_anthem | RandomForestModel | ln_ged_sb_dep | - [fatalities003_wdi_short](https://github.com/views-platform/views-models/blob/main/models/national_anthem/configs/config_queryset.py) | - [hyperparameters national_anthem](https://github.com/views-platform/views-models/blob/main/models/national_anthem/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| ominous_ox | RandomForestModel | ln_ged_sb_dep | - [fatalities003_conflict_history](https://github.com/views-platform/views-models/blob/main/models/ominous_ox/configs/config_queryset.py) | - [hyperparameters ominous_ox](https://github.com/views-platform/views-models/blob/main/models/ominous_ox/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| plastic_beach | RandomForestModel | ln_ged_sb_dep | - [fatalities003_aquastat](https://github.com/views-platform/views-models/blob/main/models/plastic_beach/configs/config_queryset.py) | - [hyperparameters plastic_beach](https://github.com/views-platform/views-models/blob/main/models/plastic_beach/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| popular_monster | RandomForestModel | ln_ged_sb_dep | - [fatalities003_topics](https://github.com/views-platform/views-models/blob/main/models/popular_monster/configs/config_queryset.py) | - [hyperparameters popular_monster](https://github.com/views-platform/views-models/blob/main/models/popular_monster/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| teen_spirit | RandomForestModel | ln_ged_sb_dep | - [fatalities003_faoprices](https://github.com/views-platform/views-models/blob/main/models/teen_spirit/configs/config_queryset.py) | - [hyperparameters teen_spirit](https://github.com/views-platform/views-models/blob/main/models/teen_spirit/configs/config_hyperparameters.py) | None | shadow | NA | Marina |
| twin_flame | HurdleModel | ln_ged_sb_dep | - [fatalities003_topics](https://github.com/views-platform/views-models/blob/main/models/twin_flame/configs/config_queryset.py) | - [hyperparameters twin_flame](https://github.com/views-platform/views-models/blob/main/models/twin_flame/configs/config_hyperparameters.py) | None | shadow | NA | Borbála |
| yellow_submarine | RandomForestModel | ln_ged_sb_dep | - [fatalities003_imfweo](https://github.com/views-platform/views-models/blob/main/models/yellow_submarine/configs/config_queryset.py) | - [hyperparameters yellow_submarine](https://github.com/views-platform/views-models/blob/main/models/yellow_submarine/configs/config_hyperparameters.py) | None | shadow | NA | Marina |

<!-- CM_TABLE_END -->

---

### PRIO-GRID-Month Model Catalog

<!-- PGM_TABLE_START -->
| Model Name | Algorithm | Target | Input Features | Non-default Hyperparameters | Forecasting Type | Implementation Status | Implementation Date | Author |
| ---------- | --------- | ------ | -------------- | --------------------------- | ---------------- | --------------------- | ------------------- | ------ |
| bad_blood | LightGBMModel | ln_ged_sb_dep | - [fatalities003_pgm_natsoc](https://github.com/views-platform/views-models/blob/main/models/bad_blood/configs/config_queryset.py) | - [hyperparameters bad_blood](https://github.com/views-platform/views-models/blob/main/models/bad_blood/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| blank_space | HurdleModel | ln_ged_sb_dep | - [fatalities003_pgm_natsoc](https://github.com/views-platform/views-models/blob/main/models/blank_space/configs/config_queryset.py) | - [hyperparameters blank_space](https://github.com/views-platform/views-models/blob/main/models/blank_space/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| caring_fish | XGBModel | ln_ged_sb_dep | - [fatalities003_pgm_conflict_history](https://github.com/views-platform/views-models/blob/main/models/caring_fish/configs/config_queryset.py) | - [hyperparameters caring_fish](https://github.com/views-platform/views-models/blob/main/models/caring_fish/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| chunky_cat | LightGBMModel | ln_ged_sb_dep | - [fatalities003_pgm_conflictlong](https://github.com/views-platform/views-models/blob/main/models/chunky_cat/configs/config_queryset.py) | - [hyperparameters chunky_cat](https://github.com/views-platform/views-models/blob/main/models/chunky_cat/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| dark_paradise | HurdleModel | ln_ged_sb_dep | - [fatalities003_pgm_conflictlong](https://github.com/views-platform/views-models/blob/main/models/dark_paradise/configs/config_queryset.py) | - [hyperparameters dark_paradise](https://github.com/views-platform/views-models/blob/main/models/dark_paradise/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| invisible_string | LightGBMModel | ln_ged_sb_dep | - [fatalities003_pgm_broad](https://github.com/views-platform/views-models/blob/main/models/invisible_string/configs/config_queryset.py) | - [hyperparameters invisible_string](https://github.com/views-platform/views-models/blob/main/models/invisible_string/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| lavender_haze | HurdleModel | ln_ged_sb_dep | - [fatalities003_pgm_broad](https://github.com/views-platform/views-models/blob/main/models/lavender_haze/configs/config_queryset.py) | - [hyperparameters lavender_haze](https://github.com/views-platform/views-models/blob/main/models/lavender_haze/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| midnight_rain | LightGBMModel | ln_ged_sb_dep | - [fatalities003_pgm_escwa_drought](https://github.com/views-platform/views-models/blob/main/models/midnight_rain/configs/config_queryset.py) | - [hyperparameters midnight_rain](https://github.com/views-platform/views-models/blob/main/models/midnight_rain/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| old_money | HurdleModel | ln_ged_sb_dep | - [fatalities003_pgm_escwa_drought](https://github.com/views-platform/views-models/blob/main/models/old_money/configs/config_queryset.py) | - [hyperparameters old_money](https://github.com/views-platform/views-models/blob/main/models/old_money/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
| orange_pasta | LightGBMModel | ln_ged_sb_dep | - [fatalities003_pgm_baseline](https://github.com/views-platform/views-models/blob/main/models/orange_pasta/configs/config_queryset.py) | - [hyperparameters orange_pasta](https://github.com/views-platform/views-models/blob/main/models/orange_pasta/configs/config_hyperparameters.py) | None | shadow | NA | Xiaolong |
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
| white_mustang |  | ln_ged_sb_dep | None | - [hyperparameters white_mustang](https://github.com/views-platform/views-models/blob/main/ensembles/white_mustang/configs/config_hyperparameters.py) | None | deployed | NA | Xiaolong |

<!-- ENSEMBLE_TABLE_END -->

---

## Funding and Partners 

<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative; margin-top: 50px;">
  <img src="image.png" alt="Funder logos" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

