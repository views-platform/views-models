def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """

    meta_config = {
        "name": "seldon_api",
        "algorithm": "API",
        # "creator": "Your name here",
        "source": {
            "forecast": {
                "cm": [
                    {
                        "ensemble": "pink_ponyclub",
                        "targets": [
                            "pred_lr_ged_sb",
                        ],
                    },
                    {
                        "postprocessor": "testcm_postprocessor",
                        "targets": [
                            "pred_lr_ged_sb",
                        ],
                    }
                ],
                "pgm": [
                    {
                        "ensemble": "skinny_love",
                        "targets": [
                            "pred_lr_ged_sb",
                        ],
                    }
                ],
            },
            "historical": {
                "pgm": [
                    {
                        "postprocessor": "seldonapi_postprocessor",
                        "targets": [
                            "lr_ged_sb", "lr_ged_ns", "lr_ged_os"
                        ],
                    }
                ],
            },
        },
    }
    return meta_config
