def get_hp_config():
    """
    Contains the hyperparameter configurations for preprocessor training.
    This configuration is "operational" so modifying these settings will impact the preprocessor's behavior during the training.
    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the preprocessor, which determine the preprocessor's behavior during the training phase.
    """
    hyperparameters = {
        "topics": ["Education & Jobs", "Politics", "Crime & Law"],
        "priority_topic": "Education & Jobs",
        "n_sublists": 5,
        "shared_ratio": 0.25,
        "batch_size": 8,  # 100
        "llm": "snowood1/ConfliBERT-scr-uncased",
        "learning_rate": 1e-5,
        "epochs": 1,
        "monte_carlo_runs": 100,
        "early_stopping": 3,
        "init_batch": 200,
        "max_samples": 2000,
        "metrics": ["accuracy", "f1"],
        "doccano_url": "http://localhost:8000",
        "doccano_user": "admin",
        "doccano_password": "password",
        "project_name": [
            "Edattack_Annotator1",
            "Edattack_Annotator2",
            "Edattack_Annotator3",
            "Edattack_Annotator4",
            "Edattack_Annotator5",
        ],
        "max_iterations": 10,
        "query_size": 10,
        "min_batch_size": 3,
        "annotation_size": 20,
        "labels": {
            "Attack_on_schools": 0,
            "Attack_on_HE": 1,
            "Attack_on_student_personal": 2,
            "Threats_intimidation": 3,
            "Military_use": 4,
            "Child_recruitement_abductions": 5,
            "Sexual_violence": 6,
            "Incidents": 7,
            "No_edattack": 8,
        },
    }
    return hyperparameters
