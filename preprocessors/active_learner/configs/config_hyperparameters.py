
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
        "batch_size": 100, #100
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
        "project_name": "Conflict Text Classification",
        "max_iterations": 10,
        "query_size": 100,
        "min_batch_size": 1,
        "labels": {"Attack_on_schools":0, "Attack_on_HE":1, "Attack_on_student_personal":2, "Threats_intimidation":3, "Military_use":4, "Child_recruitement_abductions":5, "Sexual_violence":6, "Incidents":7, "No_edattack":8},
    }
    return hyperparameters

# {0: 'Adult',
#  1: 'Art & Design',
#  10: 'Food & Dining',
#  11: 'Games',
#  12: 'Health',
#  13: 'History',
#  14: 'Home & Hobbies',
#  15: 'Industrial',
#  16: 'Literature',
#  17: 'Politics',
#  18: 'Religion',
#  19: 'Science & Tech.',
#  2: 'Software Dev.',
#  20: 'Software',
#  21: 'Sports & Fitness',
#  22: 'Transportation',
#  23: 'Travel',
#  3: 'Crime & Law',
#  4: 'Education & Jobs',
#  5: 'Hardware',
#  6: 'Entertainment',
#  7: 'Social Life',
#  8: 'Fashion & Beauty',
#  9: 'Finance & Business'}