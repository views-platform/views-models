
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
        "shared_ratio":0.25,
        "batch_size": 8, #100
        "llm": "snowood1/ConfliBERT-scr-uncased",
        "learning_rate": 1e-5,
        "epochs": 1,
        "monte_carlo_runs": 100,
        "early_stopping": 3,
        "init_batch": 200,
        "max_samples": 2000,
        "metrics": ["accuracy", "f1"],
        "doccano_url": "http://localhost:5900",
        "doccano_user": "admin",
        "doccano_password": "password",
        "project_name": ["Edattack_Annotator1_Gudrun", "Edattack_Annotator2_Roos", "Edattack_Annotator_3_Kristine", "Edattack_Annotator4_Sonja", "Edattack_Annotator5_Dylan", "Consensus_Edattack"],
        "max_iterations": 10,
        "query_size": 10,
        "min_batch_size": 3,
        "annotation_size": 20,
        "labels": {"Violence":0, "Threats & Intimidation":1, "Recruitment & Abduction":2, "Sexual Violence":3, "Military Use / Occupation":4, "Arrest/Detention":5, "Target: Student":6, "Target: Personnel":7, "Target: Infrastructure":8, "Victim Gender: Only females":'a', "Victim Gender: Only males":'b', "Non-binary mentioned":'c', "Institution: Female-focused":'d', "Institution: Male-focused":'e', "Educational Level: Primary":'f', "Educational Level: Secondary":'g', "Educational Level: Higher":'h', "Incidental Flag":'i',"Known Conflict Actor":'j', "No attack on education":'k'},
"colours": {"Violence":"#8a9bee", "Threats & Intimidation": "#8a9bee", "Recruitment & Abduction":"#8a9bee", "Sexual Violence":"#8a9bee", "Military Use / Occupation":"#8a9bee", "Arrest/Detention": "#8a9bee", "Known Conflict Actor":"#e77a81", "Target: Student":"#40864A", "Target: Personnel":"#40864A", "Target: Infrastructure":"#40864A", "Victim Gender: Only females":"#868440", "Victim Gender: Only males":"#868440", "Non-binary mentioned":"#868440", "Institution: Female-focused":"#868440", "Institution: Male-focused":"#868440", "Educational Level: Primary":"#C1C1B7", "Educational Level: Secondary":"#C1C1B7", "Educational Level: Higher":"#C1C1B7", "Incidental Flag":"#D594C8", "No attack on education":"#611C1E"},
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
