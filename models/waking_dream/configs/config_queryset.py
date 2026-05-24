def generate():
    return {
        "source": "synthetic",
        "pattern": "diagonal_gradient",
        "level": "pgm",
        "features": ["synth_target"],
        "n_entities": 1000,
        "seed": 42,
    }
