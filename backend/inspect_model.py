import joblib
import pandas as pd
import numpy as np

try:
    model = joblib.load("final_model.pkl")
    print("Model loaded.")
    print(f"Type: {type(model)}")
    
    if hasattr(model, "feature_names_in_"):
        print(f"Feature names: {model.feature_names_in_}")
    
    if hasattr(model, "n_features_in_"):
        print(f"Number of features: {model.n_features_in_}")

    # Try to see if it's a pipeline and what steps it has
    if hasattr(model, "steps"):
        print("Model is a Pipeline.")
        for name, step in model.steps:
            print(f"Step: {name}, Type: {type(step)}")
            if hasattr(step, "transformers_"):
                print(f"  Transformers: {step.transformers_}")
                for trans_name, trans, cols in step.transformers_:
                    # Check if transformer is a Pipeline
                    if hasattr(trans, "steps"):
                        for sub_name, sub_step in trans.steps:
                            if hasattr(sub_step, "categories_"):
                                print(f"    Categories for {trans_name} -> {sub_name}: {sub_step.categories_}")
                    # Check if transformer is directly the encoder
                    elif hasattr(trans, "categories_"):
                        print(f"    Categories for {trans_name}: {trans.categories_}")
            if hasattr(step, "feature_names_in_"):
                print(f"  Step Feature names: {step.feature_names_in_}")

except Exception as e:
    print(f"Error: {e}")
