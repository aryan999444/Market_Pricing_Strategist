import joblib

def load_model(path):
    """Load a save model by using joblib"""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}")
        return None
    
def save_model(model, path):
    """Saves a trained model or scaler using joblib"""
    joblib.dump(model, path)
    print(f"Model successfully saved to {path}")