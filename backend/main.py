import torch
from pathlib import Path
import tensorflow as tf
import numpy as np

def load_model(model_path: str):
    """
    Load a model from the specified path. Supports both PyTorch and Keras models.
    
    Args:
        model_path (str): Path to the saved model file
    
    Returns:
        model: Loaded model (either PyTorch or Keras)
    """
    try:
        # Check if file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Check file extension
        file_extension = Path(model_path).suffix.lower()
        
        if file_extension == '.keras':
            # Load Keras model
            return tf.keras.models.load_model(model_path)
        elif file_extension in ['.pt', '.pth']:
            # Load PyTorch model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.load(model_path, map_location=device)
            model.eval()  # Set to evaluation mode
            return model
        else:
            raise ValueError(f"Unsupported model format: {file_extension}")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Example usage
MODEL_PATH = "/Users/webninjaz-developer/Documents/Netflix_Stock_Analyse/model/lstm_model.keras"
model = load_model(MODEL_PATH)
