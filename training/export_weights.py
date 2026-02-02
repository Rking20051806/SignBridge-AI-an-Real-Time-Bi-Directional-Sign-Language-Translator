"""
Export trained CNN model weights to JSON for JavaScript inference.
Creates a format that can be loaded directly without TensorFlow.js
"""

import json
import numpy as np
from pathlib import Path
import pickle

# Use TensorFlow to load the model
import tensorflow as tf
from tensorflow import keras

OUTPUT_PATH = Path(__file__).parent / "output"

def export_model_weights():
    """Export the trained model weights in a JavaScript-friendly format."""
    
    # Load the trained model
    model_path = OUTPUT_PATH / "best_cnn_model.keras"
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load label encoder
    with open(OUTPUT_PATH / "cnn_label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    print(f"Model loaded. Classes: {label_encoder.classes_.tolist()}")
    
    # Extract weights layer by layer
    weights_data = {
        "model_type": "cnn",
        "input_shape": [64, 64, 1],
        "classes": label_encoder.classes_.tolist(),
        "layers": []
    }
    
    print("\nExtracting layers:")
    for i, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        layer_config = layer.get_config()
        layer_type = layer.__class__.__name__
        
        layer_info = {
            "index": i,
            "name": layer.name,
            "type": layer_type,
            "config": {}
        }
        
        # Extract relevant config based on layer type
        if layer_type == "Conv2D":
            layer_info["config"] = {
                "filters": layer_config["filters"],
                "kernel_size": layer_config["kernel_size"],
                "activation": layer_config["activation"],
                "padding": layer_config["padding"]
            }
            if layer_weights:
                layer_info["kernel"] = layer_weights[0].tolist()
                layer_info["bias"] = layer_weights[1].tolist()
            print(f"  {i}: Conv2D - filters={layer_config['filters']}, kernel={layer_config['kernel_size']}")
            
        elif layer_type == "Dense":
            layer_info["config"] = {
                "units": layer_config["units"],
                "activation": layer_config["activation"]
            }
            if layer_weights:
                layer_info["kernel"] = layer_weights[0].tolist()
                layer_info["bias"] = layer_weights[1].tolist()
            print(f"  {i}: Dense - units={layer_config['units']}")
            
        elif layer_type == "BatchNormalization":
            if layer_weights:
                layer_info["gamma"] = layer_weights[0].tolist()
                layer_info["beta"] = layer_weights[1].tolist()
                layer_info["moving_mean"] = layer_weights[2].tolist()
                layer_info["moving_var"] = layer_weights[3].tolist()
            layer_info["config"] = {"epsilon": layer_config.get("epsilon", 1e-5)}
            print(f"  {i}: BatchNormalization")
            
        elif layer_type == "MaxPooling2D":
            layer_info["config"] = {
                "pool_size": layer_config["pool_size"],
                "strides": layer_config.get("strides", layer_config["pool_size"])
            }
            print(f"  {i}: MaxPooling2D - pool_size={layer_config['pool_size']}")
            
        elif layer_type == "Dropout":
            layer_info["config"] = {"rate": layer_config["rate"]}
            print(f"  {i}: Dropout - rate={layer_config['rate']}")
            
        elif layer_type == "Flatten":
            print(f"  {i}: Flatten")
            
        elif layer_type == "InputLayer":
            print(f"  {i}: InputLayer")
            
        weights_data["layers"].append(layer_info)
    
    # Save to JSON
    output_file = OUTPUT_PATH / "cnn_model_full.json"
    print(f"\nSaving to: {output_file}")
    
    with open(output_file, "w") as f:
        json.dump(weights_data, f)
    
    # Get file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Also create a summary/config file for quick loading
    config_file = OUTPUT_PATH / "cnn_model_config.json"
    config = {
        "model_type": "cnn",
        "input_shape": [64, 64, 1],
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "preprocessing": {
            "resize": [64, 64],
            "grayscale": True,
            "normalize": True
        },
        "weights_file": "cnn_model_full.json"
    }
    
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_file}")
    
    print("\n✅ Export complete!")
    return weights_data

if __name__ == "__main__":
    export_model_weights()
