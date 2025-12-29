import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from pathlib import Path
from typing import Optional

class ModelLoader:
    """
    Handles loading of both DermaAI models with automatic download support
    """
    
    def __init__(self, model_dir: str = "app/model"):
        """
        Initialize model loader
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model1_path = self.model_dir / "DermaAI.keras"
        self.model2_path = self.model_dir / "efficientnetv2s.h5"
    
    def load_model1(self) -> tf.keras.Model:
        """
        Load DermaAI model (Model 1)
        
        Returns:
            Loaded Keras model
        """
        print(f"📦 Loading Model 1 (DermaAI) from {self.model1_path}...")
        
        if not self.model1_path.exists():
            raise FileNotFoundError(
                f"Model 1 not found at {self.model1_path}. "
                "Please ensure DermaAI.keras is in the app/model/ directory."
            )
        
        try:
            model = load_model(str(self.model1_path))
            print(f"✅ Model 1 loaded successfully!")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load Model 1: {str(e)}")
    
    def load_model2(self) -> tf.keras.Model:
        """
        Load Skin Cancer Classifier (Model 2)
        
        Returns:
            Loaded Keras model
        """
        print(f"📦 Loading Model 2 (Cancer Classifier) from {self.model2_path}...")
        
        if not self.model2_path.exists():
            print("⚠️  Model 2 not found locally. Attempting to download...")
            self._download_model2()
        
        try:
            # Custom layer for compatibility
            class CustomDepthwiseConv2D(DepthwiseConv2D):
                @classmethod
                def from_config(cls, config):
                    if 'groups' in config and config['groups'] == 1:
                        del config['groups']
                    return super().from_config(config)
            
            model = tf.keras.models.load_model(
                str(self.model2_path),
                custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
            )
            
            print(f"✅ Model 2 loaded successfully!")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Model 2: {str(e)}")
    
    def _download_model2(self):
        """
        Download Model 2 from Hugging Face if not available locally
        """
        try:
            from huggingface_hub import hf_hub_download
            
            print("🔄 Downloading model from Hugging Face...")
            
            # Download from Hugging Face Hub
            # Replace with your actual Hugging Face model repository
            downloaded_path = hf_hub_download(
                repo_id="YOUR_USERNAME/skin-cancer-classifier",  # Update this
                filename="efficientnetv2s.h5",
                cache_dir=str(self.model_dir)
            )
            
            # Move to correct location
            import shutil
            shutil.move(downloaded_path, self.model2_path)
            
            print("✅ Model 2 downloaded successfully!")
            
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Model 2: {str(e)}. "
                "Please manually place efficientnetv2s.h5 in app/model/ directory."
            )
    
    def verify_models(self) -> dict:
        """
        Verify that both models are available
        
        Returns:
            Dictionary with model availability status
        """
        return {
            "model1_exists": self.model1_path.exists(),
            "model2_exists": self.model2_path.exists(),
            "model1_path": str(self.model1_path),
            "model2_path": str(self.model2_path)
        }
    
    @staticmethod
    def get_model_info():
        """Get information about model requirements"""
        return {
            "model1": {
                "name": "DermaAI.keras",
                "format": "Keras HDF5",
                "size_approx": "~50-100 MB",
                "description": "Inflammatory and infectious skin diseases"
            },
            "model2": {
                "name": "efficientnetv2s.h5",
                "format": "Keras HDF5",
                "size_approx": "~80 MB",
                "description": "Cancer and precancerous lesions",
                "temperature_scaling": 2.77
            }
        }