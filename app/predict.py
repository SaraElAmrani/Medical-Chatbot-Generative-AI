import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from typing import Dict, Tuple, Optional
import io

class DermaAIPredictor:
    """
    Intelligent predictor with sequential routing logic
    """
    
    def __init__(self, model1: tf.keras.Model, model2: tf.keras.Model):
        """
        Initialize predictor with both models
        
        Args:
            model1: DermaAI model (inflammatory/infectious diseases)
            model2: Cancer classifier model
        """
        self.model1 = model1
        self.model2 = model2
        
        # Model 1 classes
        self.model1_classes = [
            'Atopic Dermatitis',
            'Eczema', 
            'Psoriasis',
            'Seborrheic Keratoses',
            'Tinea Ringworm Candidiasis'
        ]
        
        # Model 2 classes
        self.model2_classes = {
            0: "AKIEC",
            1: "BCC",
            2: "BKL",
            3: "DF",
            4: "MEL",
            5: "NV",
            6: "VASC"
        }
        
        self.model2_full_names = {
            "AKIEC": "Actinic Keratoses and Intraepithelial Carcinoma",
            "BCC": "Basal Cell Carcinoma",
            "BKL": "Benign Keratosis-like Lesions",
            "DF": "Dermatofibroma",
            "MEL": "Melanoma",
            "NV": "Melanocytic Nevi",
            "VASC": "Vascular Lesions"
        }
        
        self.temperature = 2.77
        
        # Cancer detection keywords
        self.cancer_keywords = [
            'cancer', 'cancéreux', 'cancéreuse', 'melanoma', 'mélanome',
            'carcinoma', 'carcinome', 'tumeur', 'tumor', 'malin', 'maligne',
            'métastase', 'biopsie', 'biopsy', 'oncologie', 'skin cancer',
            'akiec', 'bcc', 'basal', 'squamous', 'lésion suspecte',
            'grain de beauté', 'mole', 'suspicious', 'precancerous',
            'précancéreux', 'keratosis', 'kératose', 'changement de couleur',
            'suspect', 'évolution', 'asymétrique', 'bordure irrégulière'
        ]
        
        # Risk levels
        self.risk_levels = {
            'Atopic Dermatitis': 'Faible',
            'Eczema': 'Faible',
            'Psoriasis': 'Modéré',
            'Seborrheic Keratoses': 'Faible',
            'Tinea Ringworm Candidiasis': 'Faible',
            'AKIEC': 'Élevé',
            'BCC': 'Élevé',
            'BKL': 'Faible',
            'DF': 'Faible',
            'MEL': 'Très Élevé',
            'NV': 'Faible',
            'VASC': 'Modéré'
        }
        
        # Medical recommendations
        self.recommendations = {
            'Atopic Dermatitis': "Consultez un dermatologue pour traitement hydratant et anti-inflammatoire.",
            'Eczema': "Hydratation régulière et éviter les irritants. Consultation dermatologique recommandée.",
            'Psoriasis': "Consultation dermatologique nécessaire pour traitement adapté (topique, photothérapie).",
            'Seborrheic Keratoses': "Lésion bénigne. Consultation si changement d'aspect ou gêne esthétique.",
            'Tinea Ringworm Candidiasis': "Traitement antifongique nécessaire. Consultation médicale recommandée.",
            'AKIEC': "⚠️ URGENT : Consultation dermatologique immédiate. Risque de cancer de la peau.",
            'BCC': "⚠️ URGENT : Consultation dermatologique immédiate. Cancer de la peau à traiter.",
            'BKL': "Lésion bénigne. Surveillance recommandée, consultation si évolution.",
            'DF': "Lésion bénigne. Aucun traitement nécessaire sauf gêne esthétique.",
            'MEL': "🚨 URGENCE MÉDICALE : Consultation oncologique immédiate. Mélanome suspecté.",
            'NV': "Grain de beauté bénin. Surveillance régulière recommandée.",
            'VASC': "Lésion vasculaire. Consultation dermatologique pour évaluation."
        }
        
        # Medical categories
        self.medical_categories = {
            'inflammatory': ['Atopic Dermatitis', 'Eczema', 'Psoriasis'],
            'infectious': ['Tinea Ringworm Candidiasis'],
            'benign_growth': ['Seborrheic Keratoses', 'BKL', 'NV', 'DF'],
            'precancerous': ['AKIEC'],
            'cancerous': ['BCC', 'MEL'],
            'vascular': ['VASC']
        }
    
    def detect_cancer_context(self, user_query: Optional[str] = None) -> bool:
        """
        Detect if the context requires cancer model
        
        Args:
            user_query: User's question or context
            
        Returns:
            True if cancer context detected
        """
        if user_query is None:
            return False
        
        query_lower = user_query.lower()
        return any(keyword in query_lower for keyword in self.cancer_keywords)
    
    def _preprocess_image_model1(self, img: Image.Image) -> np.ndarray:
        """Preprocess image for Model 1"""
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def _preprocess_image_model2(self, img: Image.Image) -> np.ndarray:
        """Preprocess image for Model 2"""
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = (img_array - 0.5) * 2
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def _predict_model1(self, img_array: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Predict with Model 1"""
        predictions = self.model1.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = self.model1_classes[predicted_idx]
        confidence = float(predictions[0][predicted_idx])
        
        all_preds = {
            self.model1_classes[i]: float(predictions[0][i])
            for i in range(len(self.model1_classes))
        }
        
        return predicted_class, confidence, all_preds
    
    def _predict_model2(self, img_array: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Predict with Model 2"""
        # Get logits
        logits_model = tf.keras.Model(
            inputs=self.model2.input,
            outputs=self.model2.layers[-2].output
        )
        logits = logits_model(img_array)
        
        # Apply final layer
        final_dense = self.model2.layers[-1]
        logits = final_dense(logits)
        
        # Temperature scaling
        scaled_logits = logits / self.temperature
        scaled_probs = tf.nn.softmax(scaled_logits).numpy()[0]
        
        predicted_idx = int(np.argmax(scaled_probs))
        short_name = self.model2_classes[predicted_idx]
        predicted_class = self.model2_full_names[short_name]
        confidence = float(scaled_probs[predicted_idx])
        
        all_preds = {
            self.model2_full_names[self.model2_classes[i]]: float(scaled_probs[i])
            for i in range(len(self.model2_classes))
        }
        
        return predicted_class, confidence, all_preds
    
    def _get_medical_category(self, disease: str) -> str:
        """Get medical category for a disease"""
        for category, diseases in self.medical_categories.items():
            if any(d in disease for d in diseases):
                return category
        return 'unknown'
    
    def predict_from_bytes(
        self,
        image_bytes: bytes,
        user_query: Optional[str] = None,
        force_cancer_model: bool = False
    ) -> Dict:
        """
        Predict from image bytes with intelligent routing
        
        Args:
            image_bytes: Raw image bytes
            user_query: Optional user context
            force_cancer_model: Force cancer model usage
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Determine which model to use
        if force_cancer_model:
            use_cancer_model = True
            cancer_context = True
        else:
            cancer_context = self.detect_cancer_context(user_query)
            use_cancer_model = cancer_context
        
        # Make prediction
        if use_cancer_model:
            img_array = self._preprocess_image_model2(img)
            disease, confidence, all_preds = self._predict_model2(img_array)
            model_used = "Skin Cancer Classifier (Model 2)"
        else:
            img_array = self._preprocess_image_model1(img)
            disease, confidence, all_preds = self._predict_model1(img_array)
            model_used = "DermaAI (Model 1)"
        
        # Get additional info
        disease_key = disease.split('(')[0].strip() if '(' in disease else disease
        risk_level = self.risk_levels.get(disease_key, 'Inconnu')
        medical_category = self._get_medical_category(disease)
        recommendation = self.recommendations.get(disease_key, "Consultation médicale recommandée.")
        
        # Top 3 predictions
        sorted_preds = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)[:3]
        top_predictions = [
            {"disease": d, "confidence": float(c)}
            for d, c in sorted_preds
        ]
        
        return {
            'disease': disease,
            'confidence': float(confidence),
            'risk_level': risk_level,
            'medical_category': medical_category,
            'recommendation': recommendation,
            'top_predictions': top_predictions,
            'model_used': model_used
        }
    
    def compare_both_models(self, image_bytes: bytes) -> Dict:
        """
        Compare predictions from both models
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with both model predictions
        """
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Model 1 prediction
        img_array1 = self._preprocess_image_model1(img)
        disease1, conf1, all1 = self._predict_model1(img_array1)
        
        top1 = sorted(all1.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Model 2 prediction
        img_array2 = self._preprocess_image_model2(img)
        disease2, conf2, all2 = self._predict_model2(img_array2)
        
        top2 = sorted(all2.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'model1': {
                'model_name': 'DermaAI (Inflammatory/Infectious)',
                'disease': disease1,
                'confidence': float(conf1),
                'risk_level': self.risk_levels.get(disease1, 'Inconnu'),
                'top_3': [{"disease": d, "confidence": float(c)} for d, c in top1]
            },
            'model2': {
                'model_name': 'Cancer Classifier',
                'disease': disease2,
                'confidence': float(conf2),
                'risk_level': self.risk_levels.get(disease2.split('(')[0].strip(), 'Inconnu'),
                'top_3': [{"disease": d, "confidence": float(c)} for d, c in top2]
            },
            'recommendation': (
                "For inflammatory/infectious conditions, use Model 1. "
                "For cancer-related concerns, use Model 2."
            )
        }