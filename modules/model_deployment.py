"""
Model Deployment Module for AI Data Science Copilot.

This module provides model serialization and FastAPI service generation
for deploying trained machine learning models.
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDeployer:
    """
    A class to handle model deployment operations.
    
    Provides functionality for saving models, generating API services,
    and creating deployment artifacts.
    """
    
    def __init__(self, output_dir: str = 'saved_models'):
        """
        Initialize the Model Deployer.
        
        Args:
            output_dir: Directory to save models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.saved_models = {}
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        preprocessor: Optional[Any] = None,
        metadata: Optional[Dict] = None,
        file_format: str = 'pickle'
    ) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model object
            model_name: Name for the saved model
            preprocessor: Optional preprocessor object
            metadata: Optional metadata dictionary
            file_format: Format to save ('pickle' or 'joblib')
            
        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}"
        
        model_data = {
            'model': model,
            'preprocessor': preprocessor,
            'metadata': metadata or {},
            'saved_at': timestamp
        }
        
        if file_format == 'pickle':
            filepath = os.path.join(self.output_dir, f"{filename}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        elif file_format == 'joblib':
            try:
                import joblib
                filepath = os.path.join(self.output_dir, f"{filename}.joblib")
                joblib.dump(model_data, filepath)
            except ImportError:
                logger.warning("joblib not available, using pickle instead")
                filepath = os.path.join(self.output_dir, f"{filename}.pkl")
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        self.saved_models[model_name] = filepath
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            Dictionary with model data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        if filepath.endswith('.joblib'):
            try:
                import joblib
                model_data = joblib.load(filepath)
            except ImportError:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")
        return model_data
    
    def generate_fastapi_service(
        self,
        model_path: str,
        feature_names: List[str],
        output_dir: str = 'api',
        port: int = 8000
    ) -> str:
        """
        Generate a FastAPI prediction service.
        
        Args:
            model_path: Path to saved model file
            feature_names: List of feature names
            output_dir: Directory to save API files
            port: Port for the API service
            
        Returns:
            Path to generated API directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate main.py
        main_py_content = self._generate_main_py(model_path, feature_names, port)
        main_py_path = os.path.join(output_dir, 'main.py')
        with open(main_py_path, 'w') as f:
            f.write(main_py_content)
        
        # Generate requirements.txt
        requirements_content = """fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
"""
        requirements_path = os.path.join(output_dir, 'requirements.txt')
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        # Generate Dockerfile (optional)
        dockerfile_content = self._generate_dockerfile(port)
        dockerfile_path = os.path.join(output_dir, 'Dockerfile')
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Generate README
        readme_content = self._generate_api_readme(port)
        readme_path = os.path.join(output_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"FastAPI service generated in {output_dir}")
        return output_dir
    
    def _generate_main_py(
        self,
        model_path: str,
        feature_names: List[str],
        port: int
    ) -> str:
        """
        Generate the main.py content for FastAPI.
        
        Args:
            model_path: Path to saved model
            feature_names: List of feature names
            port: API port
            
        Returns:
            main.py content as string
        """
        feature_schema = ',\n        '.join([f'{name}: float' for name in feature_names[:10]])  # Limit for example
        
        content = f'''"""
FastAPI Prediction Service
Auto-generated by AI Data Science Copilot
"""

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(
    title="ML Prediction API",
    description="Auto-generated prediction service",
    version="1.0.0"
)

# Load model
MODEL_PATH = "{model_path}"

class ModelLoader:
    """Singleton for loading and caching the model."""
    _instance = None
    model = None
    preprocessor = None
    metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._load_model()
        return cls._instance
    
    @classmethod
    def _load_model(cls):
        """Load the model from disk."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {{MODEL_PATH}}")
        
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        cls.model = model_data['model']
        cls.preprocessor = model_data.get('preprocessor')
        cls.metadata = model_data.get('metadata', {{}})
        print(f"Model loaded successfully from {{MODEL_PATH}}")

# Initialize model loader
model_loader = ModelLoader()

# Define input schema
class PredictionInput(BaseModel):
    """Input schema for predictions."""
{chr(10).join([f'    {name}: float = Field(..., description="Feature: {name}")' for name in feature_names[:10]])}
    
    class Config:
        json_schema_extra = {{
            "example": {{
{chr(10).join([f'                "{name}": 0.0,' for name in feature_names[:5]])}
            }}
        }}

# Define batch input schema
class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    inputs: List[Dict[str, float]]

# Define output schema
class PredictionOutput(BaseModel):
    """Output schema for predictions."""
    prediction: Any
    probability: Optional[float] = None
    model_name: str = "trained_model"
    version: str = "1.0.0"

@app.get("/")
def read_root():
    """Root endpoint."""
    return {{
        "message": "ML Prediction API",
        "status": "active",
        "model_loaded": model_loader.model is not None,
        "metadata": model_loader.metadata
    }}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {{
        "status": "healthy",
        "model_loaded": model_loader.model is not None
    }}

@app.get("/model_info")
def get_model_info():
    """Get model information."""
    return {{
        "metadata": model_loader.metadata,
        "feature_names": {feature_names}
    }}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Make a single prediction.
    
    Args:
        input_data: Input features
        
    Returns:
        Prediction result
    """
    try:
        # Convert input to array
        features = np.array([[getattr(input_data, name) for name in {feature_names[:10]}]])
        
        # Apply preprocessor if available
        if model_loader.preprocessor:
            features = model_loader.preprocessor.transform(features)
        
        # Make prediction
        prediction = model_loader.model.predict(features)
        
        # Get probability if available
        probability = None
        if hasattr(model_loader.model, 'predict_proba'):
            proba = model_loader.model.predict_proba(features)
            probability = float(np.max(proba))
        
        return PredictionOutput(
            prediction=prediction[0] if len(prediction) == 1 else prediction.tolist(),
            probability=probability
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(input_data: BatchPredictionInput):
    """
    Make batch predictions.
    
    Args:
        input_data: Batch of input features
        
    Returns:
        List of predictions
    """
    try:
        # Convert inputs to array
        features_list = []
        for item in input_data.inputs:
            features_list.append([item.get(name, 0.0) for name in {feature_names}])
        
        features = np.array(features_list)
        
        # Apply preprocessor if available
        if model_loader.preprocessor:
            features = model_loader.preprocessor.transform(features)
        
        # Make predictions
        predictions = model_loader.model.predict(features)
        
        return {{
            "predictions": predictions.tolist(),
            "count": len(predictions)
        }}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
'''
        return content
    
    def _generate_dockerfile(self, port: int) -> str:
        """
        Generate Dockerfile content.
        
        Args:
            port: API port
            
        Returns:
            Dockerfile content as string
        """
        return f'''FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application
COPY . .

# Expose port
EXPOSE {port}

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]
'''
    
    def _generate_api_readme(self, port: int) -> str:
        """
        Generate README for the API.
        
        Args:
            port: API port
            
        Returns:
            README content as string
        """
        return f'''# ML Prediction API

Auto-generated FastAPI prediction service.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn main:app --reload --port {port}
```

Or use Docker:
```bash
docker build -t ml-api .
docker run -p {port}:{port} ml-api
```

## API Endpoints

### Health Check
```
GET /health
```

### Model Info
```
GET /model_info
```

### Single Prediction
```
POST /predict
Content-Type: application/json

{{
    "feature1": 1.0,
    "feature2": 2.0,
    ...
}}
```

### Batch Prediction
```
POST /predict_batch
Content-Type: application/json

{{
    "inputs": [
        {{"feature1": 1.0, "feature2": 2.0}},
        {{"feature1": 3.0, "feature2": 4.0}}
    ]
}}
```

### Interactive Documentation
Visit: http://localhost:{port}/docs

## Testing

Using curl:
```bash
curl -X POST "http://localhost:{port}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"feature1": 1.0, "feature2": 2.0}}'
```
'''
    
    def export_model_summary(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        feature_importance: Optional[pd.DataFrame] = None,
        output_path: str = 'model_summary.json'
    ) -> str:
        """
        Export a JSON summary of the model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            metrics: Performance metrics
            feature_importance: Feature importance DataFrame
            output_path: Path to save summary
            
        Returns:
            Path to saved summary file
        """
        summary = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'export_time': datetime.now().isoformat(),
            'metrics': metrics,
            'model_params': model.get_params() if hasattr(model, 'get_params') else {},
        }
        
        if feature_importance is not None:
            summary['feature_importance'] = feature_importance.head(20).to_dict(orient='records')
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Model summary exported to {output_path}")
        return output_path
    
    def list_saved_models(self) -> List[Dict[str, str]]:
        """
        List all saved models in the output directory.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for filename in os.listdir(self.output_dir):
            if filename.endswith(('.pkl', '.joblib')):
                filepath = os.path.join(self.output_dir, filename)
                models.append({
                    'filename': filename,
                    'path': filepath,
                    'created': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat(),
                    'size_mb': round(os.path.getsize(filepath) / (1024 * 1024), 2)
                })
        
        return sorted(models, key=lambda x: x['created'], reverse=True)


def deploy_model(
    model: Any,
    model_name: str,
    feature_names: List[str],
    preprocessor: Optional[Any] = None,
    metadata: Optional[Dict] = None,
    output_dir: str = 'deployment'
) -> Dict[str, str]:
    """
    Complete model deployment workflow.
    
    Args:
        model: Trained model
        model_name: Name of the model
        feature_names: List of feature names
        preprocessor: Optional preprocessor
        metadata: Optional metadata
        output_dir: Output directory
        
    Returns:
        Dictionary with deployment paths
    """
    deployer = ModelDeployer(output_dir=os.path.join(output_dir, 'models'))
    
    # Save model
    model_path = deployer.save_model(
        model=model,
        model_name=model_name,
        preprocessor=preprocessor,
        metadata=metadata
    )
    
    # Generate API
    api_dir = os.path.join(output_dir, 'api')
    deployer.generate_fastapi_service(
        model_path=model_path,
        feature_names=feature_names,
        output_dir=api_dir
    )
    
    return {
        'model_path': model_path,
        'api_directory': api_dir
    }