from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "churn-prediction"
    ENVIRONMENT: str = "development"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.resolve()
    DATA_PATH: Path = (BASE_DIR / "data" / "raw").resolve()
    PROCESSED_DATA_PATH: Path = (BASE_DIR / "data" / "processed").resolve()
    MODELS_PATH: Path = (BASE_DIR / "models").resolve()
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Model
    MODEL_VERSION: str = "v1"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
