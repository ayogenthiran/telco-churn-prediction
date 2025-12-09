from pathlib import Path
from pydantic_settings import BaseSettings

def get_project_root() -> Path:
    """Get the project root directory, works from notebooks and scripts."""
    # Try using __file__ first (works when imported as a module)
    try:
        config_file = Path(__file__).resolve()
        project_root = config_file.parent.parent
        # Verify it's the project root by checking for pyproject.toml
        if (project_root / "pyproject.toml").exists():
            return project_root.resolve()
    except (NameError, AttributeError):
        pass
    
    # If __file__ approach didn't work, search from current directory
    # This handles notebook contexts where __file__ might not be available
    current = Path.cwd()
    # Check current directory and all parents
    for path in [current] + list(current.parents):
        if (path / "pyproject.toml").exists():
            return path.resolve()
    
    # If we still can't find it, try relative to common notebook locations
    # If we're in notebooks/, go up one level
    if current.name == "notebooks" and (current.parent / "pyproject.toml").exists():
        return current.parent.resolve()
    
    # Last resort: assume we're in project root or use current directory
    return current.resolve()

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "churn-prediction"
    ENVIRONMENT: str = "development"
    
    @property
    def BASE_DIR(self) -> Path:
        """Project base directory as absolute path."""
        return get_project_root().resolve()
    
    @property
    def DATA_PATH(self) -> Path:
        """Data directory as absolute path."""
        base_dir = get_project_root().resolve()
        # Ensure we create an absolute path by joining with absolute base
        data_path = Path(base_dir) / "data" / "raw"
        # Force absolute resolution
        abs_path = data_path.resolve()
        # Double-check it's absolute
        if not abs_path.is_absolute():
            abs_path = Path(base_dir).resolve() / "data" / "raw"
        return abs_path
    
    @property
    def PROCESSED_DATA_PATH(self) -> Path:
        """Processed data directory as absolute path."""
        base_dir = get_project_root().resolve()
        processed_path = base_dir / "data" / "processed"
        return processed_path.resolve()
    
    @property
    def MODELS_PATH(self) -> Path:
        """Models directory as absolute path."""
        base_dir = get_project_root().resolve()
        models_path = base_dir / "models"
        return models_path.resolve()
    
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
        extra = "ignore"  # Ignore extra fields (like DATA_PATH, MODELS_PATH which are properties)

settings = Settings()
