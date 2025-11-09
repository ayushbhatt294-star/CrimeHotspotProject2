from typing import Optional, TYPE_CHECKING

# Ensure static type-checkers can resolve BaseSettings while still allowing a
# runtime fallback for environments that don't have pydantic-settings.
if TYPE_CHECKING:
    # for type checkers, prefer the pydantic BaseSettings symbol
    from pydantic import BaseSettings  # type: ignore
else:
    try:
        # pydantic v2 ships settings in a separate package
        from pydantic_settings import BaseSettings  # type: ignore
    except Exception:
        # fallback to pydantic v1-compatible BaseSettings
        from pydantic import BaseSettings  # type: ignore

# Ensure a concrete base class name for both runtime and static analysis
try:
    BaseSettingsLocal = BaseSettings
except NameError:
    # final fallback: import from pydantic directly
    from pydantic import BaseSettings as BaseSettingsLocal  # type: ignore
import os


class Settings(BaseSettingsLocal):  # type: ignore
    PROJECT_NAME: str = "Crime Hotspot Mapping API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    ROUTES_DIR: str = os.path.join(os.getcwd(), 'routes')
    DATA_DIR: str = os.path.join(os.getcwd(), 'data')
    MODEL_DIR: str = os.path.join(os.getcwd(), 'models')
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Caching settings
    CACHE_TTL: int = 3600  # 1 hour
    
    # Model settings
    MIN_SAMPLES_CLUSTERING: int = 3
    EPS_KM_CLUSTERING: float = 0.5
    DEFAULT_DAYS_BACK: int = 30
    
    class Config:
        case_sensitive = True

settings = Settings()