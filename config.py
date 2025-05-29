# config.py
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to manage environment variables and application settings"""
    
    # Database Configuration
    DB_HOST = "3.148.234.227"
    DB_USER = "test1"
    DB_PASSWORD = "Test1234#"
    DB_NAME = 'literature_analytics_platform'
    
    # Analysis Options
    MONTHLY_OPTIONS = ["Raw", "Monthly % Change", "Rolling Average", "Growth Rate"]
    
    FIELD_OPTIONS = [
        "Computer Science", 
        "Mathematics", 
        "Physics", 
        "Economics",
        "Electrical Eng. & Systems Science",
        "Quantitative Biology"
    ]
    
    SECTOR_OPTIONS = ["Technology", "Financials", "Healthcare", "Energy", "Industrials"]
    
    # Field Mapping
    FIELD_MAP = {
        "Computer Science": "cs",
        "Mathematics": "math",
        "Physics": "physics",
        "Economics": "econ",
        "Electrical Eng. & Systems Science": "eess",
        "Quantitative Biology": "q-bio"
    }
    
    # Sector ETF Mapping
    SECTOR_ETFS = {
        "technology": "XLK",
        "healthcare": "XLV",
        "financials": "XLF",
        "energy": "XLE",
        "industrials": "XLI"
    }
    
    # Data Collection Parameters
    ARXIV_LOOKBACK_YEARS = 5
    MARKET_LOOKBACK_YEARS = 5
    
    # LDA Model Parameters
    LDA_NUM_TOPICS = 3
    LDA_NO_BELOW = 5
    LDA_NO_ABOVE = 0.5
    LDA_PASSES = 10
    LDA_ALPHA = 'auto'
    LDA_ETA = 'auto'
    
    @classmethod
    def validate(cls):
        """Validate that all required environment variables are set"""
        required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD']
        missing_vars = []
        
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
    
    @classmethod
    def get_db_url(cls, include_database=True):
        """Get the database URL for SQLAlchemy"""
        from sqlalchemy.engine import URL
        
        params = {
            "drivername": "mysql+pymysql",
            "host": cls.DB_HOST,
            "port": 3306,
            "username": cls.DB_USER,
            "password": cls.DB_PASSWORD
        }
        
        if include_database:
            params["database"] = cls.DB_NAME
            
        return URL.create(**params)