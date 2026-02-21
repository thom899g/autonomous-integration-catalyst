"""
Objective Interpreter & Strategy Generator
Core component of the Autonomous Integration Catalyst
Translates business objectives into integration requirements and hypotheses
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

# Third-party imports
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BusinessObjective:
    """Structured representation of a business objective"""
    id: str
    raw_text: str
    parsed_components: Dict[str, Any]
    metrics: List[Dict[str, Any]]
    system_touchpoints: List[str]
    priority: int  # 1-10 scale
    timeframe: str  # e.g., "Q4 2024", "30 days"
    created_at: datetime
    confidence_score: float  # 0-1 confidence in parsing accuracy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore storage"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class IntegrationHypothesis:
    """Generated integration hypothesis with impact scoring"""
    id: str
    objective_id: str
    description: str
    integration_pattern: str  # e.g., "real-time-sync", "batch-process", "event-driven"
    source_systems: List[str]
    target_systems: List[str]
    data_elements: List[str]
    expected_impact_score: float  # 0-100 scale
    implementation_complexity: int  # 1-10 scale
    confidence_level: float  # 0-1
    similar_patterns: List[str]  # IDs of similar historical patterns
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore storage"""
        data = asdict(self)
        data['generated_at'] = self.generated_at.isoformat()
        return data

class ObjectiveParser:
    """
    Main parser that translates natural language business objectives
    into structured integration requirements
    """
    
    def __init__(self, firebase_credential_path: Optional[str] = None):
        """
        Initialize the Objective Parser
        
        Args:
            firebase_credential_path: Path to Firebase service account JSON file.
                                     If None, will attempt to use default credentials.
        
        Raises:
            ValueError: If Firebase initialization fails
            ImportError: If required libraries are not available
        """
        logger.info("Initializing Objective Parser...")
        
        # Initialize NLP model with error handling
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy NLP model successfully")
        except OSError as e:
            logger.error(f"Failed to load spaCy model: {e}")
            logger.info("Attempting to download model...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Downloaded and loaded spaCy model")
            except Exception as download_error:
                logger.error(f"Failed to download spaCy model: {download_error}")
                raise ImportError("spaCy model 'en_core_web_sm' is required. Please install with: python -m spacy download en_core_web_sm")
        
        # Initialize Firebase
        self.firestore_client = None
        self._init_firebase(firebase_credential_path)
        
        # Initialize ML components
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.cluster_fitted = False
        
        # Predefined patterns and metrics
        self.metric_patterns = {
            'increase': r'increase\s+(\w+)\s+by\s+([\d\.]+)%?',
            'reduce': r'reduce\s+(\w+)\s+by\s+([\d\.]+)%?',
            'improve': r'improve\s+(\w+)\s+(?:by\s+)?([\d\.]+)%?',
            'decrease': r'decrease\s+(\w+)\s+by\s+([\d\.]+)%?'
        }
        
        self.system_keywords = {
            'ecommerce': ['cart', 'checkout', 'product', 'inventory', 'order'],
            'crm': ['customer', 'lead', 'contact', 'sales', 'opportunity'],
            'erp': ['invoice', 'payment', 'supplier', 'procurement', 'accounting'],
            'analytics': ['report', 'dashboard', 'metric', 'kpi', 'analytics'],
            'marketing': ['campaign', 'email', 'segment', 'conversion', 'engagement']
        }
        
        logger.info("Objective Parser initialized successfully")
    
    def _init_firebase(self, credential_path: Optional[str]) -> None:
        """
        Initialize Firebase connection with comprehensive error handling
        
        Args:
            credential_path: Path to service account JSON file
            
        Raises:
            ValueError: If Firebase initialization fails
        """
        try:
            if firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps:
                logger.info("Firebase app already initialized, using existing instance")
                app = firebase_admin.get_app()
            else:
                if credential_path:
                    cred = credentials.Certificate(credential_path)
                    logger.info(f"Loading Firebase credentials from: {credential_path}")
                else:
                    logger.info("Attempting to use default Firebase credentials")
                    cred = credentials.ApplicationDefault()
                
                app = firebase_admin.initialize_app(cred)
                logger.info("Firebase app initialized successfully")
            
            self.firestore_client = firestore.client(app)
            
            # Test connection
            test_ref = self.firestore_client.collection('_health_check').document('test')
            test_ref.set({'timestamp': firestore.SERVER_TIMESTAMP}, merge=True)
            test_ref.delete()
            logger.info("Firestore connection test successful")
            
        except FileNotFoundError as e:
            logger.error(f"Firebase credential file not found: {e}")
            raise ValueError(f"Firebase credential file not found at {credential_path}")
        except ValueError as e:
            logger.error(f"Invalid Firebase credentials: {e}")
            raise ValueError(f"Invalid Firebase credentials: {e}")
        except FirebaseError as e:
            logger.error(f"Firebase initialization error: {e}")
            raise ValueError(f"Failed to initialize Firebase: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Firebase initialization: {e}")
            raise ValueError(f"Unexpected error: {e}")
    
    def parse_objective(self, objective_text: str, priority: int = 5, 
                       timeframe: str = "30 days") -> BusinessObjective:
        """
        Parse a natural language business objective into structured components
        
        Args:
            objective_text: Raw business objective text
            priority: Priority level (1-10)
            timeframe: Expected timeframe for completion
            
        Returns:
            BusinessObjective: Structured objective representation
            
        Raises:
            ValueError: If objective_text is empty or invalid
        """
        # Input validation
        if not objective_text or not isinstance(objective_text, str):
            raise ValueError("objective_text must be a non-empty string")
        
        if not 1 <= priority <= 10:
            raise ValueError("priority must be between 1 and 10")
        
        logger.info(f"Parsing objective: {objective_text[:50]}...")
        
        try:
            # Generate unique ID
            objective_id = hashlib.md5(
                f"{objective_text}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            
            # Parse with spaCy
            doc = self.nlp(objective_text.lower())
            
            # Extract components
            parsed_components = self._extract_components(doc)
            metrics = self._extract_metrics(objective_text)
            system_touchpoints = self._identify_systems(objective_text)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                parsed_components,