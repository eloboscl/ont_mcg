import os
from typing import List, Dict

# Project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output directories
INPUT_DIR = os.path.join(ROOT_DIR, 'input')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
PDF_DIR = 'G:/Mi unidad/prrrrrrrimo/PP2/'
METADATA_FILE = os.path.join(INPUT_DIR, '20240717_metadata_corpus.xlsx')

# Resource usage settings
MAX_CPU_PERCENT = 80  # Maximum CPU usage percentage
MAX_MEMORY_PERCENT = 75  # Maximum memory usage percentage
MAX_GPU_PERCENT = 90  # Maximum GPU usage percentage

# PDF processing settings
PDF_BATCH_SIZE = 100  # Number of PDFs to process in a batch

# NLP settings
MAX_SEQUENCE_LENGTH = 512  # Maximum sequence length for NLP models
NLP_BATCH_SIZE = 32  # Batch size for NLP processing

# Topic modeling settings
NUM_TOPICS = 10  # Number of topics for topic modeling
TOPIC_COHERENCE_MEASURE = 'c_v'  # Coherence measure for topic modeling

# Network analysis settings
MIN_EDGE_WEIGHT = 2  # Minimum edge weight for network visualization

# Visualization settings
PLOT_WIDTH = 1200  # Width of plots in pixels
PLOT_HEIGHT = 800  # Height of plots in pixels

# Management control terms
MANAGEMENT_CONTROL_TERMS: List[str] = [
        'strategic planning', 'enterprise strategy', 'corporate strategy','strategy',
        'management control', 'task control', 'control', 'control environment',
        'information system', 'feedback', 'feedforward', 'budgeting', 'internal control',
        'risk management', 'corporate governance', 'financial control', 'audit committee',
        'internal audit', 'BSC', 'balanced scorecard', 'financial perspective',
        'client perspective', 'resources perspective', 'process perspective', 'tableau de bord', 
        'kpi', 'key performance indicator', 'key performance indicators', 'strategy map', 
        'LOC', 'levers of control', 'management control system', 'management control systems',
        'accounting', 'accounting data', 'accounting systems', 'management accounting',
        'managerial accounting', 'management accounting system', 'controller',
        'performance measurement', 'performance management', 'indicator', 'measures',
        'measurement', 'measure', 'evaluation', 'reward', 'rewards', 'punishment', 'training', 
        'budget', 'beyond-budgeting', 'activity based costing', 'ABC', 'activity-based-costing',
        'lean', 'lean accounting', 'cost variance', 'indirect cost', 'direct cost',
        'activity-based-budgeting', 'activity based budgeting', 'strategy implementation',
        'evidence based', 'data', 'compensation', 'incentives', 'incetive scheme', 
        'alliances', 'strategic objectives', 'strategic initiatives', 'framework',
        'management control theory', 'management control framework', 'efficiency',
        'efficacy', 'effectiveness', 'profit center', 'revenue center', 'cost center',
        'investment center', 'responsability center', 'organizational structure', 'sbu',
        'strategic business unit', 'culture', 'artifacts', 'rituals', 'agency',
        'principal agent', 'asymmetric information', 'principal', 'agent', 'hierarchy', 
        'business analytics', 'decision making', 'control objectives', 'values',
        'corporate values', 'package', 'formal control', 'informal control',
        'behavior control', 'cost control', 'action control', 'behavioral control',
        'output control', 'inter organizational control', 'operational control',
        'organizational control', 'budget control', 'motivation control',
        'philosophy control', 'system control', 'leadership control', 'brand control',
        'cultural control', 'personel control', 'cybernetic control',
        'reward and compensation control', 'rewards and compensation controls',
        'planning control', 'clan control', 'market control', 'social control',
        'boundary control', 'organic control', 'sostenibility',
    ]

# Category list
CATEGORY_LIST: List[str] = [
    "Management", "Business", "Economics", "Public Administration", "Finance",
    "Accounting", "Operations Research", "Engineering", "Information Systems",
    "Psychology", "Sociology", "Education"
]

# Trend forecasting terms
TERMS_TO_FORECAST: List[str] = [
    'management control', 'performance measurement', 'balanced scorecard',
    'levers of control', 'bsc', 'management control system', 'control', 'loc',
]

# Logging configuration
LOGGING_CONFIG: Dict = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': os.path.join(OUTPUT_DIR, 'analysis.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        },
    }
}