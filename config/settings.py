import os
from typing import Dict, List

# Project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output directories
INPUT_DIR = os.path.join(ROOT_DIR, 'input')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
PDF_DIR = 'G:/Mi unidad/prrrrrrrimo/papers/'
METADATA_FILE = os.path.join(INPUT_DIR, '20240717_metadata_corpus.xlsx')

# Resource usage settings
MAX_CPU_PERCENT = 75  # Maximum CPU usage percentage
MAX_MEMORY_PERCENT = 75  # Maximum memory usage percentage
MAX_GPU_PERCENT = 90  # Maximum GPU usage percentage

# PDF processing settings
PDF_BATCH_SIZE = 16  # Number of PDFs to process in a batch

# NLP settings
MAX_SEQUENCE_LENGTH = 512  # Maximum sequence length for NLP models
NLP_BATCH_SIZE = 32  # Batch size for NLP processing

# Topic modeling settings
NUM_TOPICS = 10  # Number of topics for topic modeling
TOPIC_COHERENCE_MEASURE = 'c_v'  # Coherence measure for topic modeling

# Network analysis settings
MIN_EDGE_WEIGHT = 2  # Minimum edge weight for network visualization

# Visualization settings
PLOT_WIDTH = 1600  # Width of plots in pixels
PLOT_HEIGHT = 1200  # Height of plots in pixels

CUSTOM_STOP_WORDS = {
    # Sneaky words
    'of', 'many', 'must', 'new', 'york', 'one', 'two', 'three', 'four', 'five', 
    'six', 'seven', 'eight', 'nine', 'ten', 'thu', 'they', 'this', 'these', 
    'way', 'may', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 
    'august', 'september', 'october', 'november', 'december', 'rather', 'well', 
    'used', 'use', 'example', 'first', 'second', 'third', 'part', 'see', 'thus',
    'set', 'term', 'found', 'mean', 'given', 'number', 'point', 'using', 'effect',
    'suggest', 'although', 'often', 'particular', 'include', 'based', 'within',
    'italy'
    # Single letters and numbers
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    
    # Common academic words that don't add meaning
    'et', 'al', 'fig', 'table', 'paper', 'study', 'research', 'analysis',
    'data', 'method', 'results', 'finding', 'findings', 'found', 'showed',
    'shown', 'show', 'shows', 'discussed', 'discussion', 'conclude', 'concludes',
    'conclusion', 'conclusions', 'introduction', 'methodology', 'abstract',
    'keywords', 'key', 'words', 'word', 'page', 'pages', 'vol', 'volume',
    'chapter', 'section', 'appendix', 'doi', 'isbn', 'issn', 'journal',
    
    # Common prepositions and conjunctions
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'up', 'about',
    'into', 'over', 'after', 'beneath', 'under', 'above', 'and', 'or',
    'but', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    
    # Common verbs and their forms
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'done', 'doing', 'can', 'could', 'will',
    'would', 'shall', 'should', 'may', 'might', 'must', 'take', 'takes',
    'took', 'taken', 'taking', 'make', 'makes', 'made', 'making',
    
    # Academic-specific words
    'literature', 'review', 'theory', 'theoretical', 'framework', 'model',
    'approach', 'approaches', 'proposed', 'presents', 'presented', 'present',
    'demonstrates', 'demonstrated', 'demonstrate', 'suggests', 'suggested',
    'suggest', 'indicates', 'indicated', 'indicate', 'implies', 'implied',
    'imply', 'notes', 'noted', 'note',
    
    # Units and measurements
    'mm', 'cm', 'km', 'kg', 'mg', 'ml', 'pct', 'percent', 'percentage',
    
    # Common abbreviations
    'ie', 'eg', 'etc', 'vs', 'cf', 'pp', 'ca', 'ibid', 'nb',

    # et al
    'et al',
    
    # Others
    'also', 'however', 'therefore', 'thus', 'hence', 'moreover',
    'furthermore', 'additionally', 'consequently', 'nevertheless',
    'meanwhile', 'nonetheless', 'whereas', 'while', 'despite',
    'although', 'though', 'even', 'still', 'yet', 'otherwise',
    'accordingly', 'indeed', 'namely', 'specifically', 'particularly',
    'especially', 'significantly', 'substantially', 'relatively',
    'approximately', 'roughly', 'about', 'around', 'nearly', 'almost',
    'aa', 'aaaj', 'ab', 'abce', 'abi', 'abu', 'ac', 'acad', 'acc', 'accr', 
    'acfe', 'acp', 'act', 'ad', 'addi', 'adm', 'adv', 'ae', 'aes', 'af', 'ag', 
    'age', 'ago', 'ah', 'ai', 'aid', 'aj', 'ak', 'ali', 'alm', 'als', 'amr', 
    'ana', 'ant', 'ao', 'aos', 'ap', 'appl', 'ar', 'ard', 'ate', 'att', 
    'attributionnoncommercialnoderivatives', 'au', 'aus', 'av', 'ave', 
    'avey', 'aw', 'ax', 'ayu', 'az', 'ba', 'bai', 'bal', 'bam', 'bar', 
    'bb', 'bbr', 'bc', 'bds', 'bed', 'beha', 'ben', 'ber', 'bh', 'big', 
    'birn', 'bit', 'bj', 'bk', 'bl', 'ble', 'bls', 'bmc', 'bo', 'bol', 
    'box', 'bp', 'bpo', 'br', 'brm', 'brno', 'brp', 'bs', 'bse', 'bu', 
    'bui', 'cai', 'cal', 'cam', 'cao', 'cap', 'cas', 'cat', 'cb', 'cbn', 
    'cc', 'cd', 'ce', 'cen', 'ceo', 'cep', 'cer', 'ces', 'cfa', 'cfc', 
    'cfi', 'cfl', 'cfo', 'cfr', 'cg', 'ch', 'chi', 'cho', 'ci', 'cic', 
    'cir', 'cit', 'cj', 'cmb', 'co', 'cob', 'cog', 'col', 'con', 'coo', 
    'cop', 'cor', 'cox', 'cp', 'cpa', 'cpt', 'cr', 'cre', 
    'creativecommonsorglicensesby', 'crg', 'cri', 'cs', 'csi', 'csr', 
    'cst', 'ct', 'cul', 'cuo', 'cur', 'cus', 'cut', 'cw', 'cy', 'czk', 
    'da', 'daf', 'dai', 'dal', 'das', 'day', 'daz', 'dc', 'dcs', 'ddd', 
    'de', 'dec', 'ded', 'del', 'den', 'dep', 'der', 'des', 'det', 'dev', 
    'dey', 'df', 'dg', 'dh', 'di', 'dia', 'die', 'dif', 'dis', 'dit', 'diw', 
    'dj', 'dl', 'dla', 'dm', 'dna', 'doc', 'dom', 'dos', 'dow', 'doz', 'dp', 
    'dr', 'dry', 'ds', 'dss', 'dt', 'du', 'due', 'duh', 'dun', 'dur', 'dvt', 
    'dw', 'dz', 'ea', 'eas', 'eba', 'ebc', 'ec', 'ecb', 'eco', 'ed', 'edn', 
    'edp', 'eds', 'ee', 'eec', 'ees', 'ef', 'efa', 'eft', 'egy', 'ei', 'ej', 
    'ek', 'el', 'eld', 'ele', 'em', 'en', 'end', 'ent', 'ep', 'epl', 'eq', 
    'er', 'era', 'erm', 'erp', 'ers', 'es', 'esg', 'est', 'ets', 'ety', 'eu', 
    'eur', 'eva', 'evi', 'ew', 'ewa', 'ex', 'eye', 'fa', 'fac', 'fam', 'far', 
    'fax', 'fc', 'fdg', 'fe', 'fee', 'fer', 'ff', 'ffs', 'fg', 'fgd', 'fh', 
    'fi', 'fim', 'fin', 'fit', 'fl', 'fm', 'fms', 'foi', 'fol', 'fp', 'fr', 
    'fra', 'fre', 'fs', 'fsb', 'fte', 'fu', 'ful', 'fw', 'ga', 'gad', 'gal', 
    'gas', 'gb', 'gc', 'gd', 'ge', 'gee', 'gen', 'ges', 'get', 'gfi', 'ghg', 
    'gic', 'gm', 'go', 'goi', 'got', 'gp', 'gr', 'gri', 'gs', 'gt', 'gtm', 
    'gul', 'guo', 'ha', 'han', 'har', 'hb', 'hc', 'hcm', 'hco', 'hd', 'hdr', 
    'hh', 'hi', 'hit', 'hj', 'ho', 'hoc', 'hopf', 'hou', 'howe', 'hr', 'hra', 
    'hrm', 'hs', 'hse', 'hsu', 'httpcreativecommonsorglicen', 
    'httpcreativecommonsorglicencesbylegalcode', 
    'httpcreativecommonsorglicensesby', 'httpcreativecommonsorglicensesbyncnd',
    'httpisapsejmgovplisapnsfdocdetailsxspidwdu', 'httpsdoiorgjmar', 
    'httpsmzlemanagersamazonawscomedffecaf', 'httpswwwemeraldcominsighthtm',
    'httpswwwmdpicomjournalsustainability',
    'httpswwwtandfonlinecomactionjournalinformationjournalcoderero', 
    'hub', 'hum', 'ib', 'ic', 'ical', 'icr', 'icrs', 'ics', 'ict', 'icts',
    'id', 'ido', 'ies', 'ifj', 'ii', 'iid', 'iii', 'ij', 'il', 'im', 'imm', 
    'inc', 'ind', 'ine', 'inf', 'ing', 'inn', 'int', 'inu', 'inv', 'ior', 'ip', 
    'ipj', 'ipl', 'ir', 'irm', 'ism', 'ist', 'ity', 'iut', 'iv', 'ive', 'ize', 
    'ja', 'jai', 'jan', 'jb', 'jc', 'jd', 'je', 'jel', 'jf', 'jg', 'jh', 'jia', 
    'jit', 'jj', 'jl', 'jm', 'joo', 'jos', 'jp', 'jr', 'js', 'jt', 'jw', 'jx', 
    'jy', 'ka', 'kan', 'kd', 'ke', 'ket', 'kh', 'kk', 'kl', 'ko', 'ks', 'kw', 
    'la', 'lai', 'lar', 'las', 'lau', 'lb', 'lc', 'ld', 'le', 'lev', 'li', 
    'lie', 'lim', 'lin', 'liu', 'lk', 'lle', 'lm', 'lo', 'los', 'lot', 'lpt', 
    'lr', 'lt', 'ltd', 'lts', 'lu', 'lui', 'luo', 'luu', 'luz', 'lvrt', 'mab', 
    'mac', 'mak', 'mal', 'man', 'marn', 'mb', 'mc', 'mct', 'md', 'mea', 'meas',
    'men', 'ment', 'mer', 'mf', 'mh', 'mi', 'mia', 'mio', 'mit', 'mix', 'mj', 
    'mk', 'mktx', 'mln', 'mncs', 'mne', 'mod', 'mot', 'moti', 'mp', 'ms', 'mt', 
    'mud', 'na', 'nal', 'nan', 'nas', 'nd', 'nds', 'ne', 'nec', 'neg', 'net', 
    'nff', 'nfi', 'ng', 'ngo', 'ngos', 'nhs', 'ni', 'nies', 'nj', 'nk', 'nl', 
    'nm', 'nma', 'non', 'nos', 'nou', 'np', 'npd', 'npm', 'npo', 'npos', 'npv', 
    'nr', 'ns', 'nsi', 'num', 'ny', 'oce', 'och', 'oct', 'ods', 'oer', 'og', 
    'ogy', 'ok', 'ol', 'ols', 'omj', 'oor', 'op', 'opt', 'oro', 'ory', 'os', 
    'ot', 'ou', 'ous', 'ow', 'ows', 'pa', 'pac', 'pan', 'par', 'pay', 'pc', 
    'pd', 'pdf', 'pe', 'pee', 'per', 'permissionsemeraldinsightcom', 'pet', 
    'peu', 'ph', 'phd', 'pi', 'pis', 'pj', 'pjh', 'ple', 'plg', 'pls', 'pm', 
    'pme', 'pmi', 'pn', 'po', 'por', 'pos', 'poz', 'ppe', 'ppp', 'pr', 'pre', 
    'pri', 'prl', 'pro', 'prs', 'ps', 'pso', 'psy', 'pt', 'pu', 'pub', 'pui', 
    'pun', 'pur', 'put', 'pw', 'qc', 'qca', 'qin', 'qr', 'qty', 'que', 'quo', 
    'ra', 'rad', 'ran', 'rao', 'ray', 'rbv', 'rc', 'rd', 'rea', 'red', 'reg', 
    'rel', 'rep', 'res', 'researchekonomska', 'rev', 'rf', 'rfid', 'rfp', 'rg', 
    'rh', 'rho', 'rib', 'rid', 'rio', 'rj', 'rk', 'rl', 'rle', 'rm', 'rn', 
    'roa', 'roe', 'roi', 'ros', 'row', 'roy', 'rp', 'rpe', 'rpi', 'rq', 'rqs',
    'rr', 'rs', 'rst', 'rt', 'rw', 'sa', 'sat', 'sb', 'sc', 'scf', 'scg', 
    'schn', 'schrder', 'sdt', 'se', 'sect', 'sef', 'sen', 'sf', 'sfa', 'sfb',
    'sg', 'sh', 'sj', 'sjc', 'sk', 'sl', 'sm', 'sn', 'soe', 'sor', 'sp', 'spe',
    'spl', 'sr', 'srmr', 'ss', 'ssm', 'ssrn', 'st', 'sta', 'sz', 'taipaleenmki',
    'tbx', 'tdr', 'techx', 'ter', 'tg', 'th', 'theo', 'ther', 'tian', 
    'ticipation', 'tify', 'tikkamki', 'tingencybased', 'tion', 
    'tiotechnological', 'tj', 'tk', 'tl', 'tlcharg', 'tm', 'tms', 'tmt', 'tn',
    'tor', 'tr', 'trs', 'ts', 'tw', 'typhlotechnological', 'ug', 'ulti', 'un',
    'une', 'universit', 'universityindustry', 'universityinternal', 'ure', 
    'urz', 'usar', 'uso', 'usp', 'ver', 'vey', 'vhmaa', 'vi', 'vii', 'viii', 
    'vk', 'vsternorrland', 'vt', 'vts', 'vw', 'wa', 'wc', 'wcc', 'wccs', 
    'welldesigned', 'welldeveloped', 'wellestablished', 'wellstructured', 
    'wf', 'wfh', 'wfhchange', 'wg', 'wh', 'whistleblowing', 'wickramasinghe', 
    'wm', 'woerd', 'wr', 'writingoriginal', 'writingreview', 'wtenweerde', 'ww', 
    'wwwcairninfo', 'wwwelseviercomlocateaos', 
    'wwwemeraldgrouppublishingcomlicensingreprintshtm', 'wwwfrontiersinorg', 
    'wwwrichtmannorg', 'wwwtandfonlinecomjournalsrero', 'xx', 'xxx', 'xxxx', 
    'yes', 'yigitbasioglu', 'yy', 'za', 'zoni'

}

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