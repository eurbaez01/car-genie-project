import sys
import os

# Add project root to path so app.py and src/ can be found
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

# Change working directory to project root so relative data paths work
os.chdir(ROOT)

from app import app
