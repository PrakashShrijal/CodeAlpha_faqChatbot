# tests/conftest.py
# Ensures the project root is on sys.path for all pytest sessions.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
