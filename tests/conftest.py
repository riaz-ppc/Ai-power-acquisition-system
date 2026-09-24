import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# src.db reads DATABASE_URL at import time; tests never open a real
# connection, but the module must import without a .env present (CI).
os.environ.setdefault("DATABASE_URL", "postgresql://test@localhost/unused")
