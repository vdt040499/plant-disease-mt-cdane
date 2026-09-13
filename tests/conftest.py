"""Put the repository root on sys.path so `plantuda` imports without installing."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
