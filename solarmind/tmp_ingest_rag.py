import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag.ingest import ingest_knowledge_base
print(f"Ingested {ingest_knowledge_base()} chunks.")
