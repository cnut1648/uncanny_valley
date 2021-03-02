from pathlib import Path
from typing import List, Dict

import numpy as np

# {"folklore title": [text_path_i, ...]}
FolkLoreData = Dict[str, List[Path]]
# {"folklore title": folklore_emb}
FolkLoreEmb = Dict[str, np.ndarray]
# {"folklore title": {"folklore subtext": subtext_emb}}
FolkLoreEmbCoarse = Dict[str, FolkLoreEmb]
