from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
RAW_PICKLE = Path(os.environ.get("UC_RAW", ROOT / "grenades_by_match_0726.list"))
DATA = ROOT / "data"
OUT = ROOT / "out"
FIG = DATA / "figures"
TICK_RATE = 64
TEAM_T, TEAM_CT = 2, 3
TEAMFLASH_MIN_S = 1.0       # тиммейт ослеплён > 1 с — это тимфлеш
MIN_LINEUP_N = 30           # минимальный размер лайнапа
LINEUP_EPS = 120.0          # размер клетки сетки лайнапов (по точке броска и приземления), в юнитах карты
TF_PENALTY_S = 2.0          # один тимфлеш "стоит" 2 с ослепления врагов при ранжировании
N_ZONES = 24
SEED = 42
OPENROUTER_MODEL_DEFAULT = "deepseek/deepseek-v4.1-flash"
