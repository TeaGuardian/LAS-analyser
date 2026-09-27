from .base import Tree
from .compare import TreeComparator, TreeStatsComparator
from .generator import LasGenerator
from .CrownAxisParser import CrownAxisParser
from .LayerStackParser import LayerStackParser
from .SmartFusionParser import SmartFusionParser
from .ChmWatershedParser import ChmWatershedParser
from .AdaptiveFusionParser import AdaptiveFusionParser
from .CrownAxisStrictParser import CrownAxisStrictParser

PARSERS = {"CROWN-AXIS": CrownAxisParser,
           "CHM WATERSHED": ChmWatershedParser,
           "LAYER STACK": LayerStackParser,
           "SMART FUSION (V2)": SmartFusionParser,
           "ADAPTIVE FUSION (V17)": AdaptiveFusionParser,
           "CROWN-AXIS STRICT": CrownAxisStrictParser
            }
