"""Subpackage for collaborative learning, where multiple clients trains a single
global model without sharing their local datasets.
"""

from .core import BaseClient, BaseServer  # noqa: F401
from .dsfl import DSFLAPI, DSFLClient, DSFLServer  # noqa : F401
from .fedavg import (  # noqa: F401
    FedAVGAPI,
    FedAVGClient,
    FedAVGServer,
    MPIFedAVGAPI,
    MPIFedAVGClientManager,
    MPIFedAVGServerManager,
)
from .fedexp import FedEXPServer  # noqa: F401
from .fedgems import FedGEMSAPI, FedGEMSClient, FedGEMSServer  # noqa: F401
from .fedkd import FedKDClient  # noqa :F401
from .fedmd import (  # noqa: F401
    FedMDAPI,
    FedMDClient,
    FedMDServer,
    MPIFedMDAPI,
    MPIFedMDClientManager,
    MPIFedMDServerManager,
)
from .moon import MOONClient  # noqa :F401
from .optimizer import AdamFLOptimizer, SGDFLOptimizer  # noqa: F401
from .splitnn import SplitNNAPI, SplitNNClient  # noqa: F401
