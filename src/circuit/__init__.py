"""
CircuitKIT Engine: Edge Attribution Patching with Integrated Gradients (EAP-IG) & SVD subspace discovery.
"""

from .eap_ig_tracer import CircuitDiscoveryEAPIG
from .subspace_extractor import SubspaceExtractor

__all__ = ["CircuitDiscoveryEAPIG", "SubspaceExtractor"]
