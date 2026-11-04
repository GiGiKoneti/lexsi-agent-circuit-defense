"""
CuratorKIT Engine: Cohere native template formatting and contrastive triplet curation.
"""

from .template_formatter import CohereAgentFormatter
from .triplet_generator import CohereAgentTripletCurator

__all__ = ["CohereAgentFormatter", "CohereAgentTripletCurator"]
