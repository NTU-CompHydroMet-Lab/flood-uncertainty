"""Fusion of aligned evidential predictions."""
from .evidence_sum import fuse_evidence_sum, dirichlet_binary_stats

__all__ = ['fuse_evidence_sum', 'dirichlet_binary_stats']
