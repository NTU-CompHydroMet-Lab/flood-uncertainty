"""Cumulative evidence fusion with one shared Dirichlet(1, 1) prior.

Inputs are raw nonnegative evidence, never normalized images or probabilities.
The class axis is first: (2, ...), ordered [non-water, water]. Neural evidence
is learned pseudo-counts; the rule does not establish modality independence
or guarantee calibrated uncertainty, especially under contradictory evidence.
"""
import numpy as np


def _validate(evidence):
    values = np.asarray(evidence, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != 2:
        raise ValueError('Expected class-first evidence (2, ...): non-water, water')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Evidence must be finite and nonnegative')
    return values


def fuse_evidence_sum(sar_evidence, optical_evidence):
    """Return raw fused evidence e_sar + e_optical; prior added in stats."""
    sar, optical = _validate(sar_evidence), _validate(optical_evidence)
    if sar.shape != optical.shape:
        raise ValueError('Paired evidence shapes must match; broadcasting is forbidden')
    return sar + optical


def dirichlet_binary_stats(evidence):
    """Binary Dirichlet mean, vacuity and variance decomposition."""
    evidence = _validate(evidence)
    alpha = evidence + 1
    strength = alpha.sum(axis=0)
    probability = alpha[1] / strength
    total_variance = probability * (1 - probability)
    epistemic = total_variance / (strength + 1)
    return {'evidence': evidence, 'alpha': alpha, 'strength': strength,
            'probability': probability, 'dst_u': 2 / strength,
            'epistemic': epistemic, 'aleatoric': total_variance - epistemic,
            'classification': np.where(probability >= .5, 2, 1).astype(np.uint8)}
