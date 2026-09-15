"""Provenance and metadata analysis module for SignalScope.

Provides auxiliary forensic analysis of image metadata including EXIF tags,
PNG metadata chunks, and C2PA Content Credentials / JUMBF structures.
"""

from src.provenance.analyzer import analyze_provenance, ProvenanceResult

__all__ = ["analyze_provenance", "ProvenanceResult"]
