"""Core dataclasses and algorithms — hardware-agnostic, no qualibrate/quam imports."""

# Explicit lazy imports: callers should import directly from submodules.
# This avoids circular import issues during incremental development.
__all__ = ["GaussianEstimate", "BICDiagnoser", "BICResult", "NodeResult"]


def __getattr__(name: str):
    if name == "GaussianEstimate":
        from calib_framework.core.estimates import GaussianEstimate
        return GaussianEstimate
    if name in ("BICDiagnoser", "BICResult"):
        import calib_framework.core.bic as _bic
        return getattr(_bic, name)
    if name == "NodeResult":
        from calib_framework.core.node_result import NodeResult
        return NodeResult
    raise AttributeError(f"module 'calib_framework.core' has no attribute {name!r}")
