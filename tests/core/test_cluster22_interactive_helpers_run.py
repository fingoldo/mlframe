"""Cluster 22 regression: the two interactive diagnostic helpers run on the installed numpy and sklearn (audit 2026-11-20, 22-F23)."""

from mlframe.core.helpers import ListAllSkLearnClassifiers, MakeSureBlasAndLaPackAreInstalled


def test_blas_lapack_diagnostic_prints_both_backends(capsys):
    """The BLAS and LAPACK lines are printed instead of an ImportError from numpy.distutils (22-F23)."""
    MakeSureBlasAndLaPackAreInstalled()
    out = capsys.readouterr().out
    assert out.startswith("blas ") and "\nlapack " in out


def test_classifier_listing_prints_known_estimators(capsys):
    """The listing runs and includes the random forest classifier, whose name contains "Class" (22-F23)."""
    ListAllSkLearnClassifiers()
    assert "RandomForestClassifier" in capsys.readouterr().out
