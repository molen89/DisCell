"""Matcher tests for discell.model.labels: old and new lineage vocabularies
must both be recognised (2026-09-25 lineage relabel, see
scripts/logs/lineage_2026-09-25/AGENT_REPORT.md)."""

import pytest

from discell.model.labels import is_endothelial, is_smooth_muscle, is_tumour

OLD_TUMOUR = ["Tumor Cells", "Proliferative Tumor Cells", "VEGFA+ Tumor Cells",
              "Inflammatory Tumor Cells", "SOX2-OT+ Tumor Cells"]
NEW_TUMOUR = ["Tumour"]
OLD_SMOOTH_MUSCLE = ["Smooth Muscle Cells"]
NEW_SMOOTH_MUSCLE = ["Smooth muscle cells"]
OLD_ENDOTHELIAL = ["Tumor Associated Endothelial Cells",
                    "Stromal Associated Endothelial Cells"]
NEW_ENDOTHELIAL = ["Endothelial", "Endothelial cells"]

NOT_TUMOUR = ["Mesothelial-like cyst lining", "Unassigned", "Macrophages",
              "T and NK Cells", "Pericytes"]


@pytest.mark.parametrize("name", OLD_TUMOUR + NEW_TUMOUR)
def test_is_tumour_matches_old_and_new_names(name):
    assert is_tumour(name)


def test_is_tumour_excludes_mesothelial_cyst_lining():
    # descends from the old "Malignant Cells Lining Cyst" but the markers
    # are mesothelial, not tumour, under the new lineage map.
    assert not is_tumour("Mesothelial-like cyst lining")


def test_is_tumour_excludes_old_cyst_lining_name_too():
    # the old name also says "Malignant", but its markers are mesothelial
    # (CALB2/PRG4/BNC1/PDPN/UPK3B+, EPCAM/PAX8/ESR1-) -- the "lining"
    # exclusion deliberately overrides the naive "Malignant" substring
    # match for this class under both vocabularies.
    assert not is_tumour("Malignant Cells Lining Cyst")


@pytest.mark.parametrize("name", NOT_TUMOUR)
def test_is_tumour_rejects_non_tumour_names(name):
    assert not is_tumour(name)


def test_is_tumour_case_insensitive():
    assert is_tumour("tumour")
    assert is_tumour("TUMOR")
    assert is_tumour("malignant")
    assert is_tumour("carcinoma")


@pytest.mark.parametrize("name", OLD_SMOOTH_MUSCLE + NEW_SMOOTH_MUSCLE)
def test_is_smooth_muscle_matches_old_and_new_names(name):
    assert is_smooth_muscle(name)


def test_is_smooth_muscle_rejects_unrelated_name():
    assert not is_smooth_muscle("Pericytes")


@pytest.mark.parametrize("name", OLD_ENDOTHELIAL + NEW_ENDOTHELIAL)
def test_is_endothelial_matches_old_and_new_names(name):
    assert is_endothelial(name)


def test_is_endothelial_rejects_unrelated_name():
    assert not is_endothelial("Pericytes")


def test_tumour_associated_stroma_is_not_tumour():
    from discell.model.labels import is_tumour
    for name in ("Tumor Associated Fibroblasts", "Tumor Associated Endothelial Cells",
                 "Tumour-associated macrophages", "Stromal Associated Fibroblasts"):
        assert not is_tumour(name), name
    for name in ("Tumor Cells", "Tumour", "Proliferative Tumor Cells", "VEGFA+ Tumor Cells",
                 "SOX2-OT+ Tumor Cells", "Inflammatory Tumor Cells", "Carcinoma"):
        assert is_tumour(name), name


def test_malignant_cells_is_tumour_and_t_cells_is_not():
    from discell.model.labels import is_tumour
    assert is_tumour("Malignant Cells") and is_tumour("Malignant cells")
    assert not is_tumour("T cells") and not is_tumour("T-cells") and not is_tumour("Tumour T cells")
