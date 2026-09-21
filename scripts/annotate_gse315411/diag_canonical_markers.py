#!/usr/bin/env python
"""
Diagnostic — canonical-marker table per Leiden cluster (the annotator's dotplot).

Per cluster: mean counts per 100 transcripts of a fixed panel of canonical
lung / immune / stromal / vascular markers, saved as canonical_markers.csv
(clusters x genes); printed per cluster as the panel genes on which the
cluster reaches >= 50 % of the across-cluster maximum, i.e. the markers it
"lights up" relative to every other cluster.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu import add_common_args, pin_gpu, resolve_work, set_seed

PANEL = {
    "immune": "PTPRC CD3E CD4 CD8A NKG7 GNLY KLRD1 FOXP3 MS4A1 CD79A MZB1 JCHAIN IGKC CD68 CD163 MRC1 "
              "MARCO FABP4 SPP1 S100A8 S100A9 CSF3R FCGR3B KIT TPSAB1 CPA3 CD1C CLEC9A LILRA4 FCN1 VCAN MKI67",
    "epithelial": "EPCAM KRT8 KRT18 AGER HOPX PDPN CAV1 SFTPC SFTPB LAMP3 ABCA3 ETV5 SCGB3A2 SCGB1A1 "
                  "FOXJ1 CAPS TPPP3 MUC5B MUC5AC BPIFB1 KRT5 TP63 KRT17 KRT14 CHGA ASCL1 CALCA CFTR LTF "
                  "AZGP1 DMBT1 KRT13 HSPA6 HSPA1A",
    "endothelial": "PECAM1 CDH5 VWF CLDN5 CA4 APLN EDNRB HPGD ACKR1 PLVAP SELE GJA5 DKK2 BMX PROX1 LYVE1 "
                   "CCL21 FLT4 ESM1 ANGPT2",
    "mesenchymal": "COL1A1 COL1A2 COL3A1 DCN LUM PDGFRA PDGFRB ACTA2 MYH11 TAGLN DES CNN1 RGS5 MCAM NOTCH3 "
                   "PI16 MFAP5 SFRP2 WNT2 FGFR4 TCF21 NPNT CTHRC1 POSTN COL2A1 SOX9 ACAN MYH1 MYH2 TNNT2 "
                   "MYL2 NPPA MPZ SOX10 PLP1 HBB HBA1 GYPA HBG1 PF4 PPBP ITGA2B",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--frac", type=float, default=0.5)
    args = p.parse_args()
    pin_gpu(-1)
    set_seed(args.seed)
    root, work = resolve_work(args)

    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy import sparse

    adata = sc.read_h5ad(work / "query_emb.h5ad")
    x = sparse.csr_matrix(adata.layers["counts"])
    genes = pd.Series(np.arange(adata.n_vars), index=adata.var_names)
    wanted = [g for grp in PANEL.values() for g in grp.split()]
    present = [g for g in wanted if g in genes.index]
    missing = sorted(set(wanted) - set(present))
    print(f"{len(present)} panel genes present, missing: {missing}")
    depth = np.asarray(x.sum(axis=1)).ravel().astype(float); depth[depth == 0] = 1
    xn = sparse.diags(100.0 / depth) @ x[:, genes[present].to_numpy()]
    clusters = adata.obs["leiden"].astype(str).to_numpy()
    names = sorted(set(clusters), key=int)
    codes = pd.Categorical(clusters, names).codes
    onehot = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                               shape=(len(names), len(codes)))
    sizes = np.asarray(onehot.sum(axis=1)).ravel()
    means = pd.DataFrame(np.asarray((onehot @ xn).todense()) / sizes[:, None],
                         index=names, columns=present)
    means.to_csv(work / "canonical_markers.csv")
    rel = means / means.max(axis=0)
    for k in names:
        hits = rel.loc[k][rel.loc[k] >= args.frac].sort_values(ascending=False)
        print(f"{k:>3} n={int(sizes[int(k)]):7d}: " +
              " ".join(f"{g}({means.loc[k, g]:.2f})" for g in hits.index[:14]))
    print(f"\nwrote {work / 'canonical_markers.csv'}")


if __name__ == "__main__":
    main()
