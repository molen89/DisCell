#!/usr/bin/env bash
# Download the public data of one section into DATA_ROOT (default ./data), plus the
# one external table (CellChatDB). Resumable (curl -C -).
#
#   bash reproduce/download.sh ovarian_ffpe|lung_ffpe|ovary_ff|tma_core|tma_serial|external [DATA_ROOT]
#
# Sizes: ovarian FFPE 26.7 GB, lung FFPE 22.6 GB, ovary FF 144 GB (zip; unpacked
# about the same again), GSE315411 slide 11 53.9 GB. The 10x datasets are CC BY 4.0
# (10x Genomics); cite them as the paper does.
set -euo pipefail
section=${1:?section: ovarian_ffpe | lung_ffpe | ovary_ff | tma_core | tma_serial | external}
root=${2:-data}
mkdir -p "$root"
get() { echo "-> $2"; curl -fL -C - --retry 5 -o "$2" "$1"; }
unzip_to() {   # zip, directory: outs unpacked into the directory
  mkdir -p "$2"; unzip -n -q "$1" -d "$2"
  # some archives hold one top-level folder
  if [ ! -e "$2/experiment.xenium" ] && [ "$(ls "$2" | wc -l)" = 1 ]; then
    inner="$2/$(ls "$2")"; mv "$inner"/* "$2"/ && rmdir "$inner"; fi
}
TENX=https://cf.10xgenomics.com/samples/xenium/3.0.0
case "$section" in
  ovarian_ffpe)
    s=Xenium_Prime_Ovarian_Cancer_FFPE_XRrun
    get "$TENX/$s/${s}_outs.zip" "$root/${s}_outs.zip"
    unzip_to "$root/${s}_outs.zip" "$root/$s"
    # 10x's own cell annotation, beside the outs (read_xenium labels="auto")
    get "$TENX/$s/${s}_cell_groups.csv" "$root/$s/${s}_cell_groups.csv" ;;
  lung_ffpe)
    s=Xenium_Prime_Human_Lung_Cancer_FFPE
    get "$TENX/$s/${s}_outs.zip" "$root/${s}_outs.zip"
    unzip_to "$root/${s}_outs.zip" "$root/$s" ;;
  ovary_ff)
    # The outs archive (Xenium_Prime_Human_Ovary_FF_outs.zip, 144 GB) is offered on
    # the dataset page only; download it there and run this again to unpack:
    # https://www.10xgenomics.com/datasets/xenium-prime-fresh-frozen-human-ovary
    s=Xenium_Prime_Human_Ovary_FF
    if [ ! -f "$root/${s}_outs.zip" ]; then
      echo "download ${s}_outs.zip from the dataset page into $root/ first"; exit 1; fi
    unzip_to "$root/${s}_outs.zip" "$root/$s" ;;
  tma_core)
    # GEO GSE315411, sample GSM9427181 (slide PDLTMA06-11, Xenium Prime 5K). The
    # per-cell labels of the paper are an in-house annotation and are not public.
    g=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9427nnn/GSM9427181/suppl
    f=GSM9427181_output-XETG00048_0069221_NB_PDLTMA06-11_5K_20251010_231325.tar.gz
    get "$g/$f" "$root/$f"
    mkdir -p "$root/GSE315411"; tar -xzf "$root/$f" -C "$root/GSE315411"
    # the donor partition of every core
    get https://ftp.ncbi.nlm.nih.gov/geo/series/GSE315nnn/GSE315411/suppl/GSE315411_cell_stats.tar.gz \
        "$root/GSE315411/GSE315411_cell_stats.tar.gz" ;;
  tma_serial)
    # GEO GSE315411, sample GSM9427182 (slide PDLTMA06-10, the TMA core's serial section)
    g=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9427nnn/GSM9427182/suppl
    f=GSM9427182_output-XETG00048_0069259_NB_PDLTMA06-10_5K_20251010_231325.tar.gz
    get "$g/$f" "$root/$f"
    mkdir -p "$root/GSE315411"; tar -xzf "$root/$f" -C "$root/GSE315411" ;;
  external)
    # CellChatDB (CellChat, GPL-3.0), for the signaling member's gene list
    get https://github.com/jinworks/CellChat/raw/main/data/CellChatDB.human.rda \
        "$root/CellChatDB.human.rda" ;;
  *) echo "unknown section $section"; exit 2 ;;
esac
