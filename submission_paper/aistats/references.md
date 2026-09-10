# Everything that needs a reference

Status: `have` = entry in references.bib; `CHECK` = entry exists but details unverified;
`need` = no entry yet; `[NEEDS A SOURCE]` = a claim in the text carries the visible marker.

## Claims carrying [NEEDS A SOURCE] in the PDF

| where | claim | candidate |
|---|---|---|
| intro, method 2.2, app B.7 | published Xenium transcript-misassignment / leakage estimates of 0.1-0.5 | Salas et al. 2025 (Nat Methods, Xenium best practices)?; Cook et al. 2023?; the resolVI paper's contamination estimates; 10x segmentation notes |
| experiments 3.1 | the ovarian section itself (public 10x dataset, panel, cell counts) | 10x Genomics dataset page; Xenium Prime 5K technical note |

## Methods and tools cited or to cite

| key | what | status |
|---|---|---|
| janesick2023 | Xenium platform | have (author list truncated, CHECK) |
| slavutsky2025 | DisCoVR (two-bound objective, adversarial term, CSVAE comparison) | have |
| klys2018 | CSVAE | have |
| fischer2023 | NCEM | have |
| ergen2025 | resolVI (stop-gradient on neighbour path; contamination model) | CHECK title/authors/id |
| ganchev2010 | posterior regularisation | have |
| brody2022 | GATv2 | have |
| velickovic2018 | GAT | have (not yet cited in text) |
| kingma2014, rezende2014 | VAE / reparameterisation | have (cite in 2.4) |
| kingma2016 | free bits | have |
| tomczak2018 | VampPrior | have |
| louizos2016, xie2017, ganin2016 | adversarial invariance background | have |
| bengio2013 | straight-through estimator | have (cite in 2.5) |
| kingma2015adam, loshchilov2017 | Adam, cosine annealing | have (cite in app C) |
| tirosh2016 | cell-cycle gene sets | have |
| wolf2018 | scanpy (scoring) | have |
| hastie1989 | principal curves | have |
| moran1950 | Moran's I | have (cite in app G) |
| mcinnes2018 | UMAP (figures) | have |
| ester1996 | DBSCAN (landmark instances) | have (cite in app G) |
| kuhn1955 | Hungarian matching (B stability) | have (cite in app H) |
| pedregosa2011, paszke2019, virtanen2020 | software | have |
| ledoit2004, schafer2005 | covariance shrinkage | have (cite in 2.5) |
| cover2006 | Gaussian MI identity | have |
| roberts2017 | spatial block CV | have (cite in 3.1 / app G) |
| barber1996 | Qhull / Delaunay | have |
| dosovitskiy2021, oquab2023 | ViT, DINOv2 | have |
| kronos2025 | KRONOS image foundation model (two generations) | CHECK title/authors/id |
| strehl2002 | NMI | have (cite where NMI is defined) |
| hand2001 | multiclass AUC | have (cite in app D.5) |
| elazar2018 | adversarial removal is not enough (probe protocol) | have |
| dong2025simvi | SIMVI (intrinsic vs spatially induced states) | CHECK title/authors/venue |
| birk2025nichecompass | NicheCompass | CHECK title/authors/venue |
| khemakhem2020 | iVAE identifiability with auxiliary variables | have |
| locatello2019 | no unsupervised disentanglement without inductive bias | have |
| manski2003, rosenbaum2002 | partial identification / sensitivity analysis framing of the sweep | have |
| — | scVIVA and other spatial VAEs (related work stub) | need |
| — | G2/M cells carry ~2x mRNA (validation appendix) | need |
| — | diffuse ambient background negligible vs adjacent-cell leakage on Xenium (method 2.2) | need |
| — | Seurat cc.genes.updated.2019 list (Stuart et al. 2019 / Hao et al. 2021) | need |
| — | anndata (Virshup et al. 2021) | need (software only) |
| — | Dirichlet-multinomial for overdispersed counts | need, only if 2.2 keeps the remark |
| — | multinomial as Poisson conditioned on the total (standard text) | need or drop |
| — | Gumbel-max trick (synthetic type assignment) | need, appendix only |
| — | canonical correlation (Hotelling 1936); principal angles (Bjorck & Golub 1973) | need, appendix only |
| — | UNI / Virchow (0.5 um/px convention), only if discussed | need |
| — | 10x Xenium onboard analysis / Xenium Ranger segmentation (interior / boundary / nuclear expansion) | need |
| — | related-work stubs: spatial VAEs, GNN spatial models, segmentation correction | need |
