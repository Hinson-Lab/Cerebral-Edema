"""
ce_proteomics
=============

Analysis code for:

    Radabaugh HL, et al. Protein Predictors of Worsening Cerebral Edema in
    Traumatic Brain Injury in Critically Ill Patients. Brain Communications.

Modules
-------
config                Paths, column names, and all tunable constants
io                    Loading and saving helpers
preprocessing         Olink pivot, REDCap merge, analysis-dataset construction
univariate            Adjusted logistic screen, panel boxplots, volcano plot
elastic_net           Bootstrapped elastic-net panel selection
pca                   PCA, correlation loadings, bootstrap stability, plots
classification        Cross-validated classifiers, bootstrap CIs, calibration, DeLong
classification_plots  ROC, PR, forest, calibration, and KDE figures
interpretation        Random-forest importance, SHAP, SHAP interactions
enrichment            STRING enrichment and protein-list export for enrichment tools
diagnostics           Glucose and alcohol descriptive figures (Supp. Fig. 3)
plotting              Shared plot style and save helper
"""

__version__ = "1.0.0"
