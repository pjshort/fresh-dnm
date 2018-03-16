import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

__author__ = 'patrick short'

# Inputs:
# Genomic regions (equal sized bins performs best)
# annotated with p_snp_phylop_lt_0 (triplet mutability in neutral sites)
# annotated with any number of genomic features

elements = pd.read_table("./whole_genome_sliding.10kb_bins.mut_rates.reduced_annotation.coverage_added.constraint.txt")

elements['meta_observed_CA_prop'] = elements.meta_observed_CA_neutral/elements.meta_observed_neutral
elements['meta_observed_CG_prop'] = elements.meta_observed_CG_neutral/elements.meta_observed_neutral
elements['meta_observed_CT_prop'] = elements.meta_observed_CT_neutral/elements.meta_observed_neutral
elements['meta_observed_TA_prop'] = elements.meta_observed_TA_neutral/elements.meta_observed_neutral
elements['meta_observed_TC_prop'] = elements.meta_observed_TC_neutral/elements.meta_observed_neutral
elements['meta_observed_TG_prop'] = elements.meta_observed_TG_neutral/elements.meta_observed_neutral

X = elements.loc[:, ['p_snp_phylop_lt_0', 'low_qual_prop_BRIDGE', 'low_qual_prop_gnomad', 'median_coverage_BRIDGE', 'median_coverage_gnomad' \
                 'GC_content', 'low_complexity_regions','meta_observed_CA_prop', 'meta_observed_CT_prop', 'meta_observed_CG_prop', \
                 'meta_observed_TA_prop','meta_observed_TC_prop','meta_observed_TG_prop', \
                 'replication_timing_Koren_LCLs', 'replication_timing_DingKoren_ESCs', \
                 'recombination_rate_kong_female','recombination_rate_kong_male', 'recombination_rate_1000G_phase3', \
                         'ovary_DNase', 'hSSC_ATAC', 'hESC_ATAC', 'ovary_H3K27ac', \
    'ovary_H3K27me3', 'ovary_H3K9me3', 'ovary_H3K4me3', 'ovary_H3K4me1', \
    'ovary_H3K36me3', 'hESC_H3K27me3', 'hESC_H3K9me3', 'hESC_H3K4me3', \
    'hESC_H3K4me1', 'hESC_H3K36me3', 'hESC_H3K9ac', 'meta_observed_neutral']]

y = X['meta_observed_neutral']
X.drop('meta_observed_neutral', axis = 1, inplace=True)

# features to use for grid search
max_features = ['sqrt', int(np.round(X.shape[1]/3)), int(np.round(X.shape[1]/2))]
max_depth = [int(x) for x in np.linspace(10, 210, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10, 20, 30, 50]
min_samples_leaf = [1, 2, 4, 8, 10, 20]

# Create the random grid
random_grid = {'n_estimators': [500],
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
}

# features to use for grid search
max_features = ['sqrt', int(np.round(X.shape[1]/3)), int(np.round(X.shape[1]/2))]
max_depth = [int(x) for x in np.linspace(10, 210, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10, 20, 30, 50]
min_samples_leaf = [1, 2, 4, 8, 10, 20]

# Create the random grid
random_grid = {'n_estimators': [200],
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
               }

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=4, random_state=42, n_jobs = -1)

rf_random.fit(X, y)

print('The best set of features in the randomised search was:')