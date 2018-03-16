import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


tile_sizes = [2, 5, 10, 50, 200]

rfr_mse_list = []
lr_mse_list = []

rfr_list = []

for ts in tile_sizes:
    elements = pd.read_table("~/scratch/MUTMODEL/whole_genome_sliding.{}kb_bins.mut_rates.reduced_annotation.coverage_added.constraint.txt".format(ts))
    elements.dropna(axis = 0, inplace = True)

    elements['meta_observed_CA_prop'] = elements.meta_observed_CA_neutral/elements.meta_observed_neutral
    elements['meta_observed_CG_prop'] = elements.meta_observed_CG_neutral/elements.meta_observed_neutral
    elements['meta_observed_CT_prop'] = elements.meta_observed_CT_neutral/elements.meta_observed_neutral
    elements['meta_observed_TA_prop'] = elements.meta_observed_TA_neutral/elements.meta_observed_neutral
    elements['meta_observed_TC_prop'] = elements.meta_observed_TC_neutral/elements.meta_observed_neutral
    elements['meta_observed_TG_prop'] = elements.meta_observed_TG_neutral/elements.meta_observed_neutral

    # hard filtering to make the comparison fairer
    elements = elements[elements.median_coverage_gnomad > 20]
    elements = elements[elements.median_coverage_BRIDGE > 20]
    elements = elements[elements.low_qual_prop_gnomad < 0.3]
    elements = elements[elements.low_qual_prop_BRIDGE < 0.3]


    # linear regression

    LR = LinearRegression()
    X = elements.loc[:, ['p_snp_phylop_lt_0', 'meta_observed_neutral']]
    y = X['meta_observed_neutral']
    X.drop('meta_observed_neutral', axis = 1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    LR.fit(X_train, y_train)
    lr_mse = LR.score(X_test, y_test)

    # random forest regression
    X = elements.loc[:, ['p_snp_phylop_lt_0', \
                 'GC_content', 'low_complexity_regions','meta_observed_CA_prop', 'meta_observed_CT_prop', 'meta_observed_CG_prop', \
                 'meta_observed_TA_prop','meta_observed_TC_prop','meta_observed_TG_prop', \
                 'replication_timing_Koren_LCLs', 'replication_timing_DingKoren_ESCs', \
                 'recombination_rate_kong_female','recombination_rate_kong_male', 'recombination_rate_1000G_phase3', \
                         'ovary_DNase', 'hSSC_ATAC', 'hESC_ATAC', 'ovary_H3K27ac', \
    'ovary_H3K27me3', 'ovary_H3K9me3', 'ovary_H3K4me3', 'ovary_H3K4me1', \
    'ovary_H3K36me3', 'hESC_H3K27me3', 'hESC_H3K9me3', 'hESC_H3K4me3', \
    'hESC_H3K4me1', 'hESC_H3K36me3', 'hESC_H3K9ac', 'meta_observed_neutral']]

    RFR = RandomForestRegressor(n_estimators = 1000, max_depth = 100, max_features = int(np.round(X.shape[1]/2)), min_samples_leaf = 10, min_samples_split = 20, n_jobs=-1, verbose = 4)

    X.dropna(axis=0,inplace=True)
    y = X['meta_observed_neutral']
    X.drop('meta_observed_neutral', axis = 1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    RFR.fit(X_train, y_train)
    rfr_mse = RFR.score(X_test, y_test)

    rfr_list.append(RFR)

    lr_mse_list.append(lr_mse)
    rfr_mse_list.append(rfr_mse)

print(lr_mse_list)
print(rfr_mse_list)

