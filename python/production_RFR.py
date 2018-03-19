import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys

f = sys.argv[1]

elements = pd.read_table(f)

elements['meta_observed_CA_prop'] = elements.meta_observed_CA_neutral/elements.meta_observed_neutral
elements['meta_observed_CG_prop'] = elements.meta_observed_CG_neutral/elements.meta_observed_neutral
elements['meta_observed_CT_prop'] = elements.meta_observed_CT_neutral/elements.meta_observed_neutral
elements['meta_observed_TA_prop'] = elements.meta_observed_TA_neutral/elements.meta_observed_neutral
elements['meta_observed_TC_prop'] = elements.meta_observed_TC_neutral/elements.meta_observed_neutral
elements['meta_observed_TG_prop'] = elements.meta_observed_TG_neutral/elements.meta_observed_neutral

elements.dropna(axis = 0, inplace = True)

print(elements.shape)

X = elements.loc[:, ['p_snp_phylop_lt_0', 'low_qual_prop_BRIDGE', 'low_qual_prop_gnomad', 'median_coverage_BRIDGE', 'median_coverage_gnomad', \
                     'chr', 'telomere_dist', 'arm', \
             'GC_content', 'low_complexity_regions','meta_observed_CA_prop', 'meta_observed_CT_prop', 'meta_observed_CG_prop', \
             'meta_observed_TA_prop','meta_observed_TC_prop','meta_observed_TG_prop', \
             'replication_timing_Koren_LCLs', 'replication_timing_DingKoren_ESCs', \
             'recombination_rate_kong_female','recombination_rate_kong_male', 'recombination_rate_1000G_phase3', \
                     'ovary_DNase', 'hSSC_ATAC', 'hESC_ATAC', 'ovary_H3K27ac', \
'ovary_H3K27me3', 'ovary_H3K9me3', 'ovary_H3K4me3', 'ovary_H3K4me1', \
'ovary_H3K36me3', 'hESC_H3K27me3', 'hESC_H3K9me3', 'hESC_H3K4me3', \
'hESC_H3K4me1', 'hESC_H3K36me3', 'hESC_H3K9ac', 'meta_observed_neutral']]

# fit the linear regression
LR = LinearRegression()
y = X['meta_observed_neutral']
X.drop('meta_observed_neutral', axis = 1, inplace=True)
LR.fit(X['p_snp_phylop_lt_0'].values.reshape(-1,1), y)


def obs_exp_z_score(observed, expected):
    return (observed - expected)/np.sqrt(expected)

elements['meta_predicted_neutral_triplet_only'] = LR.predict(X['p_snp_phylop_lt_0'].values.reshape(-1,1))
elements['meta_obs_exp_ratio_neutral_triplet_only'] = elements.meta_observed_neutral/elements['meta_predicted_neutral_triplet_only']
elements['meta_z_score_neutral_triplet_only'] = obs_exp_z_score(elements.meta_observed_neutral, elements.meta_predicted_neutral_triplet_only)

y_prop = y/LR.predict(X['p_snp_phylop_lt_0'].values.reshape(-1,1))
X.drop('p_snp_phylop_lt_0', axis = 1, inplace=True)

### RFR including just the technical features

X = elements.loc[:, ['p_snp_phylop_lt_0', 'low_qual_prop_BRIDGE', 'low_qual_prop_gnomad', 'median_coverage_BRIDGE', 'median_coverage_gnomad', 'meta_observed_neutral']]

RFR = RandomForestRegressor(n_estimators = 1000, max_depth = 100, max_features = int(np.round(X.shape[1]/2)), min_samples_leaf = 10, min_samples_split = 20, n_jobs=-1, verbose = 4)
X_train, X_test, y_train, y_test = train_test_split(X, y_prop, random_state=42)
RFR.fit(X_train, y_train)

elements['meta_predicted_neutral_RFR_technical'] = RFR.predict(X)
elements['meta_obs_exp_ratio_neutral_RFR_technical'] = elements.meta_observed_neutral/elements['meta_predicted_neutral_RFR_technical']
elements['meta_z_score_neutral_RFR_technical'] = obs_exp_z_score(elements.meta_observed_neutral, elements.meta_predicted_neutral_RFR_technical)


### RFR adding in genomic features

X = elements.loc[:, ['p_snp_phylop_lt_0', 'low_qual_prop_BRIDGE', 'low_qual_prop_gnomad', 'median_coverage_BRIDGE', 'median_coverage_gnomad', \
                     'telomere_dist', 'arm', \
             'GC_content', 'low_complexity_regions', \
             'replication_timing_Koren_LCLs', 'replication_timing_DingKoren_ESCs', \
             'recombination_rate_kong_female','recombination_rate_kong_male', 'recombination_rate_1000G_phase3', \
                     'ovary_DNase', 'hSSC_ATAC', 'hESC_ATAC', 'ovary_H3K27ac', \
'ovary_H3K27me3', 'ovary_H3K9me3', 'ovary_H3K4me3', 'ovary_H3K4me1', \
'ovary_H3K36me3', 'hESC_H3K27me3', 'hESC_H3K9me3', 'hESC_H3K4me3', \
'hESC_H3K4me1', 'hESC_H3K36me3', 'hESC_H3K9ac', 'meta_observed_neutral']]


RFR = RandomForestRegressor(n_estimators = 1000, max_depth = 100, max_features = int(np.round(X.shape[1]/2)), min_samples_leaf = 10, min_samples_split = 20, n_jobs=-1, verbose = 4)
X_train, X_test, y_train, y_test = train_test_split(X, y_prop, random_state=42)
RFR.fit(X_train, y_train)

elements['meta_predicted_neutral_RFR_genomic'] = RFR.predict(X)
elements['meta_obs_exp_ratio_neutral_RFR_genomic'] = elements.meta_observed_neutral/elements['meta_predicted_neutral_RFR_genomic']
elements['meta_z_score_neutral_RFR_genomic'] = obs_exp_z_score(elements.meta_observed_neutral, elements.meta_predicted_neutral_RFR_genomic)


### RFR adding in the polymorphism features

X = elements.loc[:, ['p_snp_phylop_lt_0', 'low_qual_prop_BRIDGE', 'low_qual_prop_gnomad', 'median_coverage_BRIDGE', 'median_coverage_gnomad', \
                     'chr', 'telomere_dist', 'arm', \
             'GC_content', 'low_complexity_regions','meta_observed_CA_prop', 'meta_observed_CT_prop', 'meta_observed_CG_prop', \
             'meta_observed_TA_prop','meta_observed_TC_prop','meta_observed_TG_prop', \
             'replication_timing_Koren_LCLs', 'replication_timing_DingKoren_ESCs', \
             'recombination_rate_kong_female','recombination_rate_kong_male', 'recombination_rate_1000G_phase3', \
                     'ovary_DNase', 'hSSC_ATAC', 'hESC_ATAC', 'ovary_H3K27ac', \
'ovary_H3K27me3', 'ovary_H3K9me3', 'ovary_H3K4me3', 'ovary_H3K4me1', \
'ovary_H3K36me3', 'hESC_H3K27me3', 'hESC_H3K9me3', 'hESC_H3K4me3', \
'hESC_H3K4me1', 'hESC_H3K36me3', 'hESC_H3K9ac', 'meta_observed_neutral']]


RFR = RandomForestRegressor(n_estimators = 1000, max_depth = 100, max_features = int(np.round(X.shape[1]/2)), min_samples_leaf = 10, min_samples_split = 20, n_jobs=-1, verbose = 4)
X_train, X_test, y_train, y_test = train_test_split(X, y_prop, random_state=42)
RFR.fit(X_train, y_train)

elements['meta_predicted_neutral_RFR_full'] = RFR.predict(X)
elements['meta_obs_exp_ratio_neutral_RFR_full'] = elements.meta_observed_neutral/elements['meta_predicted_neutral_RFR_full']
elements['meta_z_score_neutral_RFR_full'] = obs_exp_z_score(elements.meta_observed_neutral, elements.meta_predicted_neutral_RFR_full)

elements.to_csv("~/scratch/MUTMODEL/whole_genome_sliding.2kb_bins.mut_rates.reduced_annotation.coverage_added.constraint.model_annotated.txt", sep = "\t")