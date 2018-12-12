import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import mannwhitneyu

# load data
data = pd.read_csv("../data/econ_and_sent_vars_consolidated_20181028_2030.txt", sep="\t")

data.index = pd.DatetimeIndex(data["month_date"])
data.drop(columns=["month_date"], inplace=True)

sent_vars = ['num_words', 'perc_pos', 'perc_neg', 'perc_uncert', 'perc_litig', 'perc_modal_wk'
    , 'perc_modal_wod', 'perc_modal_str', 'perc_constrain', 'num_alphanum', 'num_digits'
    , 'num_nums', 'avg_syll_word', 'avg_word_len', 'vocab', 'num_words_rev_weighted'
    , 'perc_pos_rev_weighted', 'perc_neg_rev_weighted', 'perc_uncert_rev_weighted'
    , 'perc_litig_rev_weighted', 'perc_modal_wk_rev_weighted', 'perc_modal_wod_rev_weighted'
    , 'perc_modal_str_rev_weighted', 'perc_constrain_rev_weighted', 'num_alphanum_rev_weighted'
    , 'num_digits_rev_weighted', 'num_nums_rev_weighted', 'avg_syll_word_rev_weighted'
    , 'avg_word_len_rev_weighted', 'vocab_rev_weighted']

econ_vars = ['ur', 'pi_gr', 'nyse_gr', 'tb_sqrt_diff', 'ics_log_diff', "pce_gr_diff"]

# lag features
sent_vars_lagged = []
for col in sent_vars:
    for i in range(1, 4):
        data[col + "_lag" + str(i)] = data[col].shift(i)
        sent_vars_lagged.append(col + "_lag" + str(i))

econ_vars_lagged = []
econ_vars_lag1 = []
for col in econ_vars:
    for i in range(1, 4):
        data[col + "_lag" + str(i)] = data[col].shift(i)
        econ_vars_lagged.append(col + "_lag" + str(i))
        if i == 1:
            econ_vars_lag1.append(col + "_lag" + str(i))

# Drop NaN rows resulting from lagged variables
data = data.loc["2011-04-01":]

best_econ = ['ics_log_diff_lag2', 'nyse_gr_lag2'
    , 'pce_gr_diff_lag1', 'pi_gr_lag3'
    , 'tb_sqrt_diff_lag2', 'ur_lag3']

best_sent = ["num_digits_lag2", "perc_constrain_lag1"
    , "perc_litig_rev_weighted_lag1"
    , "avg_word_len_rev_weighted_lag2"]

# create the "reduced" and "full" random forest objects
rf_reduced = RandomForestRegressor()
rf_full = RandomForestRegressor(10000)

# data to arrays
Xs_reduced = data[best_econ].values
Xs_full = data[best_econ + best_sent].values
y = data["pce_gr_diff"].values  # response


reduced_mse, full_mse = [], []
for i in range(5):
    ss = ShuffleSplit(n_splits=10, test_size=0.2)
    for train_idx, test_idx in ss.split(Xs_reduced):
        Xs_reduced_train = Xs_reduced[train_idx]
        Xs_sent_train = Xs_full[train_idx]
        Xs_reduced_test = Xs_reduced[test_idx]
        Xs_sent_test = Xs_full[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # fit models
        rf_reduced.fit(Xs_reduced_train, y_train)
        rf_full.fit(Xs_sent_train, y_train)
        mse_nosent = mean_squared_error(y_test, rf_reduced.predict(Xs_reduced_test))
        reduced_mse.append(mse_nosent)
        mse_sent = mean_squared_error(y_test, rf_full.predict(Xs_sent_test))
        full_mse.append(mse_sent)

# print mean of MSE calcs and result of MWU test
print(np.mean(reduced_mse))
print(np.mean(full_mse))
print(mannwhitneyu(reduced_mse, full_mse))


# final estimates of "reduced" random forest model
rf_reduced_final = RandomForestRegressor()

cv = ShuffleSplit(n_splits=5, test_size=0.2)

final_reduced_mse = []
final_reduced_r2 = []
final_reduced_adj_r2 = []

for train_idx, test_idx in cv.split(Xs_reduced):
    Xs_train, Xs_test = Xs_reduced[train_idx], Xs_reduced[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    rf_reduced_final.fit(Xs_train, y_train)
    y_pred = rf_reduced_final.predict(Xs_reduced)
    mse = mean_squared_error(y_pred, y)
    r2 = r2_score(y_pred, y)
    adj_r2 = ((1-r2) * (Xs_train.shape[0]-1))/ (Xs_train.shape[0] - Xs_train.shape[1] - 1)
    final_reduced_r2.append(r2)
    final_reduced_adj_r2.append(adj_r2)
    final_reduced_mse.append(mse)

print(f"Reduced RF Model R^2: {round(np.mean(final_reduced_r2), 4)}")
print(f"Reduced RF Model Adj. R^2: {round(np.mean(final_reduced_adj_r2), 4)}")
print(f"Reduced RF Model MSE: {round(np.mean(final_reduced_mse), 8)}")

# final estimates of "full" RF model
rf_full_final = RandomForestRegressor(n_estimators=50000)

final_full_mse = []
final_full_r2 = []
final_full_adj_r2 = []

for train_idx, test_idx in cv.split(Xs_full):
    Xs_train, Xs_test = Xs_full[train_idx], Xs_full[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    rf_full_final.fit(Xs_train, y_train)
    y_pred = rf_full_final.predict(Xs_full)
    mse = mean_squared_error(y_pred, y)
    r2 = r2_score(y_pred, y)
    adj_r2 = ((1-r2) * (Xs_train.shape[0]-1))/ (Xs_train.shape[0] - Xs_train.shape[1] - 1)
    final_full_r2.append(r2)
    final_full_adj_r2.append(adj_r2)
    final_full_mse.append(mse)

print(f"Full RF Model R^2: {round(np.mean(final_full_r2), 4)}")
print(f"Full RF Model Adj. R^2: {round(np.mean(final_full_adj_r2), 4)}")
print(f"Full RF Model MSE: {round(np.mean(final_full_mse), 8)}")
