# load libraries
library(tidyverse)
library(vars)
library(scales)
# library(ggthemes)

data <- read_delim("../data/econ_and_sent_data_20181108.txt", delim="\t")

my_mse <- function(y_hat, y_true){
  return(mean((y_hat-y_true)^2 ))
}

reduced.model.data <- data %>% dplyr::select(ur, pi_gr, nyse_gr, tb_sqrt_diff, ics_log_diff, pce_gr_diff)

# reduced model
# "SC" is Schwarz criterion (which is BIC==Baysean Information Criterion)
reduced_var_model <- VAR(reduced.model.data, type="const", ic="SC", lag.max=2)
reduced_var_pce_model <- reduced_var_model$varresult$pce_gr_diff

reduced_y_true <- reduced.model.data$pce_gr_diff[2:89]
reduced_y_pred <- reduced_var_pce_model$fitted.values

reduced_var_mse <- my_mse(reduced_y_pred, reduced_y_true)

print(reduced_var_mse)

logLik(reduced_var_pce_model)
summary(reduced_var_pce_model)

# VAR w/ sentiment
full.var.data <- data[-1]

full_var_model <- VAR(full.var.data, type="const", ic="SC", lag.max=2)
full_var_pce_model <- full_var_model$varresult$pce_gr_diff

full_y_true <- full.var.data$pce_gr_diff[2:89]
full_y_pred <- full_var_pce_model$fitted.values

sv_var_mse <- my_mse(full_y_pred, full_y_true)

print(sv_var_mse)

logLik(full_var_pce_model)
summary(full_var_pce_model)


# dump preds to file
# write.table(fullvar_y_pred, "VAR_full_pred.txt", sep="\t")
# write.table(reduced_y_pred, "VAR_reduced_pred.txt", sep="\t")

#----------------------------------------------------
# PLOT REDUCE MODEL PREDICTIONS
#----------------------------------------------------
pce_actual <- read_delim("../data/pce_true_vals_20181107.txt", delim="\t")
beg_date <- as.Date("2011-02-01") # first period in predictions - 1
end_date <- as.Date("2018-06-01") # last period in predictions
pce_actual <- filter(pce_actual, month_date >= beg_date, month_date <= end_date)

# REVERSE TRANSFORM reduced PREDICTED PCE
basline_pred_pce <- c()
for (i in 1:length(reduced_y_pred)){
  curr_per <- pce_actual$month_date[i+1]
  prev_per <- pce_actual$month_date[i]
  pce_gr_prev <- pce_actual$pce_gr[i]
  pred_gr_curr <- as.numeric(reduced_y_pred[i])
  x1 <- 1 + pce_gr_prev + pred_gr_curr
  pce_prev <- pce_actual$pce_infl_dj[i]
  x2 <- x1 * pce_prev
  basline_pred_pce = c(basline_pred_pce, x2)
}

# plot reduced model predictions
plt_reduced_data <- tibble("date"=data$date[2:89], "Predicted"=basline_pred_pce,  "Observed"=pce_actual$pce_infl_dj[2:89])
plt_reduced_data <- gather(plt_reduced_data, PCE, pce_val, -date)

ggplot(plt_reduced_data, aes(x=date, y=pce_val, group=PCE, colour=PCE)) +
  theme_tufte() +
  geom_line(size=0.5) + geom_point(size=0.5) +
  labs(x="Month", y="PCE \n($USD Billions)") +
  theme( legend.text = element_text(size=15, face="bold") # change legend label text size
        , legend.title = element_blank() # hide legend title
        , legend.position = c(0.15, 0.8) # move legend
        , axis.text=element_text(size=15, face="bold") # change axis tick label text size
        , axis.title=element_text(size=15, face="bold") # change axis title text size
        ) + scale_y_continuous(labels=comma)

xform_mse = round(my_mse(basline_pred_pce, pce_actual$pce_infl_dj[2:89]), 2)
xform_rmse = round(sqrt(xform_mse), 2)
print(paste("Mean Square Error:", xform_mse, sep=" "))
print(paste("Root Mean Square Error:", xform_rmse, sep=" "))

#----------------------------------------------------
# PLOT FULL MODEL PREDICTIONS
#----------------------------------------------------

# REVERSE TRANSFORM FULL PREDICTED PCE

pred_full_pce <- c()
for (i in 1:length(full_y_pred)){
  curr_per <- pce_actual$month_date[i+1]
  prev_per <- pce_actual$month_date[i]
  pce_gr_prev <- pce_actual$pce_gr[i]
  pred_gr_curr <- as.numeric(full_y_pred[i])
  x1 <- 1 + pce_gr_prev + pred_gr_curr
  pce_prev <- pce_actual$pce_infl_dj[i]
  x2 <- x1 * pce_prev
  pred_full_pce = c(pred_full_pce, x2)
}

# plot sentiment model predictions
plt_full_data <- tibble("date"=data$date[2:89], "Predicted"=pred_full_pce,  "Observed"=pce_actual$pce_infl_dj[2:89])
plt_full_data <- gather(plt_full_data, PCE, pce_val, -date)
ggplot(plt_full_data, aes(x=date, y=pce_val, group=PCE, colour=PCE))  +
  geom_line() +
  theme_tufte() +
  geom_line(size=0.5) + geom_point(size=0.5) +
  labs(x="Month", y="PCE \n($USD Billions)") +
  theme( legend.text = element_text(size=15, face="bold") # change legend label text size
         , legend.title = element_blank() # hide legend title
         , legend.position = c(0.15, 0.8) # move legend
         , axis.text=element_text(size=15, face="bold") # change axis tick label text size
         , axis.title=element_text(size=15, face="bold") # change axis title text size
  ) + scale_y_continuous(labels=comma)

xform_mse_full = round(my_mse(pred_full_pce, pce_actual$pce_infl_dj[2:89]), 2)
xform_rmse_full = round(sqrt(xform_mse_full), 2)
print(paste("Mean Square Error:", xform_mse_full, sep=" "))
print(paste("Root Mean Square Error:", xform_rmse_full, sep=" "))


#----------------------------------------------------
# STATISTICAL TESTS
#----------------------------------------------------
# likelihood ratio test to 
lrtest(reduced_var_model, full_var_model)
