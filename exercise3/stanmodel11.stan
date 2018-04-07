data {
  // Dimensions
  int<lower=1> N_obs;
  int<lower=0> N_cens;
  int<lower=1> D; 
  // Data
  matrix[N_obs, D] X_obs;
  matrix[N_cens, D] X_cens;
  int y_obs[N_obs];
  // Prior hyperparams
  real mu0;
  real<lower=0> sigma0;
}

parameters {
  vector[D] beta; // logintensity
  real<lower=0> phi; // overdispersion
}

transformed parameters  {
  vector[N_obs] mu_obs = X_obs * beta;
  vector[N_cens] mu_cens = X_cens * beta;
}

model {
  beta ~ normal(mu0, sigma0);
  y_obs ~ neg_binomial_2_log(mu_obs, phi);
  // (log) Likelihood for censored values
  for (n in 1:N_cens) {
    target += neg_binomial_2_log_lpmf(4 | mu_cens[n], phi);
  }
}

generated quantities {
  vector[N_obs] y_predict_obs;
  vector[N_cens] y_predict_cens;
  for (i in 1:N_obs)
    y_predict_obs[i] = neg_binomial_2_log_rng(mu_obs[i], phi);
  for (i in 1:N_cens)  
    y_predict_cens[i] = neg_binomial_2_log_rng(mu_cens[i], phi);
}