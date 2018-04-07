// Basic Poisson glm
data {
  // Dimensions 
  int <lower=0> N;
  int <lower=0> D;
  // Covariates and count outcome
  matrix[N, D] X;
  int <lower=0> y[N];
  // Prior hyperparams
  real mu0;
  real<lower=0> sigma0;
}

parameters {
  vector[D] beta; // logintensity
  real<lower=0> phi; // overdispersion
}

transformed parameters {
  vector[N] mu = X * beta; // logintensity
}

model {
  beta ~ normal(mu0, sigma0);
  y ~ neg_binomial_2_log(mu, phi);
}

generated quantities {
  vector[N] y_predict;
  for (i in 1:N)
    y_predict[i] = neg_binomial_2_log_rng(mu[i], phi);
}