// Stan model 1 for problem 1
data {
  // meta
  int<lower=1> N; // nobs
  // data
  vector[N] x; // feature
  vector[N] y; // response
  // prior
  real<lower=0> a0; // omega ~ Gamma(a0, b0)
  real<lower=0> b0;
  real mu0; // beta ~ N(mu0, (kappa0*omega)^{-1})
  real<lower=0> kappa0; 
}
parameters {
  real alpha; // intercept
  real beta; // lin coefs
  real<lower=0> omega; // precision
}
transformed parameters {
  real<lower=0> sigma; // standard dev
  vector[N] yhat; // linpred
  sigma = pow(omega, -0.5);
  yhat = alpha + x * beta;
}
model {
  // prior
  // alpha ~ 1; it's assumed
  omega ~ gamma(a0, b0);
  beta ~ normal(mu0, sigma / kappa0);
  // likelihood
  y ~ normal(yhat, sigma);
}
generated quantities {
  vector[N] residuals;
  residuals = y - yhat;
}
