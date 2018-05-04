data {
  int<lower=0> N;
  int<lower=0> D;
  matrix[N, D] X;
  vector[N, D] y;
  real<lower=0> kappa;
  real<lower=0> a;
  real<lower=0> b;
}
parameters {
  vector[D] beta;
  real<lower=0> sigma2;
}
transformed parameters {
  vector[N] mu;
  mu = X * beta;
}
model {
  sigma2 ~ inv_gamma(a, b);
  beta ~ normal(0, sqrt(sigma2 / kappa));
  for (i in 1:N) 
    y[i] ~ normal(mu, sqrt(sigma2));
}