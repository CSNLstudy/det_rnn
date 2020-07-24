library(tidyverse)

# Natural statistics prior: Girshick et al.(2011, Nat. Neursci.)
Girshick_prior = function(theta){
  p_theta = (2. - abs(sin(theta * pi/90.))) / (360./pi *(pi - 1.)) # degree space
  return(p_theta)
}

# Rejection sampler from the "Girshick prior": may be slower than fancier ways, but still accurate
Girshick_rs = function(n){
  res = vector(len=n)
  for (i in 1:n){
    cont = T
    while (cont){
      s_prop = runif(n=1, min=0, max=180)
      c_prop = runif(n=1, min=0, max=0.009)
      f_prop = Girshick_prior(s_prop)
      if (c_prop < f_prop){ cont = F }
    }
    res[i] = s_prop
  }
  return(res)
}

# Simulation : recover the loss function
N = 20000
criterion = seq(0,82.5,by=7.5)
G_sample  = Girshick_rs(N)
G_sample_n = G_sample + c(rnorm(5000, sd=1),
                          rnorm(5000, sd=10),
                          rnorm(5000, sd=20),
                          rnorm(5000, sd=30))

## Wrap-around correction (assuming no +- 360 outliers!!!)
G_sample_n[G_sample_n > 180] = G_sample_n[G_sample_n > 180] - 180
G_sample_n[G_sample_n < 0] = G_sample_n[G_sample_n < 0] + 180

## % correct as a funciton of criterion
c_calcul  = function(i_s, i_n){
  perf = vector(len=12)
  for (i in 1:12){
    perf[i] = ((criterion[i] < i_s) & (i_s < criterion[i] + 90)) == ((criterion[i] < i_n) & (i_n < criterion[i] + 90))
  }
  return (perf*1)
}

## Simulation
res_perf = vector(len=12*N)
res_crit = vector(len=12*N)
for (i in 1:N){
  res_crit[(12*(i-1)+1):(12*i)] = criterion
  res_perf[(12*(i-1)+1):(12*i)] = c_calcul(G_sample[i], G_sample_n[i])
  if (i %% 1000 == 0){ print(sprintf("Iteration :%d", i)) }
}
res_noise = rep(c(1,10,20,30), each=12*5000)

# Plot
DF = data.frame(Crit = res_crit, Perf = res_perf, Noise=res_noise)
DF$Noise = as.factor(DF$Noise)
DF %>% ggplot(aes(x=Crit,y=1-Perf,col=Noise)) + geom_smooth() + theme_bw() + 
  xlab("Discrimination Criterion") + ylab("% Incorrect(Loss)")
  



