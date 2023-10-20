#install.packages("xlsx")
library(xlsx)
#install.packages("R2jags")
library(R2jags)

#_______________________________________________________________________________

#cas 1: alpha = 0
pd0 <-read.xlsx("score_a0_torchpd.xlsx", sheetIndex = 1)

#plot graphic
boxplot(score~label, data = pd0, ylim = c(200, 600))
mean_by_group <-tapply(pd0$score, list(pd0$label), mean)

#ajouter les valeurs des moyennes au graphique
text(x=1:length(mean_by_group), y = mean_by_group, labels = mean_by_group, pos = 1, col = "red")

modelePd0 <- function(){
  #vraissemblance
  for(i in 1:N){
    y[i]~dnorm(mu[i], prec) #condition d'homoscédasticité
    mu[i] <- nev + beta*G[i]
  }
  #priors
  nev ~ dnorm(mu0, 0.01)
  beta ~ dnorm(0, 0.01)
  prec ~ dgamma(0.005, 0.01)
  #parameters of interest
  s <- sqrt(1/prec)
  mel <- nev + beta
}
donnees <- list(y=pd0$score, G=(pd0$label), N=nrow(pd0), mu0=274)
parametre <- c("nev", "beta", "s", "mel")
(resultat <- jags(data=donnees, inits = NULL, parameters.to.save = parametre, model.file = modelePd0,
                  n.chains = 3, n.iter = 55000, n.burnin = 5000, n.thin = 15))
plot(resultat$BUGSoutput$sims.list$mel, col="red", type="l")
acf(resultat$BUGSoutput$sims.list$mel, lag.max = 30, main = "")

#-------------------------------------------------------------------------------------------------------
#cas 2: alpha = 0.5

pd05 <-read.xlsx("score_a05_torchpd.xlsx", sheetIndex = 1)

#plot graphic
boxplot(score~label, data = pd05, ylim = c(200, 800))
mean_by_group <-tapply(pd05$score, list(pd05$label), mean)

#ajouter les valeurs des moyennes au graphique
text(x=1:length(mean_by_group), y = mean_by_group, labels = mean_by_group, pos = 1, col = "red")

modelePd05 <- function(){
  #vraissemblance
  for(i in 1:N){
    y[i]~dnorm(mu[i], prec) #condition d'homoscédasticité
    mu[i] <- nev + beta*G[i]
  }
  #priors
  nev ~ dnorm(mu0, 0.01)
  beta ~ dnorm(0, 0.01)
  prec ~ dgamma(0.005, 0.01)
  #parameters of interest
  s <- sqrt(1/prec)
  mel <- nev + beta
}
donnees <- list(y=pd05$score, G=(pd05$label), N=nrow(pd05), mu0=304)
parametre <- c("nev", "beta", "s", "mel")
(resultat <- jags(data=donnees, inits = NULL, parameters.to.save = parametre, model.file = modelePd05,
                  n.chains = 3, n.iter = 55000, n.burnin = 5000, n.thin = 15))
plot(resultat$BUGSoutput$sims.list$mel, col="red", type="l")
acf(resultat$BUGSoutput$sims.list$mel, lag.max = 30, main = "")
#-------------------------------------------------------------------------------------------------------
#cas 3: alpha = 1

pd1 <-read.xlsx("score_a1_torchpd.xlsx", sheetIndex = 1)

#plot graphic
boxplot(score~label, data = pd1, ylim = c(200, 1300))
mean_by_group <-tapply(pd1$score, list(pd1$label), mean)

#ajouter les valeurs des moyennes au graphique
text(x=1:length(mean_by_group), y = mean_by_group, labels = mean_by_group, pos = 1, col = "red")

modelePd1 <- function(){
  #vraissemblance
  for(i in 1:N){
    y[i]~dnorm(mu[i], prec) #condition d'homoscédasticité
    mu[i] <- nev + beta*G[i]
  }
  #priors
  nev ~ dnorm(mu0, 0.01)
  beta ~ dnorm(0, 0.01)
  prec ~ dgamma(0.005, 0.01)
  #parameters of interest
  s <- sqrt(1/prec)
  mel <- nev + beta
}
donnees <- list(y=pd1$score, G=(pd1$label), N=nrow(pd1), mu0=335)
parametre <- c("nev", "beta", "s", "mel")
(resultat <- jags(data=donnees, inits = NULL, parameters.to.save = parametre, model.file = modelePd1,
                  n.chains = 3, n.iter = 55000, n.burnin = 5000, n.thin = 15))
plot(resultat$BUGSoutput$sims.list$mel, col="red", type="l")
acf(resultat$BUGSoutput$sims.list$mel, lag.max = 30, main = "")
#-------------------------------------------------------------------------------------------------------