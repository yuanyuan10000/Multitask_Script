
library(openxlsx)

setwd("I:/4Multitask/exp")

data = read.csv('data_qqplot.csv')

library(ggplot2)

# ggplot(mtcars, aes(sample = mpg)) +
#   stat_qq() +
#   stat_qq_line()


ggplot(data, aes(sample = AUC, colour = factor(Model),shape=factor(Model))) +
  stat_qq() +
  stat_qq_line(line.p = c(0.25,0.75))+
  
  
  labs(x = "Expected Normal Value", y = "Observed Value")+
  theme(axis.text = element_text(size=15,color="black"))+
  theme(axis.title= element_text(size=15, color="black", face="bold", vjust=0.5, hjust=0.5))+
  
  theme(legend.text = element_text(size = 14))+
  theme(legend.title = element_text(size = 15, face="bold"))+
  # theme(legend.position=c(0,1),legend.justification=c(0,1))+
  theme(legend.background=element_rect(fill='grey92'))+
  
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"))+
  
  facet_wrap(~Model,nrow = 1)
  
  # coord_flip()
  

  
  