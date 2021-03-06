library(arules)
library(arulesViz)
library(readr)




#changing our data to transactional data
my_groceries <- read.transactions(file.choose(), sep = "\t",header = TRUE, col=NULL)
View(my_groceries)
str(my_groceries)
summary(my_groceries)
#code that counts lists the most frequent items
itms <- itemFrequency(my_groceries, type="absolute")
head(sort(itms, decreasing = TRUE), n=20)

#display a chart of top 20 items
itemFrequencyPlot(my_groceries, topN=20)

#General rule
gen <-apriori(my_groceries, parameter = list(sup = 0.05, conf = 0.01, 
                                             target="rules",minlen=2, maxlen=3))
inspect(gen[1:6])
inspect(head(sort(gen, by ="lift"),5))

#create your own rule
my_basket <- apriori(my_groceries, parameter = list(sup = 0.005, conf = 0.001, 
                                                    target="rules",minlen=2, maxlen=3),
                     appearance = list(rhs=c("soda","yogurt"), default ="lhs"))
inspect(my_basket[1:20])
inspect(head(sort(my_basket, by ="lift"),10))

milk_beef <- apriori(my_groceries, parameter = list(sup = 0.0045, conf = 0.2, 
                                                    target="rules",minlen=2, maxlen=3),
                     appearance = list(rhs=c("beef","whole milk"), default ="lhs"))

inspect(milk_beef[1:10])

#sort by lift
inspect(head(sort(milk_beef, by ="lift"),5))
#sort by support and confidence
inspect(sort(sort(milk_beef, by ="support"),by ="confidence")[1:5])


####rule for other vegetables
vegetables <- apriori(my_groceries, parameter = list(sup = 0.0045, conf = 0.2, 
                                                    target="rules",minlen=2, maxlen=3),
                     appearance = list(rhs=c("other vegetables"), default ="lhs"))

inspect(vegetables[1:10])
#sort by lift
inspect(head(sort(vegetables, by ="lift"),5))


###rule for bee
root_vegetables <- apriori(my_groceries, parameter = list(sup = 0.0045, conf = 0.2, 
                                                     target="rules",minlen=2, maxlen=3),
                      appearance = list(rhs=c("other vegetables"), default ="lhs"))

inspect(root_vegetables[1:10])
#sort by lift
inspect(head(sort(root_vegetables, by ="lift"),5))

###rolls burns
buns  <- apriori(my_groceries, parameter = list(sup = 0.0145, conf = 0.2, 
                                                     target="rules",minlen=2, maxlen=3),
                      appearance = list(rhs=c("rolls/buns"), default ="lhs"))

inspect(buns[1:10])
#sort by lift
inspect(head(sort(buns, by ="lift"),5))


###bottled water 
water  <- apriori(my_groceries, parameter = list(sup = 0.0045, conf = 0.2, 
                                                      target="rules",minlen=2, maxlen=3),
                       appearance = list(rhs=c("bottled water"), default ="lhs"))

inspect(water  [1:10])
#sort by lift
inspect(head(sort(water , by ="lift"),5))

##tropical fruit
tropical <- apriori(my_groceries, parameter = list(sup = 0.0045, conf = 0.2, 
                                                     target="rules",minlen=2, maxlen=3),
                      appearance = list(rhs=c("tropical fruit"), default ="lhs"))

inspect(tropical [1:10])


#sort by lift
inspect(head(sort(tropical, by ="lift"),5))

###root vegetables
root_vegetables <- apriori(my_groceries, parameter = list(sup = 0.0045, conf = 0.2, 
                                                          target="rules",minlen=2, maxlen=3),
                           appearance = list(rhs=c("root vegetables"), default ="lhs"))

inspect(root_vegetables[1:10])
#sort by lift
inspect(head(sort(root_vegetables, by ="lift"),5))











