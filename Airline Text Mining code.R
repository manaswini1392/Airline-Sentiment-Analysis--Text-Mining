# Install Important packages
install.packages("tm")
install.packages("SnowballC")
install.packages("dplyr")
install.packages("nnet")
install.packages("caTools")
install.packages("Liblinear")
install.packages('slam')
install.packages("readr")
install.packages("NLP")
#install_github("cpsievert/LDAvisData")
install.packages("lda")
install.packages("LDAvis")


# Install Important Library
library(tm)
library(SnowballC)
library(dplyr)
library(nnet)
library(caTools)
library(LiblineaR)
library(randomForest)
library(slam)
library(topicmodels)
library(wordcloud)
library(NLP)
library(openNLP)
library(lda)
library(LDAvis)


#Reading the data
airline_tweets=read.csv('Tweetsr.csv',stringsAsFactors = F)

# Creating a subset with required variables for modelling
airline_tweets = select(airline_tweets,airline_sentiment,Sentiment,airline,text)


# create 2 subset to get positive and negative sentiment
positive_senti = subset(airline_tweets, airline_sentiment == 'positive');
negative_senti= subset(airline_tweets, airline_sentiment == 'negative');
dim(positive_senti); #2363 are positive sentiments
dim(negative_senti); #9178 are negative sentiments

###################################### TEXT PREPROCESSING START#########################################################
# Remove the unwanted symbols from text
airline_tweets$text = gsub("^@\\w+ *", "", airline_tweets$text)

# To analyze the corpus text,generate a function
#analyseText = function(text_to_analyse)
 # {
# Create a corpus for collecting the text documents
corpus=Corpus(VectorSource(airline_tweets$text))
#corpus = Corpus(VectorSource(text_to_analyse))
#Text Preprocessing to clean the data
# There are few words which are occuring frequently but 
#not providing any relevant information
wordsToRemove = c('get', 'cant', 'can', 'now', 'just', 'will', 
                  'dont', 'ive', 'got', 'much' ,'each','isnt','unit','airline','virgin',
          'southwestair','americanair','usairway','unit','virginameria','jetblue','fli','amp','dfw','tri','flt')
                                                                                              
corpus=tm_map(corpus,tolower)
corpus=tm_map(corpus,PlainTextDocument)
corpus=tm_map(corpus,removePunctuation)
corpus=tm_map(corpus,removeNumbers)
corpus=tm_map(corpus,stripWhitespace)
corpus=tm_map(corpus,removeWords,stopwords('en'))
corpus = tm_map(corpus,removeWords, wordsToRemove)
corpus=tm_map(corpus,stemDocument)

#Create a Document termMatrix
dtm=DocumentTermMatrix(corpus)

# To inspect the document term matrix 
inspect(dtm[1:5,1:50])

# To find the most freq term, which are repeated more than 20 times
findFreqTerms(dtm, lowfreq =20)

#Remove the sparse terms from the documnt.This makes a matrix that is 10% empty space
Sparse_senti=removeSparseTerms(dtm,0.98)

#converting the sparse terms to data frame
SparseDF=as.data.frame(as.matrix(Sparse_senti))
colnames(SparseDF) = make.names(colnames(SparseDF))
#return(SparseDF)
#}
###################################### TEXT PREPROCESSING ENDS#########################################################
SparseDF$sentiment=airline_tweets$Sentiment

###################################### WORD CLOUDS START#########################################################
#To find the negative words after text preprocessing
negative_words = analyseText(negative_senti$text)
dim(negative_words); #80 words repeated with certain frequency across negative snetiments

#To find the positive words after text preprocessing
positive_words = analyseText(positive_senti$text)
dim(positive_words)#47 words repeated with certain frequency across positve sentiments


# To find the no of times a negative word appears in a document
neg_words = colSums(negative_words)
neg_words = neg_words[order(neg_words, decreasing = T)]

head(neg_words)

# To find the no of times a positve word appears in a document
pos_words = colSums(positive_words)
pos_words = neg_words[order(pos_words, decreasing = T)]
head(pos_words)

#we have observed in SAS as well,thank and thanks 
#appearing as 2 different columns, we will merge both columns into one
pos_words[1] = pos_words[1] + pos_words[2]
pos_words = pos_words[-2]

wordcloud = wordcloud(corpus, min.freq = 100, scale = c(10,0.5), colors = brewer.pal(6, "Dark2"), random.color = TRUE, random.order = FALSE)
# word clouds
par(mfrow = c(1,2))

neg_wc =wordcloud(freq = as.vector(neg_words), words = names(neg_words),random.order = FALSE,
          random.color = FALSE, colors = brewer.pal(9, 'Reds')[4:9])

pos_wc = wordcloud(freq = as.vector(pos_words), words = names(pos_words),random.order = FALSE,
          random.color = FALSE, colors = brewer.pal(9, 'BuPu')[4:9])

###################################### TEXT PREPROCESSING ENDS#########################################################

###################################### LDA AND LDAVis CODE START#########################################################
#corpus <- gsub("'", "", corpus)  # remove apostrophes
#corpus <- gsub("[[:punct:]]", " ", corpus)  # replace punctuation with space
#corpus <- gsub("[[:cntrl:]]", " ", corpus)  # replace control characters with space
#corpus <- gsub("^[[:space:]]+", "", corpus) # remove whitespace at beginning of documents
#corpus <- gsub("[[:space:]]+$", "", corpus) # remove whitespace at end of documents
#corpus <- tolower(corpus)  # force to lowercase

doc.list <- strsplit(as.character(corpus), "\\s+")

#create term table 
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)
vocab <- names(term.table)

index = ""
# now put the documents into the format required by the lda package:
get.terms <- function(x) {  
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (2,000)
W <- length(vocab)
doc.length <- sapply(documents, function(x) sum(x[2, ]))
N <- sum(doc.length)
term.frequency <- as.integer(term.table)

# MCMC and model tuning parameters:
K <- 10
G <- 100
alpha <- 0.02
eta <- 0.02
library(lda)
set.seed(357)
t1 <- Sys.time()

# Apply lda to find the relevant topics in documents
ldaout <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                      num.iterations = G, alpha = alpha, 
                                      eta = eta, initial = NULL, burnin = 0,
                                      compute.log.likelihood = TRUE)



print(t2)

t2 <- Sys.time()
t2 - t1

#visualize the lda 
theta <- t(apply(ldaout$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(ldaout$topics) + eta, 2, function(x) x/sum(x)))


AirlineReviews <- list(phi = phi,
                       theta = theta,
                       doc.length = doc.length,
                       vocab = vocab,
                       term.frequency = term.frequency)


json <- createJSON(phi = AirlineReviews$phi, 
                   theta = AirlineReviews$theta, 
                   doc.length = AirlineReviews$doc.length, 
                   vocab = AirlineReviews$vocab, 
                   term.frequency = AirlineReviews$term.frequency)


help("serVis")
install.packages("servr")
#serVis(json, out.dir = 'visual', open.browser = FALSE)
serVis(json, out.dir = 'visual2', open.browser = interactive())


################################# LDA AND LDAVis ENDS #################################################################

################################# Data Modelling #################################################################

# Predicting the overall sentiments for airlines

#Modelling with multinominal  regression as we have 3 sentiments -Positive,negative and neutral
table(airline_tweets$Sentiment)

# We have 9178 as negative reviews, which is more than neutral and 
#favorable.We can choose the level of our outcome that we wish to use as our baseline and specify this in relevel function
SparseDF$sentiment=as.factor(as.character(SparseDF$sentiment))
SparseDF$sentiment=relevel(SparseDF$sentiment,ref="-1")

#creating a test sample
set.seed(123)
split=sample.split(airline_tweets$Sentiment,SplitRatio = 0.75)

#creating the train and test
Train_airline=subset(SparseDF,split==T)
Test_airline=subset(SparseDF,split==F)

#Multinominal Modelling on Training Data
tweetmult=multinom(sentiment~.,data=Train_airline)
#predicting with multinomial training
predictmultinomTrain=predict(tweetmult,newdata=Train_airline)
#confusion matrix
table(Train_airline$sentiment,predictmultinomTrain)
#Accuracy on Training data
(6335+480+837)/(6335+341+207+1684+480+161+839+96+837)
# 69.67

#Predicting the accuracy of modelling on testing data set
predictmultinomTest=predict(tweetmult,newdata=Test_airline)
#confusion matrix test
table(Test_airline$sentiment,predictmultinomTest)
#Accuracy on test data set
(2124+171+272)/(2124+102+68+563+171+41+301+18+272)
#70.13

#Using liblinear regularized models and checking the accuracy
#we have to handle prediction varible seperately in this model
Train_Input=Train_airline[,1:167]
Train_Decision=Train_airline[,167]

#test
Test_Input=Test_airline[,1:167]
Test_Decision=Test_airline[,168]


#implementing liblinear for 7 models in a forloop and checking
Models=c(0:7)
#checking with different costs for coefficients
ModelCosts=c(1000,1,0.001)
bestCost=NA
bestACC=0
bestType=NA
for(Model in Models)
{
  for(costs in ModelCosts)
  {
    accuracy=LiblineaR(Train_Input,target = Train_Decision,type=Model,cost=costs,
                       bias=TRUE,cross=5,verbose = F)
  cat("Results for cost=",costs,",accuracy=",accuracy,"\n",sep = "");
  if(accuracy>bestACC)
  {
    bestCost=costs
    bestACC=accuracy
    bestType=Model
  }
    }
}
cat("best Model",best_type,"best cost",bestCost,"best accuracy",bestACC)

#constructing the best model

bestModel=LiblineaR(Train_Input,target = Train_Decision,type=bestType,cost=bestCost
                    ,bias = TRUE,verbose = F)
#predicting test
pr=FALSE
if(bestType==0||bestType==7)pr=TRUE
predictTest=predict(bestModel,Test_Input,proba=pr);


#saving the results
results=table(Test_Decision,predictTest$predictions)

#Accuracy table is printed below

#TestDecision   -1    0    1
# -1 3262  275  134
#  0   586  546  108
#  1   302  140  503
# (3262+546+503)/nrow(TestInput)
# accuracy of 0.736168

#Random Forest
Train_airline$Sentiment=as.factor(Train_airline$Sentiment)
Test_airline$Sentiment=as.factor(Test_airline$Sentiment)
tweetRF<-randomForest(Sentiment ~., data = Train_airline,nodesize = 25,ntree=200)
predictRF = predict(tweetRF,newdata= Test_airline)
table(Test_airline$Sentiment,predictRF)

################################# Data Modelling ENDS #################################################################


