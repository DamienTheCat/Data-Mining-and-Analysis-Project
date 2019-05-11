library(dplyr)
library(e1071)
library(caret)
library(doSNOW)
library(tree)
library(ggplot2)
library(quanteda)


setwd("C:/Users/shiva/Desktop/DMPA/Project")
#Read the entire dataset
text0 <- read.csv("train.csv", stringsAsFactors = FALSE)
#Extracting the columns we need for the analysis
text1<- data.frame(ID=text0$id, TARGET=text0$target, COMMENT_TEXT=text0$comment_text)

#Adding label "lbl" based on the value of TARGET which specifies the toxicity of a particualr document with a cutoff of 0.5
text1$lbl <- ifelse(text1$TARGET >= 0.5, "Toxic", "Non-Toxic")
text1$lbl <- as.factor(text1$lbl)
proptable <- data.frame(prop.table(table(text1$lbl)))

#Stratified partitioning
indexes1 <- createDataPartition(text1$lbl, p = 0.0015, list = FALSE)
main <- text1[indexes1,]

indexes2 <- createDataPartition(main$lbl, p = 0.7, list = FALSE)
train1 <- main[indexes2,]
test1 <- main[-indexes2,]



train1$COMMENT_TEXT <- as.character(train1$COMMENT_TEXT)
test1$COMMENT_TEXT <- as.character(test1$COMMENT_TEXT)


#Wordcloud 

#For toxic comments
toxic.df <- subset(text1, text1$lbl == "Toxic")
str(toxic.df)
toxic.df$COMMENT_TEXT <- as.character(toxic.df$COMMENT_TEXT)

toxic.token <- tokens(toxic.df$COMMENT_TEXT, what = "word", 
                      remove_numbers = TRUE, remove_punct = TRUE,
                      remove_symbols = TRUE, remove_hyphens = TRUE)

#removing stopwords
toxic.token <- tokens_select(toxic.token, stopwords(), 
                             selection = "remove")

toxic.dfm <- dfm(toxic.token, tolower = FALSE)
col <- sapply(seq(0.1, 1, 0.1), function(x) adjustcolor("red", x))
textplot_wordcloud(toxic.dfm,min_size = 1, max_size = 4, min_count = 3,
                   min.freq = 1,max.words=200, random.order=FALSE, rot.per=0.35,
                   labelcolor = "red", labelsize = 1.5,
                   labeloffset = 0,  color = rev(RColorBrewer::brewer.pal(10, "RdBu")))


#For non toxic comments
toxic.dfn <- subset(text1, text1$lbl == "Non-Toxic")
str(toxic.df)
toxic.dfn$COMMENT_TEXT <- as.character(toxic.dfn$COMMENT_TEXT)

toxic.tokenn <- tokens(toxic.dfn$COMMENT_TEXT, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)

#removing stopwords
toxic.tokenn <- tokens_select(toxic.tokenn, stopwords(), 
                              selection = "remove")

toxic.dfmn <- dfm(toxic.tokenn, tolower = FALSE)
col <- sapply(seq(0.1, 1, 0.1), function(x) adjustcolor("red", x))
textplot_wordcloud(toxic.dfmn,min_size = 0.5, max_size = 4, min_count = 3,
                   min.freq = 1,max.words=200, random.order=FALSE, rot.per=0.35,
                   labelcolor = "red", labelsize = 1.5,
                   labeloffset = 0,  color = rev(RColorBrewer::brewer.pal(10, "RdBu")))


#Wordcloud END


#Preprocessing train
#creating tokens
train.tokens1 <- tokens(train1$COMMENT_TEXT, what = "word", 
                        remove_numbers = TRUE, remove_punct = TRUE,
                        remove_symbols = TRUE, remove_hyphens = TRUE)
#removing stopwords
train.tokens1 <- tokens_select(train.tokens1, stopwords(), 
                               selection = "remove")
#stemming
train.tokens1 <- tokens_wordstem(train.tokens1, language = "english")
# Create our first bag-of-words model.
train.tokens.dfm1 <- dfm(train.tokens1, tolower = FALSE)
#Trimming to remove words occuring less than 0.01% of the time
train.tokens.dfm1 <- dfm_trim(train.tokens.dfm1, min_termfreq = 0.0001, termfreq_type = "prop")


#Preprocessing test
#creating tokens
test.tokens1 <- tokens(test1$COMMENT_TEXT, what = "word", 
                        remove_numbers = TRUE, remove_punct = TRUE,
                        remove_symbols = TRUE, remove_hyphens = TRUE)
#removing stopwords
test.tokens1 <- tokens_select(test.tokens1, stopwords(), 
                               selection = "remove")
#stemming
test.tokens1 <- tokens_wordstem(test.tokens1, language = "english")
# Create our first bag-of-words model.
test.tokens.dfm1 <- dfm(test.tokens1, tolower = FALSE)




#Making a DFM and token matrix
# Time the code execution
start.time <- Sys.time()

# Create a cluster to work on 6 logical cores.
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

# Transform to a matrix and inspect.

train.tokens.matrix <- as.matrix(train.tokens.dfm1)
train.tokens.dfm1 <- cbind(ID = train1$ID, lbl = train1$lbl, data.frame(train.tokens.dfm1))

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution
total.time <- Sys.time() - start.time
total.time

names(train.tokens.dfm1) <- make.names(names(train.tokens.dfm1))



# Use caret to create stratified folds for 10-fold cross validation repeated

# 3 times (i.e., create 30 random stratified samples)
set.seed(12345)

cv.folds <- createMultiFolds(train1$lbl, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         
                         repeats = 3, index = cv.folds)
str(cv.cntrl)





#Tree model
# Time the code execution
start.time <- Sys.time()
# Create a cluster to work on 6 logical cores.

cl <- makeCluster(6, type = "SOCK")

registerDoSNOW(cl)

# As our data is non-trivial in size at this point, use a single decision

# tree alogrithm as our first model. 

rpart.cv.1 <- train(lbl ~ ., data = train.tokens.dfm1, method = "rpart",
                    
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.

stopCluster(cl)

# Total time of execution

total.time <- Sys.time() - start.time

total.time

rpart.cv.1 





#Naive Bayes Classifier
model.nb <- textmodel_nb(train.tokens.dfm1, train1$lbl, prior = "docfreq")
summary(model.nb)
coef(model.nb)

test_matched <- dfm_match(test.tokens.dfm1, features = featnames(train.tokens.dfm1))
Pred1 <- predict(model.nb, newdata = test_matched)
summary(Pred1)
prop.table(table(Pred1))




#Topic Modeling
# MATRIX TAKES Source()
trainvector <- as.vector(train1$COMMENT_TEXT)
testvector <- as.vector(test1$COMMENT_TEXT)

# CREATE SOURCE FOR VECTORS
trainsource <- VectorSource(trainvector)
testsource <- VectorSource(testvector)

# CREATE CORPUS FOR DATA
traincorpus <- Corpus(trainsource)
testcorpus <- Corpus(testsource)

dtmtrain <- DocumentTermMatrix(traincorpus,
                              control = list(stripWhitespace = TRUE,
                                             tolower = TRUE,
                                             stopwords = TRUE,
                                             removePunctuation = TRUE,
                                             removeNumbers = TRUE, sparse = TRUE))
dtmtest <- DocumentTermMatrix(testcorpus,
                               control = list(stripWhitespace = TRUE,
                                              tolower = TRUE,
                                              stopwords = TRUE,
                                              removePunctuation = TRUE,
                                              removeNumbers = TRUE))

raw.sum=apply(dtmtrain,1,FUN=sum)
dtmtrain=dtmtrain[raw.sum!=0,]

# Now for some topics
SEED = sample(1:1000000, 1) 
k = 5  
models <- list(
  CTM       = CTM(dtmtrain, k = k, control = list(seed = SEED, var = list(tol = 10^-4), em = list(tol = 10^-3))),
  VEM       = LDA(dtmtrain, k = k, control = list(seed = SEED)),
  VEM_Fixed = LDA(dtmtrain, k = k, control = list(estimate.alpha = FALSE, seed = SEED)),
  Gibbs     = LDA(dtmtrain, k = k, method = "Gibbs", control = list(seed = SEED, burnin = 1000,
                                                               thin = 100,    iter = 1000))
)

model1 <- LDA(dtmtrain, k = k, method = "Gibbs", control = list(seed = SEED, burnin = 1000,
                                                                thin = 100,    iter = 1000))

str(model1@wordassignments)
str(model1@terms)
lapply(models, terms, 50)




#Working on Twitter data with #trump

str(tweets)
tweets$x <- as.character(tweets$x)


tweets <- read.csv("tweets_trump.csv")
#Preprocessing train
#creating tokens
t.tokens1 <- tokens(tweets$x, what = "word", 
                        remove_numbers = TRUE, remove_punct = TRUE,
                        remove_symbols = TRUE, remove_hyphens = TRUE)
#removing stopwords
t.tokens1 <- tokens_select(t.tokens1, stopwords(), 
                               selection = "remove")
#stemming
t.tokens1 <- tokens_wordstem(t.tokens1, language = "english")
# Create our first bag-of-words model.
t.tokens.dfm1 <- dfm(t.tokens1, tolower = FALSE)

#Using dfm_match to avoid rows with zero entries
t_matched <- dfm_match(t.tokens.dfm1, features = featnames(train.tokens.dfm1))

Predt <- predict(model.nb, newdata = t_matched)
summary(Predt)
prop.table(table(Predt))

