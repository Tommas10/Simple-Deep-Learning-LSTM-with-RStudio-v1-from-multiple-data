#In my HP commercial laptop Win10, I wrote [ Simple Deep Learning - LSTM algorithm in R Language ] with R Language with RStudio development tool. 
#Under Windows 10.
#Creatd by Tommas Huang 2019-06-18

install.packages("tensorflow")
install.packages("keras")
install.packages("rlist")
install.packages("lambda.r")
install.packages("matrixStats")
install.packages("tidyverse")

library(keras)
library(stringr)
library(rlist)
library(lambda.r)

EMBEDDING_FILES = c(
  "C:/Users/TommasHuang/Documents/Deep Learning - LSTM/Simple LSTM with R/crawl-300d-2M.vec",
  "C:/Users/TommasHuang/Documents/Deep Learning - LSTM/Simple LSTM with R/glove.840B.300d.txt")
NUM_MODELS = 2
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220

load_embeddings = function(path) {
  embedding_index <- new.env(parent = emptyenv())
  con = file(path, "r")
  readLines(con, n = 1)
  while (TRUE) {
    line = readLines(con, n = 1)
    if (length(line)==0){break}
    values <- strsplit(line, ' ', fixed = TRUE)[[1]]
    if (values[[1]] != '') {
      embedding_index[[values[[1]]]] <- as.numeric(values[-1])
    }
  }
  close(con)
  return(embedding_index)
}


build_matrix = function(word_index,path) {
  embedding_index = load_embeddings(path)
  embedding_matrix = matrix(0,length(word_index)+1, 300)
  names=names(word_index)
  for (i in 1:length(word_index)) {
    embed = mget(names[i], envir = embedding_index, mode = "any",
                 inherits = TRUE,ifnotfound= list(NULL))[[1]]
    if (!is.null(embed)) {embedding_matrix[i+1,] = embed}
  }
  return(embedding_matrix)
}


build_model = function(embedding_matrix, num_aux_targets){
  words = layer_input(shape = c(MAX_LEN))
  x = layer_embedding(words, dim(embedding_matrix)[1],dim(embedding_matrix)[2],weights=list(embedding_matrix), trainable=FALSE)  
  x = layer_spatial_dropout_1d(x, rate = .3)
  x = bidirectional(x, layer = layer_cudnn_lstm(units=LSTM_UNITS, return_sequences= TRUE))
  x = bidirectional(x, layer = layer_cudnn_lstm(units=LSTM_UNITS, return_sequences= TRUE))
  hidden = layer_concatenate(list(layer_global_max_pooling_1d(x),layer_global_average_pooling_1d(x)))
  hidden = layer_add(list(hidden, layer_dense(hidden, units=DENSE_HIDDEN_UNITS, activation = 'relu')))
  hidden = layer_add(list(hidden, layer_dense(hidden, units=DENSE_HIDDEN_UNITS, activation = 'relu')))
  result = layer_dense(hidden, units = 1, activation = 'sigmoid')
  aux_result = layer_dense(hidden, units = num_aux_targets, activation='sigmoid')
  model = keras_model(inputs = words, outputs=c(result, aux_result))
  model %>% compile(loss = "binary_crossentropy", optimizer = 'adam')
  return(model)
}

preprocess = function(data){
  data = str_replace_all(data, "[^[:alnum:]]", " ")
}

train = read.csv("C:/Users/TommasHuang/Documents/Deep Learning - LSTM/Simple LSTM with R/train.csv",nrows=10000)
test = read.csv("C:/Users/TommasHuang/Documents/Deep Learning - LSTM/Simple LSTM with R/test.csv", nrow = 1, stringsAsFactors = FALSE, sep = ",")

x_train = preprocess(train$comment_text)
y_train = ifelse(train$target == 0.4,1,0)
y_aux_train = as.matrix(train[,c('target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat')])
x_test = preprocess(test$comment_text)

tokenizer = fit_text_tokenizer(text_tokenizer(), append(x_train, x_test))

x_train = texts_to_sequences(tokenizer, x_train)
x_test = texts_to_sequences(tokenizer, x_test)
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)

embedding_matrix = cbind(build_matrix(tokenizer$word_index, EMBEDDING_FILES[1]),build_matrix(tokenizer$word_index, EMBEDDING_FILES[2]))

checkpoint_predictions = c()
weights = c()

for (model_idx in 1:NUM_MODELS) {
  model = build_model(embedding_matrix, ncol(y_aux_train))
  for (global_epoch in 1:EPOCHS) {
    model %>% fit(
      x_train,
      list(y_train, y_aux_train), 
      batch_size=BATCH_SIZE, 
      epochs=1, 
      verbose=2,
      callbacks=callback_learning_rate_scheduler(function(epoch,lr) {1e-3 * (0.6 ** global_epoch)})
    )
    checkpoint_predictions = cbind(checkpoint_predictions, (model %>% predict(x_test, batch_size=2048))[[1]])
    weights = cbind(weights, 2 ** global_epoch)
  }
}

predictions = rowWeightedMeans(checkpoint_predictions, weights=weights)

submission = data.frame(
  id = test$id,
  prediction = predictions)

write.csv(submission, file = "C:/Users/TommasHuang/Documents/Deep Learning - LSTM/Simple LSTM with R/submission1.csv", row.names=FALSE)
