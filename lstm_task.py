from seq2seq_solver import lstm

LSTM = lstm(lrs=.001, resume=False, start_epoch=0, epochs=200)

LSTM.train()