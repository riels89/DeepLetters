from CTC_solver_w_CNN import lstm

LSTM = lstm(lrs=.001, resume=False, start_epoch=0, epochs=200)

LSTM.train()
