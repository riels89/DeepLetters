import caffe
caffe.set_mode_cpu()
#net = caffe.Net('model/submit-net.prototxt', caffe.TRAIN)
net_full_conv = caffe.Net('model/submit-net.prototxt',
                          'model/1miohands-v2.caffemodel',
                          caffe.TEST)
print("test1")
