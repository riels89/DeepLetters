import caffe
caffe.set_mode_cpu()
net = caffe.Net('edited_model/submit-net.prototxt',
                'edited_model/1miohands-v2.caffemodel',
                caffe.TEST)
# net_full_conv = caffe.Net('edited_model/submit-net.prototxt',
#                           'edited_model/1miohands-v2.caffemodel',
#                           caffe.TEST)
#print("test1")
