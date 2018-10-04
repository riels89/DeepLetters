import caffe
caffe.set_mode_cpu()
net = caffe.Net('edited_model/submit-net.prototxt',
                'model/1miohands-v2.caffemodel',
                caffe.TEST)
layers = net.layers
layer for layer in layers:
    print(net.blobs[])
# net_full_conv = caffe.Net('edited_model/submit-net.prototxt',
#                           'edited_model/1miohands-v2.caffemodel',
#                           caffe.TEST)
#print("test1")
# net.blobs['loss1/ave_pool'] = None
# net.blobs['loss1/conv'] = None
# net.blobs['loss1/relu_conv'] = None
# net.blobs['loss1/fc'] = None
# net.blobs['loss1/relu_fc'] = None
# net.blobs['loss1/drop_fc'] = None
# net.blobs['loss1/SLclassifier'] = None
# net.blobs['loss1/loss'] = None
# net.blobs['loss1/top-1'] = None
# net.blobs['loss1/top-5'] = None
#
# net.blobs['loss2/ave_pool'] = None
# net.blobs['loss2/conv'] = None
# net.blobs['loss2/relu_conv'] = None
# net.blobs['loss2/fc'] = None
# net.blobs['loss2/relu_fc'] = None
# net.blobs['loss2/drop_fc'] = None
# net.blobs['loss2/SLclassifier'] = None
# net.blobs['loss2/loss'] = None
# net.blobs['loss2/top-1'] = None
# net.blobs['loss2/top-5'] = None

#net.save('edited_model/removed_labels.caffemodel')