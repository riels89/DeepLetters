python MMdnn/mmdnn/conversion/_script/convert.py -sf caffe -in edited_model/submit-net.prototxt --inputshape 227,227,3 -iw model/1miohands-v2.caffemodel -df tensorflow -om model_weights.npy

MMdnn/mmdnn/conversion/_script/convertToIR.py -f caffe -n edited_model/submit-net.prototxt -w model/1miohands-v2.caffemodel -o caffe_resnet_IR