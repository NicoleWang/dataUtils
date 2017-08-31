from __future__ import print_function
import numpy as np
import argparse
from PIL import Image, ImageFilter
import time

import sys
sys.path.insert(0,'/home/dingzhonggan/workspace/scene/caffe-optimize/python')
import caffe
import google.protobuf as pb
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter

caffe_config_filename = 'deploy_darknet_nobn.prototxt'
modelname = 'scene_darknet_nobn.caffemodel'
origin_caffe_config_filename = 'deploy_darknet.prototxt'
origin_modelname = 'scene_darknet_gooddata_iter_6000.caffemodel'


net = {}
with open(origin_caffe_config_filename,'r') as fp:
    net = NetParameter()
    pb.text_format.Parse(fp.read(),net)

net_model = caffe.Net(caffe_config_filename,caffe.TEST)

origin_net_model = caffe.Net(origin_caffe_config_filename,origin_modelname,caffe.TEST)

for layer in net.layer:
    if layer.type == 'Scale':
        bn_layer_name = layer.name.replace('scale','bn')
        mean  = origin_net_model.params[bn_layer_name][0].data
        var   = origin_net_model.params[bn_layer_name][1].data
        
        value = origin_net_model.params[bn_layer_name][2].data
        mean = mean / value
        var = var / value

        scale = origin_net_model.params[layer.name][0].data
        bias  = origin_net_model.params[layer.name][1].data
        var = np.sqrt(var + 1e-5)
        temp = scale / var
        temp2 = bias - (scale*mean/var)

        '''
        if 'r' not in layer.name:
            a = net_model.params[layer.name][0].data
            b = net_model.params[layer.name][1].data
            a[:] = temp[:]
            b[:] = temp2[:]
            continue
        '''
        conv_layer_name = layer.name.replace('scale_','')
        a = net_model.params[conv_layer_name][0].data
        b = net_model.params[conv_layer_name][1].data
        conv_weight = origin_net_model.params[conv_layer_name][0].data
        
        for index in xrange(len(a)):
            #a[index,:,:,:] = conv_weight[index,:,:,:] * temp[index]
            if conv_layer_name == "conv1":
                a[index,:,:,:] = conv_weight[index,:,:,:] * temp[index] * 128
            else:
                a[index,:,:,:] = conv_weight[index,:,:,:] * temp[index]
        
        b[:] = temp2[:]

    if layer.name[0:12] == "conv16/data_":
        a = net_model.params[layer.name][0].data
        b = net_model.params[layer.name][1].data
        conv_weight = origin_net_model.params[layer.name][0].data
        conv_bies = origin_net_model.params[layer.name][1].data
        a[:] = conv_weight[:]
        b[:] = conv_bies[:]

net_model.save(modelname)

