#!/usr/bin/env python3


import os
import time
import logging
import argparse
import math
from math import log10, floor  #code for checking the data

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize

from util.pspnet import PSPNet

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from std_msgs.msg import Float32
#from pspnet_ros.msg import Varinfo  Can not use custom msg with python 3

#cv2.ocl.setUseOpenCL(False)

#cost_low_list =[1,3,5,6]
#cost_med_list = [2,7]
#cost_high_list = [4,8]

#cost_low_list =[3,5,6]  #[3,6]
#cost_med_list = [2,7]
#cost_high_list = [4,8]

#reduced 2 list

cost_low_list =[5]  #asphalt
cost_med_list = [1,2,3,6,7] #dirt sand grass gravel mulch
cost_high_list = [4,8,9] # water bush background



def myhook():
  print("shutdown time!")

def round_to_2(x):
    if x <= 0 :
       return 0
    else :
       return round(x, -int(floor(log10(abs(x))))+1)


class pspnet_node:
    
    def __init__(self):
        rospy.init_node('pspnet_node', anonymous=True)
        


        self.class_num = rospy.get_param('~number_of_class',21)
        self.base_size = rospy.get_param('~base_size',512)
        self.test_h = rospy.get_param('~image_height_net_input',473)
        self.test_w = rospy.get_param('~image_width_net_input',473)
        self.scale = rospy.get_param('~scale',1)
        self.zoom_factor = rospy.get_param('~zoom_factor',1)
        self.gpu_num = rospy.get_param('~number_of_gpu',2)
        self.layer_num = rospy.get_param('~number_of_pspnet_layers',50)
        self.model_path = rospy.get_param('~model_path','exp/rugd/pspnet50/model/train_epoch_100.pth')
        self.colors_path = rospy.get_param('~color_list_path','data/rugd/rugd_colors.txt')
        self.names_path = rospy.get_param('~class_name_list_path','data/rugd/rugd_names.txt')
        self.urf_enable = rospy.get_param('~urf_enable',False)
        self.adaptive_urf_enable = rospy.get_param('~adaptive_urf_enable',False)

        self.is_imgcostmap_pub = rospy.get_param('~publish_image_for_costmap',True)
        self.is_imgleginhibit_pub = rospy.get_param('~publish_image_for_leginhibit',True)
        self.is_semantic_img_pub = rospy.get_param('~publish_semantic_segmentation_image',False)

        self.in_time = rospy.Time() #create time instance since the pspnet takes lots of time to process the image. need to store the input time

        self.cost_low = rospy.get_param('~cost_low',0)
        self.cost_med = rospy.get_param('~cost_med',128)
        self.cost_high = rospy.get_param('~cost_high',255)

        self.invalid_cost =  rospy.get_param('~cost_ur',32)
        self.default_std_dev = rospy.get_param('~default_std_dev',0.0009)

        self.colors = np.loadtxt(self.colors_path).astype('uint8')

        self.prob_multiplier = 10

        self.prob_thresh = rospy.get_param('~probability_treshold',0.5) # 0.5 means 0.5 probability
        self.scaled_prob_thresh = math.ceil(self.prob_thresh*self.prob_multiplier)
        
        self.checkparam()
        rospy.loginfo("Parameter set check completed")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in range(0,self.gpu_num))
        
        value_scale = 255
        self.mean = [0.485, 0.456, 0.406]
        self.mean = [item * value_scale for item in self.mean]
        self.std = [0.229, 0.224, 0.225]
        self.std = [item * value_scale for item in self.std]

        self.prediction_org = np.zeros((1,1,self.class_num),dtype=float)
        self.prediction_w_noise = np.zeros((1,1,self.class_num),dtype=float)
        self.cost_image_nour = np.zeros((1,1),dtype=float)
        
        self.model = PSPNet(layers=self.layer_num, classes=self.class_num, zoom_factor=self.zoom_factor, pretrained=False)
        print(self.model)
        self.model = torch.nn.DataParallel(self.model).cuda()
        cudnn.benchmark = True  # correct location for cudnn.benchmark?
        rospy.loginfo("=> loaded checkpoint '{%s}'",self.model_path)

        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        self.img_receive_flag = False
        
        self.color_img_topic = rospy.get_param('~color_image_topic_name','camera/color/raw')
        self.depth_img_topic = rospy.get_param('~depth_image_topic_name','camera/depth/registered_raw')
        self.depth_caminfo_topic = rospy.get_param('~depth_camera_info_topic_name','camera/depth/camerainfo')
        
        self.image_for_costmap_topic_name = rospy.get_param('~image_for_costmap_topic_name','pspnet_output/cost_image')
        self.image_for_leginhibit_topic_name = rospy.get_param('~image_for_leginhibit_topic_name','pspnet_output/leg_inhibit_image')
        self.semantic_segmentation_image_topic_name = rospy.get_param('~semantic_segmentation_image_topic_name','pspnet_output/image_raw')

        self.publish_sync_depth_topic_name = rospy.get_param('~self.publish_sync_depth_topic_name','pspnet_output/sync_depth_raw')
        
        #self.colorimgsub = rospy.Subscriber(self.color_img_topic,Image,self.color_img_callback, queue_size = 2)
        #self.depthimgsub = rospy.Subscriber(self.depth_img_topic,Image,self.depth_img_callback, queue_size = 2)
        self.depthcam_info_sub = rospy.Subscriber(self.depth_caminfo_topic,CameraInfo,self.camera_info_callback,queue_size=1);
        
        self.colorimgsub = message_filters.Subscriber(self.color_img_topic,Image);
        self.depthimgsub = message_filters.Subscriber(self.depth_img_topic,Image);
        
        self.ats = message_filters.ApproximateTimeSynchronizer([self.colorimgsub, self.depthimgsub], queue_size =2, slop =0.1)
        
        self.ats.registerCallback(self.color_depth_callback)

        self.bridge = CvBridge()

        if(self.is_semantic_img_pub == True):
           #self.image_pub = rospy.Publisher(self.semantic_segmentation_image_topic_name,Image,queue_size = 1)
           self.image_pub = rospy.Publisher(self.semantic_segmentation_image_topic_name+"/compressed",CompressedImage,queue_size = 1)
           if self.urf_enable == True:
              #self.image_pub_noised = rospy.Publisher(self.semantic_segmentation_image_topic_name+"noised",Image,queue_size = 1)
              self.image_pub_noised = rospy.Publisher(self.semantic_segmentation_image_topic_name+"noised/compressed",CompressedImage,queue_size = 1)

        if(self.is_imgcostmap_pub == True):
           self.image_costmap_pub = rospy.Publisher(self.image_for_costmap_topic_name,Image,queue_size = 1)
           if self.urf_enable == True:
              self.invalid_region_image_costmap_pub = rospy.Publisher(self.image_for_costmap_topic_name+"invalid_region",Image,queue_size = 1)

        self.sync_depth_image_pub = rospy.Publisher(self.publish_sync_depth_topic_name,Image,queue_size = 1)

        self.gaussian_std_dev_info_pub = rospy.Publisher('pspnet_output/gaussian_std_dev',Float32,queue_size = 1)
        self.ur_areainfo_pub = rospy.Publisher('pspnet_output/ur_area',Float32,queue_size = 1)
        self.ur_ratioinfo_pub = rospy.Publisher('pspnet_output/ur_ratio',Float32,queue_size = 1)
        self.ur_costinfo_pub = rospy.Publisher('pspnet_output/ur_cost',Float32,queue_size = 1)

        ############### adaptive urf parameter set #####################
        self.TP_ur_cost_rate = 0.4  #(parmeter range : 0 ~ 1)
        self.TP_ur_cost_init = 128
        self.TP_ur_min_area = 5  #(unit : number of pixel)
        self.TP_std_dev_deadzone_min = 0.2  #(parmeter range : 0 ~ 0.2)
        self.TP_std_dev_deadzone_max = 0.3  #(parmeter range : 0.2 ~ 0.4)
        self.TP_adp_std_dev_init = 0.03
        self.TP_std_dev_min = 0.01
        self.TP_std_dev_max = 0.06

        ############### adaptive urf variable set #####################
        self.gaussian_std_dev = self.TP_adp_std_dev_init
        self.gaussian_std_dev_old = self.TP_adp_std_dev_init
        self.TP_std_dev_rate = 0.4
        self.ur_area = 0
        self.ur_ratio = 0
        self.TP_ur_infer_factor = 3
        self.TP_nour_infer_factor = 1
        self.ur_cost = self.TP_ur_cost_init
        self.ur_cost_old = self.TP_ur_cost_init

        self.TestNoiseBase = np.zeros((480,848), dtype="int8")
        for y in range(480):
           for x in range(848):
              if (y + x)%2 == 0:
                 e = 1
              else :
                 e = -1 

              self.TestNoiseBase[y,x] = e 

        rospy.loginfo("Initialization completed")

    def gaussian_noise(self,image_in, mean=0, std_dev=0.1):
        image = np.array(image_in / 255, dtype=float)
        noise = np.random.normal(mean, std_dev, image.shape)
        out = image + noise
        low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
        return out

    def checkparam(self):
        
        if self.class_num < 2:
            rospy.logerr("Increase the number of class")
            rospy.on_shutdown(myhook)
            
        if (not (self.zoom_factor in [1, 2, 4, 8])):
            rospy.logerr("Zoom factor should be 1 or 2 or 4 or 8")
            rospy.on_shutdown(myhook)
            
        if ((self.test_h - 1) % 8 != 0):
            rospy.logerr("(self.test_h -1) % 8 should be 0 ")
            rospy.on_shutdown(myhook)
            
        if ((self.test_w - 1) % 8 != 0):
            rospy.logerr("(self.test_w -1) % 8 should be 0 ")
            rospy.on_shutdown(myhook)
        
        if ((self.layer_num != 50) and (self.layer_num != 100)):
            rospy.logerr("layer number should be 50 or 100 ")
            rospy.on_shutdown(myhook)

        if os.path.isfile(self.model_path):
            rospy.loginfo("=> loading checkpoint '{}'".format(self.model_path))
        else:
            rospy.logerr("=> no checkpoint found at '{}'".format(self.model_path))

    def color_depth_callback(self, color_img_msg, depth_img_msg):
         
        if self.img_receive_flag == False:

           self.img_receive_flag = True

           self.in_time = color_img_msg.header.stamp

           self.color_img = self.bridge.imgmsg_to_cv2(color_img_msg, "bgr8") # if the cv_bridge encoding is "bgr8" then the stored image is "bgr"
           img_for_imshow = self.color_img.copy() # cv2.imshow use the bgr order
           self.depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg)#self.depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "mono16")
        
           self.color_img = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2RGB)


           #cv2.imshow("image callback input", img_for_imshow)
           #cv2.waitKey(0)
           #cv2.destroyAllWindows() 


           #self.color_img = np.float32(self.color_img)

           h, w, c = self.color_img.shape
           
           #ToTensor = transform.ToTensorOnlyimg(self.color_img)
           #self.color_img = ToTensor(self.color_img)
           
           rospy.loginfo("iamge shape height : %d, width : %d, channel : %d",h,w,c)
           rospy.loginfo("numpy to tensor conversion completed")
           
        
    def camera_info_callback(self,msg):
    
        self.depth_scale = 0.001;
        self.fx = msg.K[0];
        self.fy = msg.K[4];
        self.px = msg.K[2];
        self.py = msg.K[5];
    
    def pspnetloop(self):
        if self.img_receive_flag == True:
           self.model.eval()
           rospy.loginfo('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
           batch_time = AverageMeter()
        
           if self.urf_enable == True:
              no_loop_noise = 2
           else:
              no_loop_noise = 1

        
           if (self.adaptive_urf_enable == False): 
               self.gaussian_std_dev = self.default_std_dev  #default : 0.0009

           for loop in range(0,no_loop_noise):
        
              end = time.time()
        
              #input = np.squeeze(self.color_img.numpy(), axis=0)
              #image = np.transpose(input, (1, 2, 0))
              image = self.color_img.copy()

              # add gaussian noise  initial frame does not include gaussian noise
              if loop > 0:
                 image = self.gaussian_noise(image,0 , self.gaussian_std_dev)
              
              #cv2.imshow("noised image", image)
              #cv2.waitKey() 
              h, w, _ = image.shape
              prediction = np.zeros((h, w, self.class_num), dtype=float)
              if(loop == 0):
                 self.prediction_org = np.zeros((h, w, self.class_num), dtype=float)
              else: #loop == 1
                 self.prediction_w_noise = np.zeros((h, w, self.class_num), dtype=float)

              long_size = round(self.scale * self.base_size)
              new_h = long_size
              new_w = long_size
              if h > w:
                 new_w = round(long_size/float(h)*w)
              else:
                 new_h = round(long_size/float(w)*h)

              if(loop == 0):
                 self.prediction_org = np.zeros((new_h, new_w, self.class_num), dtype=float)
              else: #loop == 1
                 self.prediction_w_noise = np.zeros((new_h, new_w, self.class_num), dtype=float)

              image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
              #if loop > 0:
              #   image_scale = self.gaussian_noise(image_scale, 0, self.gaussian_std_dev)
              prediction = self.scale_process(image_scale, h, w)
              if(loop == 0):
                 self.prediction_org = np.copy(prediction)
                 
                 prediction_org = cv2.resize(self.prediction_org, (w, h), interpolation=cv2.INTER_LINEAR)
                 prediction_max = np.argmax(prediction_org, axis=2)
                 gray = np.uint8(prediction_max)
                 color = colorize(gray, self.colors)
                 result_image = np.array(color)
                 result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                 if self.urf_enable == True :
                    self.CostImgSinglePred(prediction, w, h, True)

                 msg = CompressedImage()
                 msg.header.stamp = self.in_time
                 msg.format = "jpeg"
                 msg.data = np.array(cv2.imencode('.jpg', result_image)[1]).tostring()

                 if(self.is_semantic_img_pub == True):
                    self.image_pub.publish(msg)

              else: #loop == 1
                 self.prediction_w_noise = np.copy(prediction)

              #self.CostImgSinglePred(prediction)

              batch_time.update(time.time() - end)
              end = time.time()
           
              rospy.loginfo('>>>>> PSPNET processing time : %.3f sec, loop number: %d',batch_time.val, loop)
           


           if self.urf_enable == True:
              self.CostImgNoisedPred(self.prediction_org, self.prediction_w_noise, w, h, True)
              prediction_org = cv2.resize(self.prediction_org, (w, h), interpolation=cv2.INTER_LINEAR)
              prediction_w_noise = cv2.resize(self.prediction_w_noise, (w, h), interpolation=cv2.INTER_LINEAR)
              prediction_max = np.argmax(prediction_org, axis=2)
              prediction_noised_max = np.argmax(prediction_w_noise, axis=2)

              gray = np.uint8(prediction_max)
              gray_noised = np.uint8(prediction_noised_max)
              color = colorize(gray, self.colors)
              color_noised = colorize(gray_noised, self.colors)

              result_image = np.array(color)
              result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
              result_image_noised = np.array(color_noised)
              result_image_noised = cv2.cvtColor(result_image_noised, cv2.COLOR_RGB2BGR)

              #msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
              #msg_noised = self.bridge.cv2_to_imgmsg(result_image_noised, "bgr8")
              msg = CompressedImage()
              msg.header.stamp = self.in_time
              msg.format = "jpeg"
              msg.data = np.array(cv2.imencode('.jpg', result_image)[1]).tostring()

              msg_noised = CompressedImage()
              msg_noised.header.stamp = self.in_time
              msg_noised.format = "jpeg"
              msg_noised.data = np.array(cv2.imencode('.jpg', result_image_noised)[1]).tostring()

              if(self.is_semantic_img_pub == True):
                 self.image_pub.publish(msg)
                 self.image_pub_noised.publish(msg_noised)
           else:
              self.CostImgSinglePred(self.prediction_org, w, h)
              prediction_org = cv2.resize(self.prediction_org, (w, h), interpolation=cv2.INTER_LINEAR)
              prediction_max = np.argmax(prediction_org, axis=2)

              gray = np.uint8(prediction_max)
              color = colorize(gray, self.colors)
           
              #Convert PIL image file to numpy array
              result_image = np.array(color)
              result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

              #cv2.imshow("result image", result_image)
              #cv2.waitKey()

              #msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
              msg = CompressedImage()
              msg.header.stamp = self.in_time
              msg.format = "jpeg"
              msg.data = np.array(cv2.imencode('.jpg', result_image)[1]).tostring()

              if(self.is_semantic_img_pub == True):
                 self.image_pub.publish(msg)

           msg = Float32()
           msg.data = self.ur_area
           self.ur_areainfo_pub.publish(msg)
       
           msg.data = self.ur_ratio
           self.ur_ratioinfo_pub.publish(msg)

           msg.data = self.ur_cost
           self.ur_costinfo_pub.publish(msg)
           
           msg.data = self.gaussian_std_dev
           self.gaussian_std_dev_info_pub.publish(msg)
           
           self.img_receive_flag = False
           


    def scale_process(self,image_scale, img_h, img_w,stride_rate=2/3):
        #rospy.loginfo('>>>>>>>>>>>>>>>> Start Scale process >>>>>>>>>>>>>>>>')
        ori_h, ori_w, _ = image_scale.shape
        pad_h = max(self.test_h - ori_h, 0)
        pad_w = max(self.test_w - ori_w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image_scale, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.mean)
        new_h, new_w, _ = image.shape
        stride_h = int(np.ceil(self.test_h*stride_rate))
        stride_w = int(np.ceil(self.test_w*stride_rate))
        grid_h = int(np.ceil(float(new_h-self.test_h)/stride_h) + 1)
        grid_w = int(np.ceil(float(new_w-self.test_w)/stride_w) + 1)
        prediction_crop = np.zeros((new_h, new_w, self.class_num), dtype=float)
        count_crop = np.zeros((new_h, new_w), dtype=float)
        rospy.loginfo('>>>>>>>>>>>>>>>> new height : %d, new width : %d, Number of netprocess : %d>>>>>>>>>>>>>>>>',new_h,new_w,grid_h*grid_w)
        
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                s_h = index_h * stride_h
                e_h = min(s_h + self.test_h, new_h)
                s_h = e_h - self.test_h
                s_w = index_w * stride_w
                e_w = min(s_w + self.test_w, new_w)
                s_w = e_w - self.test_w
                image_crop = image[s_h:e_h, s_w:e_w].copy()
                count_crop[s_h:e_h, s_w:e_w] += 1
                prediction_crop[s_h:e_h, s_w:e_w, :] += self.net_process(image_crop, use_std = True)
        prediction_crop /= np.expand_dims(count_crop, 2)
        prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
        #prediction = cv2.resize(prediction_crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        #return prediction
        return prediction_crop    
        
        
    def net_process(self,image, use_std=True, flip=True):

        #rospy.loginfo('>>>>>>>>>>>>>>>> Start Net process >>>>>>>>>>>>>>>>')

        input = torch.from_numpy(image.transpose((2, 0, 1))).float()

        if use_std is False:
            for t, m in zip(input, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(input, self.mean, self.std):
                t.sub_(m).div_(s)

        #print("mean :", self.mean, "std :", self.std)
        
        input_no_cuda = input.clone()
        input_no_cuda = input_no_cuda.unsqueeze(0)

        #print("input_no_cuda check :")
        #print("red :", input_no_cuda[0,0,10,10],"green :", input_no_cuda[0,1,10,10], "blue :", input_no_cuda[0,2,10,10])

        input = input.unsqueeze(0).cuda()  # add additional dimension added dimension is 0

        #print("input.unsqueeze(0).cuda() check :")
        #print("red :", input[0,0,10,10],"green :", input[0,1,10,10], "blue :", input[0,2,10,10])


        if flip:
            input = torch.cat([input, input.flip(3)], 0)
        with torch.no_grad():
            #rospy.loginfo('>>>>>>>>>>>>>>>> Image is inputted to Neural Net>>>>>>>>>>>>>>>>')
            output = self.model(input)
            rospy.loginfo('>>>>>>>>>>>>>>>> Output comes out from Neural Net>>>>>>>>>>>>>>>>')
        _, _, h_i, w_i = input.shape
        _, _, h_o, w_o = output.shape
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        if flip:
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        output = output.data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        return output

    def CostImgNoisedPred(self, prediction, prediction_noised, org_w, org_h, avg_filtering = False):

        batch_time = AverageMeter()
        end = time.time()
        h, w, _ = prediction.shape
        cost_image = np.zeros((h,w), dtype="uint8")
        cost_low_image = np.zeros((h,w), dtype="uint8")
        cost_med_image = np.zeros((h,w), dtype="uint8")
        cost_high_image = np.zeros((h,w), dtype="uint8")
        InvRegionBaseSum = np.zeros((h,w), dtype="uint8")

####################################################### cost low start ################################################################
        SumPred = np.zeros((h,w), dtype="float")
        SumPredNoise = np.zeros((h,w), dtype="float")
        for layer_num in cost_low_list:
           SumPred += prediction[:,:,layer_num]
           SumPredNoise += prediction_noised[:,:,layer_num]
        
        SumPred = SumPred*self.prob_multiplier
        SumPred = np.uint8(SumPred)
        ret,PredThdL = cv2.threshold(SumPred,(self.scaled_prob_thresh-1),1,cv2.THRESH_BINARY)

        SumPredNoise = SumPredNoise*self.prob_multiplier
        SumPredNoise = np.uint8(SumPredNoise)
        SumPredNoiseMask = PredThdL*SumPredNoise  # keep the pixel value of SumPredNoise when value at PredThd is 1, and make other pixels zero
        ret,PredNoiseThd = cv2.threshold(SumPredNoiseMask,(self.scaled_prob_thresh-1),1,cv2.THRESH_BINARY)   
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)

        InterMidImg = cv2.subtract(2*PredThdL,PredNoiseThd)   #if pixel value = 2 (not trustful), if pixel value = 1 (trustful) if pixel value = 0 (undetected)

        ret,ValidRegionBaseL = cv2.threshold(InterMidImg,1,1,cv2.THRESH_TOZERO_INV)  #conversion result : pixel value : 2 --> 0, pixel value : 0 --> 0, 1 --> 1
        ret,cost_low_image = cv2.threshold(ValidRegionBaseL,0,self.cost_low,cv2.THRESH_BINARY) # conversion result : pixel value : 1 --> self.cost_low
  
        ret,InvRegionBaseL = cv2.threshold(InterMidImg,1,1,cv2.THRESH_BINARY) #conversion result : pixel value : 2 --> 1, pixel value : 0, 1 --> 0
        InvRegionBaseSum = cv2.add(InvRegionBaseSum,InvRegionBaseL)

####################################################### cost med start ################################################################

        SumPred = np.zeros((h,w), dtype="float")
        SumPredNoise = np.zeros((h,w), dtype="float")
        for layer_num in cost_med_list:
           SumPred += prediction[:,:,layer_num]
           SumPredNoise += prediction_noised[:,:,layer_num]

        SumPred = SumPred*self.prob_multiplier
        SumPred = np.uint8(SumPred)
        ret,PredThdM = cv2.threshold(SumPred,(self.scaled_prob_thresh-1),1,cv2.THRESH_BINARY)

 
        SumPredNoise = SumPredNoise*self.prob_multiplier
        SumPredNoise = np.uint8(SumPredNoise)
        SumPredNoiseMask = PredThdM*SumPredNoise  # keep the pixel value of SumPredNoise when value at PredThd is 1, and make other pixels zero
        ret,PredNoiseThd = cv2.threshold(SumPredNoiseMask,(self.scaled_prob_thresh-1),1,cv2.THRESH_BINARY)  
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)

        InterMidImg = cv2.subtract(2*PredThdM,PredNoiseThd)   #if pixel value = 2 (not trustful), if pixel value = 1 (trustful) if pixel value = 0 (undetected)

        ret,ValidRegionBaseM = cv2.threshold(InterMidImg,1,1,cv2.THRESH_TOZERO_INV)  #conversion result : pixel value : 2 --> 0, pixel value : 0 --> 0, 1 --> 1
        ret,cost_med_image = cv2.threshold(ValidRegionBaseM,0,self.cost_med,cv2.THRESH_BINARY) # conversion result : pixel value : 1 --> self.cost_med
      
        ret,InvRegionBaseM = cv2.threshold(InterMidImg,1,1,cv2.THRESH_BINARY) #if pixel value > 1, 1 is assigned. Other pixels are 0
        InvRegionBaseSum = cv2.add(InvRegionBaseSum,InvRegionBaseM)

####################################################### cost high start ################################################################

        SumPred = np.zeros((h,w), dtype="float")
        SumPredNoise = np.zeros((h,w), dtype="float")
        for layer_num in cost_high_list:
           SumPred += prediction[:,:,layer_num]
           SumPredNoise += prediction_noised[:,:,layer_num]
        SumPred = SumPred*self.prob_multiplier
        SumPred = np.uint8(SumPred)
        ret,PredThdH = cv2.threshold(SumPred,(self.scaled_prob_thresh-1),1,cv2.THRESH_BINARY)

 
        SumPredNoise = SumPredNoise*self.prob_multiplier
        SumPredNoise = np.uint8(SumPredNoise)
        SumPredNoiseMask = PredThdH*SumPredNoise  # keep the pixel value of SumPredNoise when value at PredThd is 1, and make other pixels zero
        ret,PredNoiseThd = cv2.threshold(SumPredNoiseMask,(self.scaled_prob_thresh-1),1,cv2.THRESH_BINARY)  
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)

        InterMidImg = cv2.subtract(2*PredThdH,PredNoiseThd)   #if pixel value = 2 (not trustful), if pixel value = 1 (trustful) if pixel value = 0 (undetected)

        ret,ValidRegionBaseH = cv2.threshold(InterMidImg,1,1,cv2.THRESH_TOZERO_INV)   #conversion result : pixel value : 2 --> 0, pixel value : 0 --> 0, 1 --> 1
        ret,cost_high_image = cv2.threshold(ValidRegionBaseH,0,self.cost_high,cv2.THRESH_BINARY) # conversion result : pixel value : 1 --> self.cost_high 
        
        ret,InvRegionBaseH = cv2.threshold(InterMidImg,1,1,cv2.THRESH_BINARY) #if pixel value > 1, 1 is assigned. Other pixels are 0
        InvRegionBaseSum = cv2.add(InvRegionBaseSum,InvRegionBaseH)

        invalid_region_image_cost = np.zeros((h,w), dtype="uint8")
        if self.adaptive_urf_enable == True:
            invalid_region_image_cost = self.AdpUrfCost(InvRegionBaseSum, InvRegionBaseL, InvRegionBaseM, InvRegionBaseH, ValidRegionBaseL, ValidRegionBaseM, ValidRegionBaseH, PredThdL, PredThdH)
        else:
            self.ur_cost = self.invalid_cost
            ret,invalid_region_image_cost = cv2.threshold(InvRegionBaseSum,0,self.ur_cost,cv2.THRESH_BINARY) #if pixel value > 0 invalid_cost is assigned 

        cost_image = cv2.add(cost_image,cost_low_image)
        cost_image = cv2.add(cost_image,cost_med_image)
        cost_image = cv2.add(cost_image,cost_high_image)
        cost_image = cv2.add(cost_image,invalid_region_image_cost)
        ###FillUp commad is need to remove the 0 cost #####
        ret,cost_image_FillUp = cv2.threshold(cost_image,0,self.cost_med,cv2.THRESH_BINARY_INV) #if pixel value : 0 --> cost_med  pixel value : 1,2,..255 --> 0
        cost_image = cv2.add(cost_image,cost_image_FillUp)

        if avg_filtering == False:
           cost_image = cv2.resize(cost_image, (org_w, org_h), interpolation=cv2.INTER_NEAREST)
        else :
           cost_image = np.float32(cost_image)
           cost_image_nour = np.float32(self.cost_image_nour)
           cost_image = cv2.add(self.TP_ur_infer_factor*cost_image, self.TP_nour_infer_factor*cost_image_nour)
           cost_image = np.uint8(cost_image/(self.TP_ur_infer_factor+self.TP_nour_infer_factor))
           cost_image = cv2.resize(cost_image, (org_w, org_h), interpolation=cv2.INTER_NEAREST)

        msg = self.bridge.cv2_to_imgmsg(cost_image)
        msg.header.stamp = self.in_time #rospy.Time.now()
        msg.height = org_h
        msg.width = org_w

        invalid_region_image_cost = cv2.resize(invalid_region_image_cost, (org_w, org_h), interpolation=cv2.INTER_NEAREST)
        ### below threshold is just for visualization #####
        ret_1,inv_region_img_visual1 = cv2.threshold(invalid_region_image_cost,(int(self.cost_low)-1),128,cv2.THRESH_BINARY)   # 128 is just for gray color
        ret_2,inv_region_img_visual2 = cv2.threshold(invalid_region_image_cost,(int(self.cost_low)-1),255,cv2.THRESH_BINARY_INV)
        inv_region_img_visual1 = np.uint8(inv_region_img_visual1)
        inv_region_img_visual2 = np.uint8(inv_region_img_visual2)

        inv_region_img_visual = cv2.add(inv_region_img_visual1,inv_region_img_visual2)

        msg_invalid_region = self.bridge.cv2_to_imgmsg(inv_region_img_visual)
        msg_invalid_region.header.stamp = rospy.Time.now()
        msg_invalid_region.height = org_h
        msg_invalid_region.width = org_w

        depth_msg = self.bridge.cv2_to_imgmsg(self.depth_img)
        depth_msg.header.stamp = self.in_time #rospy.Time.now()
        depth_msg.height = org_h
        depth_msg.width = org_w

        if(self.is_imgcostmap_pub == True):
           self.image_costmap_pub.publish(msg)
           self.sync_depth_image_pub.publish(depth_msg)
           if self.urf_enable == True:
              self.invalid_region_image_costmap_pub.publish(msg_invalid_region)
        
        batch_time.update(time.time() - end)
        end = time.time()
           
        rospy.loginfo('>>>>> Post processing time : %.3f sec',batch_time.val)


    def CostImgSinglePred(self, prediction, org_w, org_h, avg_filtering = False):
        
        h, w, _ = prediction.shape
        cost_image = np.zeros((h,w), dtype="uint8")
        SumPred = np.zeros((h,w), dtype="float")
        #print("initial cost image")
        #print(cost_image)
        for layer_num in cost_low_list:
           SumPred += prediction[:,:,layer_num]

        SumPred = SumPred*self.prob_multiplier
        SumPred = np.uint8(SumPred)
        ret,thresh_image = cv2.threshold(SumPred,(self.scaled_prob_thresh-1),self.cost_low,cv2.THRESH_BINARY)   
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)
        cost_image = cv2.add(cost_image,thresh_image)

        SumPred = np.zeros((h,w), dtype="float")
        for layer_num in cost_med_list:
           SumPred += prediction[:,:,layer_num]

        SumPred = SumPred*self.prob_multiplier
        SumPred = np.uint8(SumPred)
        ret,thresh_image = cv2.threshold(SumPred,(self.scaled_prob_thresh-1),self.cost_med,cv2.THRESH_BINARY)    
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)
        cost_image = cv2.add(cost_image,thresh_image)

        #------------ code for checking the data ---------------------

        #file_low_cost = open("low_cost_image.txt","w")
 
        #for i in range(0,h): 
        #   file_low_cost.write("\n") 

        #   for j in range(0,w):  
        #      element = round_to_2(cost_image[i,j])  
        #      element_str = str(element) 
        #      L1 = [element_str, " "] 
        #      file_low_cost.writelines(L1) 

        #file_low_cost.close() 

        #----------- code for checking the data end -------------------

        SumPred = np.zeros((h,w), dtype="float")
        for layer_num in cost_high_list:
           SumPred += prediction[:,:,layer_num]

        SumPred = SumPred*self.prob_multiplier
        SumPred = np.uint8(SumPred)
        ret,thresh_image = cv2.threshold(SumPred,(self.scaled_prob_thresh-1),self.cost_high,cv2.THRESH_BINARY)  
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)
        cost_image = cv2.add(cost_image,thresh_image)
         ###FillUp commad is need to remove the 0 cost #####
        ret,cost_image_FillUp = cv2.threshold(cost_image,0,self.cost_med,cv2.THRESH_BINARY_INV) #if pixel value : 0 --> cost_med  pixel value : 1,2,..255 --> 0
        cost_image = cv2.add(cost_image,cost_image_FillUp)

        if avg_filtering == False:
           cost_image = cv2.resize(cost_image, (org_w, org_h), interpolation=cv2.INTER_NEAREST)
           msg = self.bridge.cv2_to_imgmsg(cost_image)
           msg.header.stamp = self.in_time #rospy.Time.now()
           msg.height = org_h
           msg.width = org_w

           depth_msg = self.bridge.cv2_to_imgmsg(self.depth_img)
           depth_msg.header.stamp = self.in_time #rospy.Time.now()
           depth_msg.height = org_h
           depth_msg.width = org_w

           if(self.is_imgcostmap_pub == True):
              self.image_costmap_pub.publish(msg)
              self.sync_depth_image_pub.publish(depth_msg)
        else :
           self.cost_image_nour = np.float32(cost_image.copy())

    def AdpUrfCost(self, InvRegionBaseSum, InvRegionBaseL, InvRegionBaseM, InvRegionBaseH, ValidRegionBaseL, ValidRegionBaseM, ValidRegionBaseH, PredThdL, PredThdH):
        h, w = InvRegionBaseSum.shape
        w1 = 5
        w2 = w -5
        h1 = int(h - h/3)
        h2 = int(h - h/4)
        h3 = h-5

        #rospy.loginfo("w1 : %d w2 : %d h1 : %d h2 : %d", w1, w2, h1, h2)

        self.ur_area = cv2.countNonZero(InvRegionBaseSum[h1:h3, w1:w2])
        self.ur_ratio = self.ur_area/((h3-h1)*(w2-w1))

        #debug_L_ur_area = cv2.countNonZero(InvRegionBaseL[h1:h3, w1:w2])
        #debug_M_ur_area = cv2.countNonZero(InvRegionBaseM[h1:h3, w1:w2])
        #debug_H_ur_area = cv2.countNonZero(InvRegionBaseH[h1:h3, w1:w2])
           

        #rospy.loginfo("ur_area : %f ur_ratio : %f ur_area_L : %f ur_area_M : %f ur_area_H : %f", self.ur_area, self.ur_ratio, debug_L_ur_area, debug_M_ur_area, debug_H_ur_area)

        self.UrfVarUpdate()

        #_, InvRegLCost = cv2.threshold(InvRegionBaseL,0,int((self.cost_low + self.cost_med)/2),cv2.THRESH_BINARY)
        
        #_, InvRegHCost = cv2.threshold(InvRegionBaseH,0,int((self.cost_med + self.cost_high)/2), cv2.THRESH_BINARY)

        #invRegMCost = np.zeros((h,w), dtype="uint8")

        areaM4L = cv2.countNonZero(ValidRegionBaseM[h2:h3, w1:w2])
        areaH4L = cv2.countNonZero(ValidRegionBaseH[h2:h3, w1:w2])  
        ratio_M = areaM4L/(areaM4L + areaH4L + 0.001)
        ratio_H = areaH4L/(areaM4L + areaH4L + 0.001)

        cost4L = 0
        if (areaM4L < self.TP_ur_min_area) and (areaH4L < self.TP_ur_min_area):
           #rospy.loginfo("InvRegMCost = self.cost_med")
           cost4L = (self.cost_low + self.cost_med)/2
        else :
           cost_balance = ratio_M*self.cost_med + ratio_H*self.cost_high
           #rospy.loginfo("cost_balance : %f ", cost_balance)
           cost4L = int((cost_balance + self.cost_low)/2)

        _, InvRegLCost = cv2.threshold(InvRegionBaseL, 0, cost4L, cv2.THRESH_BINARY)

        
        areaL4M = cv2.countNonZero(ValidRegionBaseL[h2:h3, w1:w2])
        areaH4M = cv2.countNonZero(ValidRegionBaseH[h2:h3, w1:w2])  
        ratio_L = areaL4M/(areaL4M + areaH4M + 0.001)
        ratio_H = areaH4M/(areaL4M + areaH4M + 0.001)

        #rospy.loginfo("areaL4M : %f areaH4M : %f ratio_L : %f ratio_H : %f ", areaL4M, areaH4M, ratio_L, ratio_H)

        cost4M = 0
        if (areaL4M < self.TP_ur_min_area) and (areaH4M < self.TP_ur_min_area):
           #rospy.loginfo("InvRegMCost = self.cost_med")
           cost4M = self.cost_med
        else :
           cost_balance = ratio_L*self.cost_low + ratio_H*self.cost_high
           #rospy.loginfo("cost_balance : %f ", cost_balance)
           cost4M = int((cost_balance + self.cost_med)/2)

        _, InvRegMCost = cv2.threshold(InvRegionBaseM, 0, cost4M, cv2.THRESH_BINARY)


        areaL4H = cv2.countNonZero(ValidRegionBaseL[h2:h3, w1:w2])
        areaM4H = cv2.countNonZero(ValidRegionBaseM[h2:h3, w1:w2])  
        ratio_L = areaM4L/(areaL4H + areaM4H + 0.001)
        ratio_M = areaH4L/(areaL4H + areaM4H + 0.001)

        cost4H = 0
        if (areaL4H < self.TP_ur_min_area) and (areaM4H < self.TP_ur_min_area):
           #rospy.loginfo("InvRegMCost = self.cost_med")
           cost4H = (self.cost_high + self.cost_med)/2
        else :
           cost_balance = ratio_L*self.cost_low + ratio_M*self.cost_med
           #rospy.loginfo("cost_balance : %f ", cost_balance)
           cost4H = int((cost_balance + self.cost_high)/2)

        _, InvRegHCost = cv2.threshold(InvRegionBaseH, 0, cost4H, cv2.THRESH_BINARY)
  
        invalid_region_image_cost =np.zeros((h,w), dtype="uint8")
        invalid_region_image_cost = cv2.add(invalid_region_image_cost,np.uint8(InvRegLCost))
        invalid_region_image_cost = cv2.add(invalid_region_image_cost,np.uint8(InvRegMCost))
        invalid_region_image_cost = cv2.add(invalid_region_image_cost,np.uint8(InvRegHCost))
       
       # _, invalid_region_image_cost = cv2.threshold(InvRegionBaseSum,0,int(self.ur_cost),cv2.THRESH_BINARY) #if pixel value > 0 invalid_cost is assigned 
        
        return invalid_region_image_cost

    def UrfVarUpdate(self):
        
        ur_inc = 0.0
        
        if (self.ur_ratio >= self.TP_std_dev_deadzone_min) and (self.ur_ratio <= self.TP_std_dev_deadzone_max):
            ur_inc = 0.0
        elif (self.ur_ratio <= self.TP_std_dev_deadzone_min) :
            ur_inc = 0.005
        elif (self.ur_ratio >= self.TP_std_dev_deadzone_max) :
            ur_inc = -0.005

        gaussian_std_dev_new = self.gaussian_std_dev + ur_inc

        self.gaussian_std_dev = self.TP_std_dev_rate*gaussian_std_dev_new + (1 - self.TP_std_dev_rate)*self.gaussian_std_dev_old
         
        if (self.gaussian_std_dev >= self.TP_std_dev_max):
            self.gaussian_std_dev = self.TP_std_dev_max

        if (self.gaussian_std_dev <= self.TP_std_dev_min):
            self.gaussian_std_dev = self.TP_std_dev_min

        self.gaussian_std_dev_old = self.gaussian_std_dev

    def UrfCostUpdate(self, PredThdL_InvMask, PredThdM_InvMask, PredThdH_InvMask, PredThdLSub, PredThdMSub, PredThdHSub, w1, w2, h1, h2):

        temp_weg = 0.5
        areaL = cv2.countNonZero(PredThdLSub)
        areaM = cv2.countNonZero(PredThdMSub)
        areaH = cv2.countNonZero(PredThdHSub)
        areaLInv = cv2.countNonZero(PredThdL_InvMask)
        areaMInv = cv2.countNonZero(PredThdM_InvMask)
        areaHInv = cv2.countNonZero(PredThdH_InvMask)
        
        costL = 0
        costM = 0
        costH = 0
        costLInv_detect = 0
        costMInv_detect = 0
        costHInv_detect = 0
        ur_cost_new = self.TP_ur_cost_init
        
        if (areaLInv > (temp_weg*areaL)) or (areaL < self.TP_ur_min_area):
            costL = self.cost_low
            costLInv_detect = 1

        if (areaMInv > (temp_weg*areaM)) or (areaM < self.TP_ur_min_area):
            costM = self.cost_med
            costMInv_detect = 1

        if (areaHInv > (temp_weg*areaH)) or (areaH < self.TP_ur_min_area):
            costH = self.cost_high
            costHInv_detect = 1

        if (costLInv_detect + costMInv_detect + costHInv_detect) == 0:
            ur_cost_new = self.TP_ur_cost_init
        elif (costMInv_detect + costHInv_detect) == 0:
            ur_cost_new = (self.cost_high + self.cost_med)/2
        elif (costLInv_detect + costMInv_detect) == 0:
            ur_cost_new = (self.cost_low + self.cost_med)/2
        elif costLInv_detect == 0:
            ur_cost_new = (self.cost_low + self.cost_med)/2
        elif costMInv_detect == 0:
            ur_cost_new = self.cost_med
        elif costHInv_detect == 0:
            ur_cost_new = (self.cost_high + self.cost_med)/2
        else :     
            ur_cost_new = self.cost_med
        if self.ur_ratio > self.TP_std_dev_deadzone_max*2:
            ur_cost_new = self.TP_ur_cost_init
        self.ur_cost = self.TP_ur_cost_rate*ur_cost_new + (1-self.TP_ur_cost_rate)*self.ur_cost_old

        #rospy.loginfo("areaL : %d areaM : %d areaH : %d ", areaL, areaM, areaH)
        #rospy.loginfo("CostL : %d CostM : %d CostH : %d ur_cost_new : %f ur_cost : %f", costL, costM, costH, ur_cost_new, self.ur_cost)

        self.ur_cost_old = self.ur_cost       

if __name__ == '__main__':

    pspnet = pspnet_node()
    
    #https://roboticsbackend.com/ros-rate-roscpy-roscpp/
    #Note that if the code execution takes more than 100ms, then the Rate will not sleep, and the program will directly go to the next iteration. In this case, the 10Hz frequency is not respected.
    rate = rospy.Rate(10)
    
    while  not rospy.is_shutdown():
    
        pspnet.pspnetloop()
    
        rate.sleep()
