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
        self.noised_pspnet_use = rospy.get_param('~noised_pspnet_use',False)

        self.is_imgcostmap_pub = rospy.get_param('~publish_image_for_costmap',True)
        self.is_imgleginhibit_pub = rospy.get_param('~publish_image_for_leginhibit',True)
        self.is_semantic_img_pub = rospy.get_param('~publish_semantic_segmentation_image',False)

        self.in_time = rospy.Time() #create time instance since the pspnet takes lots of time to process the image. need to store the input time

        self.cost_low = rospy.get_param('~cost_low',0)
        self.cost_med = rospy.get_param('~cost_med',128)
        self.cost_high = rospy.get_param('~cost_high',255)

        self.invalid_cost =  rospy.get_param('~cost_ur',32)

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
           self.image_pub = rospy.Publisher(self.semantic_segmentation_image_topic_name,Image,queue_size = 1)
           if self.noised_pspnet_use == True:
              self.image_pub_noised = rospy.Publisher(self.semantic_segmentation_image_topic_name+"noised",Image,queue_size = 1)

        if(self.is_imgcostmap_pub == True):
           self.image_costmap_pub = rospy.Publisher(self.image_for_costmap_topic_name,Image,queue_size = 1)
           if self.noised_pspnet_use == True:
              self.invalid_region_image_costmap_pub = rospy.Publisher(self.image_for_costmap_topic_name+"invalid_region",Image,queue_size = 1)

        self.sync_depth_image_pub = rospy.Publisher(self.publish_sync_depth_topic_name,Image,queue_size = 1)


        rospy.loginfo("Initialization completed")

    def gaussian_noise(self,image_in, mean=0, var=0.01):
        image = np.array(image_in / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
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
        
           if self.noised_pspnet_use == True:
              no_loop_noise = 2
           else:
              no_loop_noise = 1

           gaussian_noise_m = 0
           gaussian_noise_var = 0.0004  #default : 0.0009

           for loop in range(0,no_loop_noise):
        
              end = time.time()
        
              #input = np.squeeze(self.color_img.numpy(), axis=0)
              #image = np.transpose(input, (1, 2, 0))
              image = self.color_img.copy()

              # add gaussian noise  initial frame does not include gaussian noise
              if loop > 0:
                 image = self.gaussian_noise(image, gaussian_noise_m ,gaussian_noise_var)
              
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
            
              image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
              prediction = self.scale_process(image_scale, h, w)
              if(loop == 0):
                 self.prediction_org = np.copy(prediction)
              else: #loop == 1
                 self.prediction_w_noise = np.copy(prediction)

              #self.CostImgSinglePred(prediction)
              

              #print("prediction check :")
              #print("class 0 :",prediction[10,10,0],"class 1 :",prediction[10,10,1],"class 2 :",prediction[10,10,2],"class 3 :",prediction[10,10,3],"class 4 :",prediction[10,10,4],"class 5 :",prediction[10,10,5],"class 6 :",prediction[10,10,6],"class 7 :",prediction[10,10,7],"class 8 :",prediction[10,10,8])

              batch_time.update(time.time() - end)
              end = time.time()
           
              rospy.loginfo('>>>>> PSPNET processing time : %.3f sec, loop number: %d',batch_time.val, loop)
           


           if self.noised_pspnet_use == True:
              self.CostImgNoisedPred(self.prediction_org, self.prediction_w_noise)
              prediction_max = np.argmax(self.prediction_org, axis=2)
              prediction_noised_max = np.argmax(self.prediction_w_noise, axis=2)

              gray = np.uint8(prediction_max)
              gray_noised = np.uint8(prediction_noised_max)
              color = colorize(gray, self.colors)
              color_noised = colorize(gray_noised, self.colors)

              result_image = np.array(color)
              result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
              result_image_noised = np.array(color_noised)
              result_image_noised = cv2.cvtColor(result_image_noised, cv2.COLOR_RGB2BGR)

              msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
              msg_noised = self.bridge.cv2_to_imgmsg(result_image_noised, "bgr8")

              if(self.is_semantic_img_pub == True):
                 self.image_pub.publish(msg)
                 self.image_pub_noised.publish(msg_noised)
           else:
              self.CostImgSinglePred(self.prediction_org)
              prediction_max = np.argmax(self.prediction_org, axis=2)

              #print("prediction argmax check :",prediction_max[10,10])


              #------------ code for checking the data ---------------------

              #file_prediction = open("semgmented_prediction_image.txt","w")
 
              #for i in range(0,h): 
              #   file_prediction.write("\n") 

              #   for j in range(0,w):  
              #      element = round_to_2(prediction_max[i,j])  
              #      element_str = str(element) 
              #      L1 = [element_str, " "] 
              #      file_prediction.writelines(L1) 

              #file_prediction.close() 

              #----------- code for checking the data end -------------------
              gray = np.uint8(prediction_max)
              color = colorize(gray, self.colors)
           
              #Convert PIL image file to numpy array
              result_image = np.array(color)
              result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

              #cv2.imshow("result image", result_image)
              #cv2.waitKey()

              msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
              if(self.is_semantic_img_pub == True):
                 self.image_pub.publish(msg)


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
        prediction = cv2.resize(prediction_crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        return prediction    
        
        
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

    def CostImgNoisedPred(self, prediction, prediction_noised):

        h, w, _ = prediction.shape
        cost_image = np.zeros((h,w), dtype="uint8")
        cost_low_image = np.zeros((h,w), dtype="uint16")
        cost_med_image = np.zeros((h,w), dtype="uint16")
        cost_high_image = np.zeros((h,w), dtype="uint16")
        invalid_region_image = np.zeros((h,w), dtype="uint16")

        for layer_num in cost_low_list:
           single_predict = prediction[:,:,layer_num]
           single_predict = single_predict*self.prob_multiplier
           single_predict = np.uint8(single_predict)
           ret,thresh_image = cv2.threshold(single_predict,0.2*self.prob_multiplier,1,cv2.THRESH_BINARY)

           single_predict_noised = prediction_noised[:,:,layer_num]
           single_predict_noised = single_predict_noised*self.prob_multiplier
           single_predict_noised = np.uint8(single_predict_noised)
           single_predict_noised_masked = thresh_image*single_predict_noised  # keep the pixel value of single_predict_noised when value at thresh_image is 1, and make other pixels zero
           ret,thresh_image_noised = cv2.threshold(single_predict_noised_masked,self.scaled_prob_thresh,1,cv2.THRESH_BINARY)   
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)

           intermid_image = cv2.subtract(2*thresh_image,thresh_image_noised)   #if pixel value = 2 (not trustful), if pixel value = 1 (trustful) if pixel value = 0 (undetected)

           ret,thresh_image_cost = cv2.threshold(intermid_image,1,255,cv2.THRESH_TOZERO_INV)
           ret,thresh_image_cost = cv2.threshold(thresh_image_cost,0,self.cost_low,cv2.THRESH_BINARY)
           thresh_image_cost = np.uint16(thresh_image_cost)                      
           cost_low_image = cv2.add(cost_low_image,thresh_image_cost)

           ret,thresh_image_invalid = cv2.threshold(intermid_image,1,self.invalid_cost,cv2.THRESH_BINARY)
           thresh_image_invalid = np.uint16(thresh_image_invalid)
           invalid_region_image = cv2.add(invalid_region_image,thresh_image_invalid)

        ret,cost_low_image = cv2.threshold(cost_low_image,(self.cost_low-1),self.cost_low,cv2.THRESH_BINARY)

        for layer_num in cost_med_list:
           single_predict = prediction[:,:,layer_num]
           single_predict = single_predict*self.prob_multiplier
           single_predict = np.uint8(single_predict)
           ret,thresh_image = cv2.threshold(single_predict,0.2*self.prob_multiplier,1,cv2.THRESH_BINARY)

           single_predict_noised = prediction_noised[:,:,layer_num]
           single_predict_noised = single_predict_noised*self.prob_multiplier
           single_predict_noised = np.uint8(single_predict_noised)
           single_predict_noised_masked = thresh_image*single_predict_noised  # keep the pixel value of single_predict_noised when value at thresh_image is 1, and make other pixels zero
           ret,thresh_image_noised = cv2.threshold(single_predict_noised_masked,self.scaled_prob_thresh,1,cv2.THRESH_BINARY)   
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)

           intermid_image = cv2.subtract(2*thresh_image,thresh_image_noised)   #if pixel value = 2 (not trustful), if pixel value = 1 (trustful) if pixel value = 0 (undetected)

           ret,thresh_image_cost = cv2.threshold(intermid_image,1,255,cv2.THRESH_TOZERO_INV)
           ret,thresh_image_cost = cv2.threshold(thresh_image_cost,0,self.cost_med,cv2.THRESH_BINARY)
           thresh_image_cost = np.uint16(thresh_image_cost)                     
           cost_med_image = cv2.add(cost_med_image,thresh_image_cost)

           ret,thresh_image_invalid = cv2.threshold(intermid_image,1,self.invalid_cost,cv2.THRESH_BINARY)
           thresh_image_invalid = np.uint16(thresh_image_invalid)
           invalid_region_image = cv2.add(invalid_region_image,thresh_image_invalid)

        ret,cost_med_image = cv2.threshold(cost_med_image,(self.cost_med-1),self.cost_med,cv2.THRESH_BINARY)

        for layer_num in cost_high_list:
           single_predict = prediction[:,:,layer_num]
           single_predict = single_predict*self.prob_multiplier
           single_predict = np.uint8(single_predict)
           ret,thresh_image = cv2.threshold(single_predict,0.2*self.prob_multiplier,1,cv2.THRESH_BINARY)

           single_predict_noised = prediction_noised[:,:,layer_num]
           single_predict_noised = single_predict_noised*self.prob_multiplier
           single_predict_noised = np.uint8(single_predict_noised)
           single_predict_noised_masked = thresh_image*single_predict_noised  # keep the pixel value of single_predict_noised when value at thresh_image is 1, and make other pixels zero
           ret,thresh_image_noised = cv2.threshold(single_predict_noised_masked,self.scaled_prob_thresh,1,cv2.THRESH_BINARY)   
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)

           intermid_image = cv2.subtract(2*thresh_image,thresh_image_noised)  #if pixel value = 2 (not trustful), if pixel value = 1 (trustful) if pixel value = 0 (undetected)

           ret,thresh_image_cost = cv2.threshold(intermid_image,1,255,cv2.THRESH_TOZERO_INV)
           ret,thresh_image_cost = cv2.threshold(thresh_image_cost,0,self.cost_high,cv2.THRESH_BINARY)
           thresh_image_cost = np.uint16(thresh_image_cost)                      
           cost_high_image = cv2.add(cost_high_image,thresh_image_cost)

           ret,thresh_image_invalid = cv2.threshold(intermid_image,1,self.invalid_cost,cv2.THRESH_BINARY)
           thresh_image_invalid = np.uint16(thresh_image_invalid)
           invalid_region_image = cv2.add(invalid_region_image,thresh_image_invalid)

        ret,cost_high_image = cv2.threshold(cost_high_image,(self.cost_high-1),self.cost_high,cv2.THRESH_BINARY)

        cost_low_image = np.uint8(cost_low_image)
        cost_med_image = np.uint8(cost_med_image)
        cost_high_image = np.uint8(cost_high_image)
        cost_image = cv2.add(cost_image,cost_low_image)
        cost_image = cv2.add(cost_image,cost_med_image)
        cost_image = cv2.add(cost_image,cost_high_image)
        ret,invalid_region_image_cost = cv2.threshold(invalid_region_image,(self.invalid_cost-1),self.invalid_cost,cv2.THRESH_BINARY)
        invalid_region_image_cost = np.uint8(invalid_region_image_cost)
        cost_image = cv2.add(cost_image,invalid_region_image_cost)

        msg = self.bridge.cv2_to_imgmsg(cost_image)
        msg.header.stamp = self.in_time #rospy.Time.now()
        msg.height = h
        msg.width = w


        ### below threshold is just for visualization #####
        ret_1,inv_region_img_visual1 = cv2.threshold(invalid_region_image,(self.invalid_cost-1),128,cv2.THRESH_BINARY)   # 128 is just for gray color
        ret_2,inv_region_img_visual2 = cv2.threshold(invalid_region_image,(self.invalid_cost-1),255,cv2.THRESH_BINARY_INV)
        inv_region_img_visual1 = np.uint8(inv_region_img_visual1)
        inv_region_img_visual2 = np.uint8(inv_region_img_visual2)

        inv_region_img_visual = cv2.add(inv_region_img_visual1,inv_region_img_visual2)

        msg_invalid_region = self.bridge.cv2_to_imgmsg(inv_region_img_visual)
        msg_invalid_region.header.stamp = rospy.Time.now()
        msg_invalid_region.height = h
        msg_invalid_region.width = w

        depth_msg = self.bridge.cv2_to_imgmsg(self.depth_img)
        depth_msg.header.stamp = self.in_time #rospy.Time.now()
        depth_msg.height = h
        depth_msg.width = w

        if(self.is_imgcostmap_pub == True):
           self.image_costmap_pub.publish(msg)
           self.sync_depth_image_pub.publish(depth_msg)
           if self.noised_pspnet_use == True:
              self.invalid_region_image_costmap_pub.publish(msg_invalid_region)

    def CostImgSinglePred(self, prediction):
        
        h, w, _ = prediction.shape
        cost_image = np.zeros((h,w), dtype="uint8")

        #print("initial cost image")
        #print(cost_image)
        for layer_num in cost_low_list:
           single_prediction = prediction[:,:,layer_num]
           single_prediction = single_prediction*self.prob_multiplier
           single_prediction = np.uint8(single_prediction)
           ret,thresh_image = cv2.threshold(single_prediction,self.scaled_prob_thresh,self.cost_low,cv2.THRESH_BINARY)   
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)
           cost_image = cv2.add(cost_image,thresh_image)

        for layer_num in cost_med_list:
           single_prediction = prediction[:,:,layer_num]
           single_prediction = single_prediction*self.prob_multiplier
           single_prediction = np.uint8(single_prediction)
           ret,thresh_image = cv2.threshold(single_prediction,self.scaled_prob_thresh,self.cost_med,cv2.THRESH_BINARY)   
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

        for layer_num in cost_high_list:
           single_prediction = prediction[:,:,layer_num]
           single_prediction = single_prediction*self.prob_multiplier
           single_prediction = np.uint8(single_prediction)
           ret,thresh_image = cv2.threshold(single_prediction,self.scaled_prob_thresh,self.cost_high,cv2.THRESH_BINARY)   
                                                                     #cv2.threshold(input_image,threshold,max_value,threshold_type)
           cost_image = cv2.add(cost_image,thresh_image)


        #------------ code for checking the data ---------------------

        #file_high_cost = open("high_cost_image.txt","w")
 
        #for i in range(0,h): 
        #   file_high_cost.write("\n") 

        #   for j in range(0,w):  
        #      element = round_to_2(cost_image[i,j])  
        #      element_str = str(element) 
        #      L1 = [element_str, " "] 
        #      file_high_cost.writelines(L1) 

        #file_high_cost.close() 

        #----------- code for checking the data end -------------------



        msg = self.bridge.cv2_to_imgmsg(cost_image)
        msg.header.stamp = self.in_time #rospy.Time.now()
        msg.height = h
        msg.width = w

        depth_msg = self.bridge.cv2_to_imgmsg(self.depth_img)
        depth_msg.header.stamp = self.in_time #rospy.Time.now()
        depth_msg.height = h
        depth_msg.width = w

        if(self.is_imgcostmap_pub == True):
           self.image_costmap_pub.publish(msg)
           self.sync_depth_image_pub.publish(depth_msg)


if __name__ == '__main__':

    pspnet = pspnet_node()
    
    #https://roboticsbackend.com/ros-rate-roscpy-roscpp/
    #Note that if the code execution takes more than 100ms, then the Rate will not sleep, and the program will directly go to the next iteration. In this case, the 10Hz frequency is not respected.
    rate = rospy.Rate(10)
    
    while  not rospy.is_shutdown():
    
        pspnet.pspnetloop()
    
        rate.sleep()
