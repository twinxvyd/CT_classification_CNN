# -*- coding: utf-8 -*-

import os
import nibabel as nib
from scipy import ndimage
import pydicom
import numpy as np
import natsort
import matplotlib.pyplot as plt

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    
    scan = scan.get_fdata()
    a = 1
    
    if scan.shape[0] == 512:
        scan = ndimage.rotate(scan, 90, reshape=False)
        scan = scan.transpose(2,0,1)
    #for i, y in enumerate(scan):
    #    plt.imsave(data_path2+str(i)+".png",y, cmap=plt.cm.bone)
    #input(1)
    #for i in np.transpose(scan,(2,0,1)):
    #    print(i.shape)
    #    plt.imshow(i, cmap=plt.cm.bone)
    #    plt.show()

    return scan


def normalize(volume):
    
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    
    #for i, y in enumerate(volume):
    #    plt.imsave(data_path2+str(i)+".png",y, cmap=plt.cm.bone)
    #input(2)
    
    # print("정규화 완")
    return volume
    
def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    img = img.transpose(1,2,0)
    desired_depth = 32
    desired_width = 192
    desired_height = 192
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    #for i, y in enumerate(img.transpose(2,0,1)):
    #    plt.imsave(data_path2+str(i)+".png",y, cmap=plt.cm.bone)
    #input(3)
    
    return img



def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    # print(1)
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

data = "MC_3DCNN/data/"

data_list_train1 = list()
label_list_train1 = list()
data_list_train2 = list()
label_list_train2 = list()
data_list_train3 = list()
label_list_train3 = list()
data_list_train4 = list()
label_list_train4 = list()
data_list_train5 = list()
label_list_train5 = list()
data_list_train6 = list()
label_list_train6 = list()
data_list_train7 = list()
label_list_train7 = list()
data_list_train8 = list()
label_list_train8 = list()
data_list_train9 = list()
label_list_train9 = list()
data_list_train10 = list()
label_list_train10= list()

for i in os.listdir(data):
    print(i)
    for num, y in enumerate(os.listdir(data+i)):
        print(num)
        if num < 10:
            # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
            scan = process_scan(data+i+"/"+y)
            print(scan.shape)
            if i == "covid":
                data_list_train1.append(scan)
                label_list_train1.append(2)
            elif i == "normal":
                data_list_train1.append(scan)
                label_list_train1.append(1)
            else:
                data_list_train1.append(scan)
                label_list_train1.append(3)
        elif 10<= num < 20:
            # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
            scan = process_scan(data+i+"/"+y)
            print(scan.shape)
            if i == "covid":
                data_list_train2.append(scan)
                label_list_train2.append(2)
            elif i == "normal":
                data_list_train2.append(scan)
                label_list_train2.append(1)
            else:
                data_list_train2.append(scan)
                label_list_train2.append(3)
        elif 20<= num < 30:
            # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
            scan = process_scan(data+i+"/"+y)
            print(scan.shape)
            if i == "covid":
                data_list_train3.append(scan)
                label_list_train3.append(2)
            elif i == "normal":
                data_list_train3.append(scan)
                label_list_train3.append(1)
            else:
                data_list_train3.append(scan)
                label_list_train3.append(3)
        elif 30 <= num < 40:
            # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
            scan = process_scan(data+i+"/"+y)
            print(scan.shape)
            if i == "covid":
                data_list_train4.append(scan)
                label_list_train4.append(2)
            elif i == "normal":
                data_list_train4.append(scan)
                label_list_train4.append(1)
            else:
                data_list_train4.append(scan)
                label_list_train4.append(3)
        elif 40 <= num < 50:
            # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
            scan = process_scan(data+i+"/"+y)
            print(scan.shape)
            if i == "covid":
                data_list_train5.append(scan)
                label_list_train5.append(2)
            elif i == "normal":
                data_list_train5.append(scan)
                label_list_train5.append(1)
            else:
                data_list_train5.append(scan)
                label_list_train5.append(3)
        elif 50 <= num < 60:
            # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
            scan = process_scan(data+i+"/"+y)
            print(scan.shape)
            if i == "covid":
                data_list_train6.append(scan)
                label_list_train6.append(2)
            elif i == "normal":
                data_list_train6.append(scan)
                label_list_train6.append(1)
            else:
                data_list_train6.append(scan)
                label_list_train6.append(3)
        elif 60 <= num < 70:
            # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
            scan = process_scan(data+i+"/"+y)
            print(scan.shape)
            if i == "covid":
                data_list_train7.append(scan)
                label_list_train7.append(2)
            elif i == "normal":
                data_list_train7.append(scan)
                label_list_train7.append(1)
            else:
                data_list_train7.append(scan)
                label_list_train7.append(3)
        elif 70 <= num < 80:
            # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
            scan = process_scan(data+i+"/"+y)
            print(scan.shape)
            if i == "covid":
                data_list_train8.append(scan)
                label_list_train8.append(2)
            elif i == "normal":
                data_list_train8.append(scan)
                label_list_train8.append(1)
            else:
                data_list_train8.append(scan)
                label_list_train8.append(3)
        elif 80 <= num < 90:
            # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
            scan = process_scan(data+i+"/"+y)
            print(scan.shape)
            if i == "covid":
                data_list_train9.append(scan)
                label_list_train9.append(2)
            elif i == "normal":
                data_list_train9.append(scan)
                label_list_train9.append(1)
            else:
                data_list_train9.append(scan)
                label_list_train9.append(3)
        else:
            # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
            scan = process_scan(data+i+"/"+y)
            print(scan.shape)
            if i == "covid":
                data_list_train10.append(scan)
                label_list_train10.append(2)
            elif i == "normal":
                data_list_train10.append(scan)
                label_list_train10.append(1)
            else:
                data_list_train10.append(scan)
                label_list_train10.append(3)

    # a = 1
    # for po in np.transpose(scan,(2,0,1)):
    #     plt.imsave(i +"_sample/"+str(a)+ ".png",po, cmap=plt.cm.bone)
    #     a = a+1


data_list_train1 = np.array(data_list_train1)
label_list_train1 = np.array(label_list_train1)
data_list_train2 = np.array(data_list_train2)
label_list_train2 = np.array(label_list_train2)
data_list_train3 = np.array(data_list_train3)
label_list_train3 = np.array(label_list_train3)
data_list_train4 = np.array(data_list_train4)
label_list_train4= np.array(label_list_train4)
data_list_train5 = np.array(data_list_train5)
label_list_train5 = np.array(label_list_train5)
data_list_train6 = np.array(data_list_train6)
label_list_train6 = np.array(label_list_train6)
data_list_train7 = np.array(data_list_train7)
label_list_train7 = np.array(label_list_train7)
data_list_train8 = np.array(data_list_train8)
label_list_train8 = np.array(label_list_train8)
data_list_train9 = np.array(data_list_train9)
label_list_train9 = np.array(label_list_train9)
data_list_train10 = np.array(data_list_train10)
label_list_train10 = np.array(label_list_train10)

print(data_list_train1.shape)
print(label_list_train1.shape)
print(data_list_train2.shape)
print(label_list_train2.shape)
print(data_list_train3.shape)
print(label_list_train3.shape)
print(data_list_train4.shape)
print(label_list_train4.shape)
print(data_list_train5.shape)
print(label_list_train5.shape)
print(data_list_train6.shape)
print(label_list_train6.shape)
print(data_list_train7.shape)
print(label_list_train7.shape)
print(data_list_train8.shape)
print(label_list_train8.shape)
print(data_list_train9.shape)
print(label_list_train9.shape)
print(data_list_train10.shape)
print(label_list_train10.shape)

np.save('train_data_list_1.npy', data_list_train1)
np.save('train_label_list_1.npy', label_list_train1)
np.save('train_data_list_2.npy', data_list_train2)
np.save('train_label_list_2.npy', label_list_train2)
np.save('train_data_list_3.npy', data_list_train3)
np.save('train_label_list_3.npy', label_list_train3)
np.save('train_data_list_4.npy', data_list_train4)
np.save('train_label_list_4.npy', label_list_train4)
np.save('train_data_list_5.npy', data_list_train5)
np.save('train_label_list_5.npy', label_list_train5)
np.save('train_data_list_6.npy', data_list_train6)
np.save('train_label_list_6.npy', label_list_train6)
np.save('train_data_list_7.npy', data_list_train7)
np.save('train_label_list_7.npy', label_list_train7)
np.save('train_data_list_8.npy', data_list_train8)
np.save('train_label_list_8.npy', label_list_train8)
np.save('train_data_list_9.npy', data_list_train9)
np.save('train_label_list_9.npy', label_list_train9)
np.save('data_list_train_10.npy', data_list_train10)
np.save('label_list_train_10.npy', label_list_train10)

            
            
            
            