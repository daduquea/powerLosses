import os
from PIL import Image
import numpy as np

def generator_from_list(path_to_images, list_of_images, path_to_gt, num_classes, height, width, is_train_set = True, batch_size = 1):
    X_return, Y_return = np.empty([batch_size, height, width, 1]), np.empty([batch_size, height, width, num_classes])
    i_sample = 0
    while True:
        random_index = np.random.choice(len(list_of_images))
        file_name = list_of_images[random_index]
        I_img = Image.open(path_to_images + file_name).convert('RGB')
        I = np.array(I_img)/255.

        gt_file_name = file_name.split('.')[0] + '.npy'
        Y = np.load(path_to_gt+gt_file_name)

        X = I[:,:,0]

        X=np.expand_dims(X,axis=2)

        Y = Y[:,:,:-1] # discard channel of background
        
        X_return[i_sample,:,:,:] = X
        Y_return[i_sample,:,:,:] = Y
        i_sample += 1
        if i_sample == batch_size:
            #print(np.unique(X_return), np.unique(Y_return))
            yield X_return,Y_return
            X_return, Y_return = np.empty([batch_size, height, width, 1]), np.empty([batch_size, height, width, num_classes])
            i_sample = 0 



def build_arrays_from_path(path_to_img, path_to_gt):
    y_list = list()
    x_list = list()
    list_of_names = os.listdir(path_to_img)
    list_of_names.sort()
    for filename in list_of_names:
        I_img = Image.open(path_to_img+filename).convert('RGB')
        I = np.array(I_img)/255.
        gt_file_name = filename.split('.')[0] + '.npy'
        Y = np.load(path_to_gt+gt_file_name)
        Y = Y[:,:,:-1]

        x_list.append(I[:,:,0].reshape([I.shape[0], I.shape[1], 1]))
        y_list.append(Y)    

    return np.array(x_list), np.array(y_list)
