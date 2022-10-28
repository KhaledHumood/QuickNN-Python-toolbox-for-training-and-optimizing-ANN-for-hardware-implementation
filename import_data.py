import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io
from skimage.measure import block_reduce
from skimage.transform import resize, rescale


def import_data(image_size, res_inputs, data_path, save_path, file_name):
    

    no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    train_data = np.loadtxt(data_path + "mnist_train.csv", 
                            delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv", 
                           delimiter=",") 
    
    
    train_imgs = np.asfarray(train_data[:, 1:])
    test_imgs = np.asfarray(test_data[:, 1:]) 
    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])
    
    train_imgs2 = np.zeros((60000, image_pixels));
    test_imgs2 = np.zeros((10000, image_pixels));
    train_imgs3 = np.zeros((60000, image_pixels));
    test_imgs3 = np.zeros((10000, image_pixels));
    
    ######## resize images to 20x20 ########
    for i in range(60000):
        img_temp = train_imgs[i].reshape((28,28))
        img_temp = resize(img_temp, (image_size, image_size), anti_aliasing=True)
        img_temp = img_temp.reshape((1,image_pixels))
        train_imgs2[i,:] = img_temp;
    train_imgs2 = np.rint(train_imgs2) 
    for i in range(10000):
        img_temp = test_imgs[i].reshape((28,28))
        img_temp = resize(img_temp, (image_size, image_size), anti_aliasing=True)
        img_temp = img_temp.reshape((1,image_pixels))
        test_imgs2[i,:] = img_temp;
    test_imgs2 = np.rint(test_imgs2) 
    ######## quantize inputs to 4 bits ########
    new_res = (2 ** res_inputs) - 1 
    for i in range(60000):
        for j in range(image_pixels):
            scale = (255)/(new_res)
            train_imgs3[i][j] = train_imgs2[i][j]/scale
    train_imgs3 = np.rint(train_imgs3)        
    for i in range(10000):
        for j in range(image_pixels):
            scale = (255)/(new_res)
            test_imgs3[i][j] = test_imgs2[i][j]/scale
    test_imgs3 = np.rint(test_imgs3) 
    
    
    ###### map pixed values to 0 to 1 ########
    fac = 1 / new_res
    train_imgs3 = train_imgs3 * fac 
    test_imgs3 = test_imgs3 * fac 
    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])
    
    
    with open(save_path + file_name, "bw") as fh:
        data = (train_imgs3, 
                test_imgs3, 
                train_labels,
                test_labels)
        pickle.dump(data, fh)
        
        
    for i in range(10):
        img = train_imgs3[i].reshape((image_size,image_size))
        plt.imshow(img, cmap="Greys")
        plt.show()    

