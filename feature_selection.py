import cv2
import os
import numpy as np
from skimage import feature

class Features:
    def __init__(self, coordinates_dataframe, path_to_images, sliding_box_width, common_height, step_size, is_visualized):
        self.__is_vis = is_visualized
        self.__width = sliding_box_width
        self.__height = common_height
        self.__half_width = (sliding_box_width/100)*50
        self.__step_size = step_size
        images = []
        labels = []
        images_names = os.listdir(path_to_images)
        pedestrian_presence = coordinates_dataframe[['filename','x0','x1']].to_numpy()
        for name in images_names:
            images.append(cv2.imread(str(path_to_images)+'/'+name))
            im_number = int(name.replace(".png",''))
            if im_number in pedestrian_presence[:,0]:
                idx = np.where(pedestrian_presence[:,0]==im_number)[0][0]
                labels.append([1, pedestrian_presence.item(idx, 1), pedestrian_presence.item(idx, 2)] )
            else:
                labels.append([0,0,0])   
        self.__images = images
        self.__labels = labels

    def make_features(self):
        parts = []
        parts_labels = []
        for i in range(len(self.__images)):
            for (x, window) in self.__sliding_window(self.__images[i]):
                parts.append(window)
                if self.__labels[i][1] - self.__half_width <= x and self.__labels[i][2] + self.__half_width>= x + self.__width:
                    parts_labels.append(1)
                else:
                    parts_labels.append(0)

                if(self.__is_vis):
                    clone = self.__images[i].copy()   
                    cv2.rectangle(clone, (x, 0), (x + self.__width, self.__height), (0, 255, 255), 2)
                    cv2.imshow("Full Image", clone)
                    cv2.waitKey(60)
        return self.__extract_hog_features(parts), parts_labels

    def show_results(self, predictions, sliding_speed_ms):
        j = 0
        for i in range(len(self.__images)):
            for (x, _) in self.__sliding_window(self.__images[i]):
                clone = self.__images[i].copy()
                if predictions[j]==1:
                    cv2.rectangle(clone, (x, 0), (x + self.__width, self.__height), (0, 255, 0), 2)
                else:
                    cv2.rectangle(clone, (x, 0), (x + self.__width, self.__height), (0, 0, 255), 2)
                cv2.imshow("Image", clone)
                cv2.waitKey(sliding_speed_ms)
                j = j+1
    
    def save_result(self, predictions, folder_to_save):
        j = 0
        for i in range(len(self.__images)):
            clone = self.__images[i].copy()
            is_found = False
            for (x, _) in self.__sliding_window(self.__images[i]):
                if predictions[j]==1:
                    cv2.rectangle(clone, (x, 0), (x + self.__width, self.__height), (0, 255, 0), 2)
                    is_found = True
                j = j+1
            if is_found:
                cv2.imwrite(folder_to_save+ '/' + str(j) + '.png', clone)

    def __sliding_window(self, image):
        image_w = image.shape[1]
        for x in range(0, image_w, self.__step_size):
            window = image[0: self.__height, x: x + self.__width]
            if window.shape[1] < self.__width: 
                break
            yield (x, window)

    def __extract_hog_features(self, images):
        hog_features = []
        for image in images:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hog_feature = feature.hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
            hog_features.append(hog_feature)
        return np.array(hog_features)