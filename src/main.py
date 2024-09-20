import os
import cv2
import pickle
import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt

from os.path import dirname, abspath, join
from tqdm import tqdm
from numpy.linalg import norm
from tensorflow.keras import Sequential, Model
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.vgg19 import VGG19

config = configparser.ConfigParser()
config.read(join(
                join(
                    dirname(abspath(__file__)), 'utils')
                , 'config.ini')
            )

class loadModel:
    def __init__(self):
        pass
    
    def begin(self, feature_list_path, filenames_path, architecture):
        
        if architecture == 'resnet50v2':
            from tensorflow.keras.applications.resnet_v2 import preprocess_input
            model = ResNet50V2(weights=config['model']['weights'],include_top=False,input_shape=(224,224,3))
        else:
            from tensorflow.keras.applications.vgg19 import preprocess_input
            model = VGG19(weights=config['model']['weights'],include_top=False,input_shape=(224,224,3))
            
        model.trainable = False

        model = Sequential([
            model,
            GlobalMaxPooling2D()
        ])
        feature_list, filenames = self.after_model(feature_list_path, filenames_path)
        return model, feature_list, filenames, preprocess_input
        
    def after_model(self, feature_list_path, filenames_path):
        feature_list = np.array(pickle.load(open(feature_list_path, 'rb')))
        filenames = pickle.load(open(filenames_path, 'rb'))

        new_filenames = [data.split('/')[-1:] for data in filenames]
        
        return feature_list, new_filenames
        

class features_extraction:
    def __init__(self):
        pass
    
    def get_product_id(self, data_path, product):
        df_data = pd.read_csv(data_path)
        nn = []
        for nama in df_data.values:
            new_nama = str(nama[1]).replace('/', '_').replace('  ', '').replace('"', '').replace('!', '')
            nn.append(new_nama)
        df_data['NewName'] = nn
        df_data.drop(columns=['PdNama'], inplace=True)
        result = [data_num[0] for data_num in df_data.values if product == data_num[1]]
        return result

    def get_prediction(self, query_image, model, feature_list, filenames, preprocess_input): # show_image hanya digunakan untuk proses development, wajib false ketika sudah menjadi api
        base_path = os.getcwd()
        data_path = os.path.join(base_path, 'data')
        product_path = os.path.join(data_path, 'all_product.csv')
        
        resized_image = cv2.resize(query_image, (224, 224))
        expanded_img_array = np.expand_dims(resized_image, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        neighbors = NearestNeighbors(n_neighbors=10,algorithm=config['knn']['algorithm'],metric=config['knn']['metric'])
        neighbors.fit(feature_list)

        distances,indices = neighbors.kneighbors([normalized_result])
            
        pred = []
        
        for i in range(10):
            index = indices[0][i]
            distance = distances[0][i]
            result = filenames[index][0].split('.jpg')[0]
            score = round((1-(distance/1))*100, 2)
            product_name = result
            id_product = self.get_product_id(product_path, product=product_name)
            pred.append([id_product[0], product_name, score])
            
        df = pd.DataFrame(pred, columns=['id_product', 'product_name', 'score'])
        return df
    
    def show_image(self, image_path, title):
        im = cv2.imread(image_path)
        plt.imshow(im)
        plt.title(title)
        plt.axis('off')
        plt.show()