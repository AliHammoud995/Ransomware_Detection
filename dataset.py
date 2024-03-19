import pandas as pd
import os
import math
from keras.preprocessing.image import ImageDataGenerator
import shutil


class prepare_data:
  def __init__(self,data_path,img_path,class_index):
    self.data_path=data_path
    self.img_path=img_path
    self.class_index=class_index


  def split_train_test_val(self,df_train, df_test, df_val, classe,split_ratio_test,split_ratio_val):

    images = os.listdir(os.path.join(self.img_path,classe))
    images = [classe+"/"+elem for elem in images]
    index = self.class_index[classe]

    dim = len(images)
    size_test = math.floor(dim*split_ratio_test)
    size_val = math.floor((dim) * split_ratio_val)

    df_test['img_code'] = images[:size_test]
    df_val['img_code'] = images[size_test:size_test+size_val]
    df_train['img_code'] = images[size_test+size_val:]

    df_test['target'] = classe
    df_val['target'] =classe
    df_train['target'] = classe
    print(classe, df_train.shape, df_test.shape, dim)

    return df_train, df_test, df_val

  
  def create_csv_data(self):

    df_train_complete = pd.DataFrame(columns = ['img_code','target'])
    df_test_complete = pd.DataFrame(columns = ['img_code','target'])
    df_val_complete = pd.DataFrame(columns = ['img_code','target'])

    for classe in self.class_index.keys():
      df_train = pd.DataFrame(columns = ['img_code','target'])
      df_test = pd.DataFrame(columns = ['img_code','target'])
      df_val = pd.DataFrame(columns = ['img_code','target'])

      df_train, df_test, df_val = self.split_train_test_val(df_train, df_test, df_val,classe, 0.2,0.2)
      df_train_complete = pd.concat([df_train_complete, df_train], ignore_index = True)
      df_test_complete = pd.concat([df_test_complete, df_test], ignore_index = True)
      df_val_complete =pd.concat([df_val_complete, df_val], ignore_index = True)



    df_train_complete.to_csv(os.path.join(self.data_path+"/csvs/train.csv"))
    df_test_complete.to_csv(os.path.join(self.data_path+"/csvs/test.csv"))
    df_val_complete.to_csv(os.path.join(self.data_path+"/csvs/val.csv"))


