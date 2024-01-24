# detector for rotated-object based on CenterNet

### preface

 ~~~
  ${ROOT}
  |-- backbone
  `-- |-- e2cnn
      |-- dlanet.py
      |-- dlanet_dcn.py
      |-- hourglass.py
  |-- Loss.py
  |-- dataset.py
  |-- train.py
  |-- predict.py
 ~~~

#### train your data
 * label your data use labelGenerator;
 * modify all num_classes to your classes num, and modify the num of hm in your back_bone, such as:
   def DlaNet(num_layers=34, heads = {'hm': your classes num, 'wh': 2, 'ang':1, 'reg': 2}, head_conv=256, plot=False):

