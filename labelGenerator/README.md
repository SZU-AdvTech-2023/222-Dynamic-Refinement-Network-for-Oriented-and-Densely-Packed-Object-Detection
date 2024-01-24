# instructions
1.[工具](https://link.zhihu.com/?target=https%3A//github.com/chinakook/labelImg2)label your images, one image will output one .xml file


    <object>
    	<robndbox>
    		<cx>602.1491</cx>
    		<cy>523.8509</cy>
    		<w>93.0</w>
    		<h>105.0</h>
    		<angle>0.54</angle>
    	</robndbox>
    </object>
	
Where the Angle is 0° at 12 o 'clock, clockwise, with a maximum of 179.99999° (rotated 180°, equivalent to no rotation), this is represented in terms of π.

2.use this function PascalVOC2coco to convert all "*.xml" file label produced by 1. to a .josn file, also is train data of R-CenterNet.
π is going to be converted into 180, and compute angle loss.
![image](https://pic2.zhimg.com/80/v2-b34dac0e5256cd81d6f0a008cc77308d_720w.jpg)
