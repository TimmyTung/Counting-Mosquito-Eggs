# Counting-Mosquito-Eggs
 This program is a python file that would fit best to run on a terminal shell in python version of **3.5 and above**. 
 This program reads in **an image** with 2 optional flags that allow users to input the clearness and background noise of the image. It is  written such that it runs only a picture and exit right after the output.  
 Below is a help menu of the program:
 ```
 usage: main.py [-h] [-c] [-b] [-f filePath]

Count numbers of mosquito eggs in a picture

optional arguments:
  -h, --help   show this help message and exit
  -c           Boolean value for clear picture, default value as False
  -b           Boolean value for background noise, default value as False
  -m           Use this flag if picture is micro with little background noise
  -l           Use this flag if picture has large eggs
  -f filePath  Read in a single file and print the count of mosquito eggs
 ```
 This is an example of running an image labeled _EggFilter1.tif_:
 ```
 python main.py -c -b -f EggFilter1.tif
 Eggs in the image:  1467
 
 ```
