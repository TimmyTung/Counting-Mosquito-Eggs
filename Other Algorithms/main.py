# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:40:41 2022

@author: Zi Jia Tan
"""

import argparse
import os
import sys
import countingEggs

# flag for countingEgg.py to determine what blur to use
blurFlag1 = 'gaussian'
blurFlag2 = 'median'

# main function
def main():
    #setup title
    parser = argparse.ArgumentParser(description='Count numbers of mosquito eggs in a picture')
    
    #setup flags argument for command line 
    parser.add_argument('-c', action='store_true', default=False, dest='clearPic',
                        help="Boolean value for clear picture, default value as False")

    parser.add_argument("-b", action='store_true', default=False, dest='bgNoise',
                        help="Boolean value for background noise, default value as False")

    parser.add_argument("-m", action='store_true', default=False, dest='microNoNoise',
                        help="Use this flag if picture is micro with little background noise")
    
    parser.add_argument("-l", action='store_true', default=False, dest='largeEggs',
                        help="Use this flag if picture has large eggs")
    
    parser.add_argument('-f', metavar='filePath', action='store', dest='file', type=str,
                        help='Read in a single file and print the count of mosquito eggs')
    
    result = parser.parse_args()
    #a statement if the length of command line is too short it exits
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    
    #make sure the file path contains such file
    if os.path.isfile(result.file):
        #statements for each flags that has been flagged and call the subsequent functions
        if result.microNoNoise:
            countingEggs.countmicroLessNoise(result.file)
        elif result.largeEggs:
            countingEggs.countLargeEggs(result.file)
        elif result.clearPic:
            if result.bgNoise:
                countingEggs.count(result.file, blurFlag1)
            else:
                countingEggs.count(result.file, blurFlag2)
        else:
            countingEggs.count(result.file, blurFlag2)
    else:
        print(f"Error: {result.file} does not exist")


if __name__ == '__main__':
    main()
