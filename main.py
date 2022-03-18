# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:40:41 2022

@author: 
"""

import argparse
import os
import sys
import countingEggs

blurFlag1 = 'gaussian'
blurFlag2 = 'median'


def main():
    parser = argparse.ArgumentParser(description='Count numbers of mosquito eggs in a picture')

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
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    if os.path.isfile(result.file):
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