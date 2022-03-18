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

    parser.add_argument('-f', metavar='filePath', action='store', dest='file', type=str,
                        help='Read in a single file and print the count of mosquito eggs')
    result = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    if os.path.isfile(result.file):
        if result.clearPic:
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
