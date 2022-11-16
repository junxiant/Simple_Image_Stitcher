import glob
import argparse
import pathlib

import numpy
from matplotlib import pyplot as plt
import cv2

def get_coord(img_path):
    '''Function to get coordinates (rows,cols) based on color for each image.
        Median blur to remove noise.
        Use opencv's range for color thresholding, followed by dilation.
        Then use hough circles to detect the circles in the mask image.
        
        Inputs:
        img_path: The path to the individual image.    
        
        Outputs:
        return col_arr: An array of (row,col) based on the number of colored circles detected.
    '''
    blue = [(0,0,200),(20,20,255)] # lower and upper 
    red = [(200,0,0),(255,20,20)]
    dot_colors = [blue, red]
    col = ['blue', 'red']
    col_arr = []
    i=0

    img = cv2.imread(img_path, 1)   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.medianBlur(img, 3)

    for lower, upper in dot_colors:
        index = 0

        # Threshold these colors
        mask = cv2.inRange(blur,lower,upper) 
        
        # Dilalte to increase circle
        kernel = (7,7)
        mask = cv2.dilate(mask,kernel,iterations=1)

        # Use hough circles to detect circles, adjust params accordingly
        circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,5,param1=20,param2=7,
                                minRadius=0,maxRadius=25)    
        
        # If there are circles, find how many circles for each color
        if circles is not None:
            circles = numpy.round(circles[0, :]).astype("int")
            
            # Count 
            for (x, y, r) in circles:
                # cv2.circle(mask, (x, y), 5, (255, 0, 255), -1)
                index = index + 1
                
        # print(f"For {col[i]}")
        # print(f"No. of circles detected:", index)
        i = i + 1
        col_arr.append(index)
        
        # To display the mask
        # plt.imshow(mask)
        # plt.show()
    
    return col_arr


def plot_grid(sorted_coords):
    '''Function to plot the individual images together.
        Since the list is already sorted, it just needs to plot accordingly from the start.
        
        Inputs:
        sorted_coords: A sorted list of images and their coordinates. 
        e.g. [('./Test Data/fumo\\jeKu7769tCCqm5NG.jpg', (1, 1)), ..].
        
        Outputs:
        Function will display and save the final image.
    '''
    rows = cols = sorted_coords[-1][1][0]

    fig = plt.figure(figsize=(12, 12))
    
    # upper_b = (cols*rows) + 1
    
    for i in range(1, len(sorted_coords)+1):
        # Because the x,y started at 1, it needs to do i-1.
        img = cv2.imread(sorted_coords[i-1][0])
        fig.add_subplot(rows, cols, i)
        print(f"Plotting {i} out of {len(sorted_coords)}")

        plt.axis('off')
        plt.imshow(img)
    
    # Get folder name
    p = pathlib.Path(sorted_coords[0][0])
    p = p.parts[0]
    
    # Uncomment this line below to see the grids
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(f"{p}_final_img.jpg",pad_inches=0)
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    args = parser.parse_args()
    
    input_folder = args.input_folder
    
    # Get list of images first
    img_list = glob.glob(f"{input_folder}/*")

    print("Number of images: ", len(img_list))
    # Create dict
    full_coords = dict()
    
    # Put these results inside a dictionary
    for img in img_list:
        col_arr = get_coord(img)
        full_coords[f"{img}"] = (col_arr[0], col_arr[1])
        
        # Blue is row, Red is col
        # full_coords["row"] = col_arr[0]
        # full_coords["col"] = col_arr[1]
        
    # Sort this dictionary based on values (x,y)
    sorted_coords = sorted(full_coords.items(), key=lambda x: x[1])
    
    # print("Full coords", full_coords)
    # print("Sorted coords", sorted_coords)
    
    # Just display plot accordingly
    plot_grid(sorted_coords)

