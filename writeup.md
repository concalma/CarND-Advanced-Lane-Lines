**Advanced Lane Finding Project**

**Alberto Vigata**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test_calibration.png "Undistorted"
[image2]: ./output_images/undistorted.png "Road Transformed"
[image3]: ./output_images/binarization.png "Binary Example"
[image4]: ./output_images/warping.png "Warp Example"
[image6]: ./output_images/output_example.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[image10]: ./output_images/binary_warped_cold.jpg "Binary warped cold detection"
[image11]: ./output_images/curvature_warp_to_camera_calc.jpg "warp to camera calc"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  
The writeup is provided in markup in the file writeup.md along with the IPython notebook P4.ipynb.
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cell 1 of the IPython notebook 

The calibration images on the camera_cal directory are used to obtain the distortion matrix M and other parameters.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the tests images using the `cv2.undistort()` function and obtained:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
In cell 2 there's the function `img_undistort` cell 2:9. It uses the parameters from calibration to undistort an image. Here is an example.

![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I experimented with multiple combinations of gradients and color transforms as it can be seen on cell 3 of the notebook. I plotted out the output of all these processes to find out what was working and what wasn't as seen here:


![alt text][image3]

I ended settling for a combination of "S channel + sobel X" for my pipeline

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To create the perspective transform I opened the 'straight_lines.png' file **after aplying undistortion** on a picture editor to identify the coordinates of the straight lines onto a trapezoid. This code can be seen on cell with id=4. 

The funciton `warp_binarize` on cell 4:37  takes as inputs an image (`img`) and uses the M matrix found earlier on 4:27.  I chose the hardcode the source and destination points in the following manner:

```
a,b,c,d = [[577,463],[706,464],[1037,675],[268,675]]
src = np.float32([a,b,c,d])

e,f,g,h = [[320,0], [960,0], [960,720], [320,720]]
dst = np.float32([e,f,g,h])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 577, 463      | 320, 0        | 
| 706, 464      | 960, 0      |
| 1037, 675     | 960, 720      |
| 268, 675      | 320, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The line detection logic is encapsulated inside a class with the name of LineDetect() in cell 5 line 30. The main function that performs line detection is the `LineDetect.find_lines` at cell 5:74. It provides a state machine for an initial state `UNLOCKED` where we use the `LineDetect.find_lines_cold` method to do a detection from scratch. When the bootstrap line detection has happened state switches to `ONGOING` and we'll use the method `LineDetect.find_lines_ongoing` at cell 5:117 

##### `LineDetect.find_lines_cold` 
Does a bootstrap line detection by using a windowing technique. A histogram of the bottom part of the picture is done, and the peaks of it identified to center pixel selection windows. An iterative process, by centering subsquent windows on the center of the pixel data is performed. This is a picture of the output of this process:
![alt text][image10]
A second order polynomial fit is then performed on the pixel data to yield A,B and C parameters for the polynomial. `f(y) = A*x^2 + B*x + C` The fit happens in `LineDetect.find_lines_cold` cell 5:273


##### `LineDetect.find_lines_ongoing` 
Defined on cell 5:117 this is similar to `LineDetect.find_lines_cold`. Main difference is that the algorithm is primed with the previous line fit and a windows is appied left and right from the fit to mark pixels belonging to the line. 

##### Line result validation
In order to weed out erronous line detections I look to make sure both curves are somewhat parallel. For this I calculate the standard deviation of the width between every y position of both lanes. Parallel lines should have smaller deviations. The code is at cell 5:105. Through experimentation a threshold of 50 (pixels) seems to be a good value. 


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
In order to calculate the radius of the curvature I use the helper function on cell 5:340 `calculate_curvature` using the formula

`curvature = (1 + 2*A*y + B^2) ^ 1.5 / abs(2*A)`

Because calculation of the curvature is done in 'warped pixel space' I wrote down on a real notebook the transformation equations for the polynomial parameters from warped space to real space. Here's a scan of my notebook:

![alt text][image11]

The new A', B' and C' parameters are the polynomial parameters in real camera space. I calculate them on cell 5:349 and then plug them on the curvature formula in cell 5:353. The output is the curve in meters.

**note** We assume a fixed meter/pixel values as defined in cell 5:343, 30/720 meters/pixel for y dimension, 3.7/700 for x dimension

#####5.1 Measurement smoothing
In order to remove the jitter from individual frame measurements I perfom a exponential moving average of the polynomial parameters in `LineDetect.EMA()` on cell 5:49  with `alpha=0.2`. This seems to provide good *memory* from old parameters and enough latency to preserve the new


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
After having obtained a polynomial fit I draw the polynomials onto the warped image and fill the space in between too. This is done in cell 6:4 method `unwrap`.  After drawing onto this output I use the *inverse* matrix of perspective correction to obtain an *unwarped undistorted* image. An example of this can be seen in this image:


![alt text][image6]

---

###Pipeline (video)
The `Pipeline()` class defined in cell 7:4 contains a `run` method that is performs all the steps in order for our lane finding. 

1. Remove image lens distortion (13:12)
2. Warping and binarization (13:14)
3. Call the `LineDetect()` class to do the lane detection logic. This keeps the state of detection between calls. If `find_lines` fails to obtain a line measurement then `valid=False` and we just draw original undistorted picture. This could be used upstream to know we can't lock on any line measurement. 
                    

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is a video of line detection happening in **warped space**  [link to my warped video](./output_warped.mp4)


Here's a [link to my video result](./output_final.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A particular problem I encountered was that after warping, the line markers far into the horizon have a lot more weight in the polynomial fit than the pixels close to the car. Because of distortion effects camera/lense/sensor noise.... these pixels comes with a lot more `signal noise` and it's hard to use them for polynomial fits. To mitigate this I lowered the horizon of the perspective transformation so visually better pixels were taking into account. This however, makes the curvature being less apparent in warped space.


Also with the current pipeline the challenge videos don't perform particularly well. This is due to:
* Spurious lane finding after binarization due to wall of high contrast in the sun/shade of wall of highway
* Spurious lane finding due to uneven brightness tone in repaved road

I would explore the following to improve on this project:

* **More experimentation is needed on binarization** to explore better thresholding techniques than the one I used 'S channel + sobel x'
* **Better sanity checking of found lines** to make sure they are consistent. Right now only checking for parallelism. Do they have similar curvature? Is the width of the lane reasonable?
* **Explore alternative algos for line finding**. Because we know the final result have to be two similar 2 order polynomials we could do a brute force search with different A,B,C values onto the binarized_warped image. An example algorithm to try:

1. Select candidate A,B,C values
2. Select pixels on picture from poly with A,B,C and a window around them.
3. Obtain matching metric from this calculation (pixel counting, convolution, ...)
4. Add offsets to A,B,C and repeat
5. After this process is done select best matching A,B,C values from the search space and do final polyfit on that region.


