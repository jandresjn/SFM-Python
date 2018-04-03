# SFM_PARROT_THESIS

Welcome to my sfm project.

I made the repository public because im looking for people who can help me with to correct the bundle adjustment process.

I am following the pipeline of the book "Mastering Opencv 3" in my own way on python.

Structure of the code:
- main.py - Creates the object of my sfm algorithm and can recieve camera parameters manually or with a npz (camera calibration opencv tutorial).

- sfm_neme.py - Is the main class,  it has all the pipeline of the sfm algorithm, the principal function is sfmsolver which uses all the other functions including de bundle adjustment.

- imagesUtilities.py and vtkpointcloud.py are simple classes. The first is a structure of view, which contains principal params, like 2dpoints path camera params etc. And the vtkpointcloud.py is a simple implementation of vtk for showing the 3d points, it recieves a median point of the model, so the camera tries to point to the center of the model.

The process is simple, the program reads all the paths of images camera initial params etc.
find2d3dPointsBase() is the function which creates the first cloud point, it first compare the first image with the following n images ( not all the images) and selects the best knowing the lower inliers ratio and a min match points.

addview() It compare each view with the previous ones (not all, you can select the number, for example the last 10 or less images), and with the max inliers ratio selects the winner. The process for getting new poses is for pnpSolveRansac() .

triangulateAndFind3dPoints() It adjust the data and uses the filter of reprojection error for getting the better points.
findP2dAsociadosAlineados() It is a pretty compact function which searchs the correspondences of points 2d asociated used in solvePnpRansac().

- BundleAdjustment im using the Sba lourakis python wrapper, which is limited and i try to put the information in the same way of the example texts.

-Projects and fun functions are from other bundle adjusment that had similar results but slower. Function project() -reprojection error- , was incompatible with the project function and triangulation of opencv. The project2 was my temporal solution but was even slower.
Conlusion:

The codes have good results with temple set and carzy horse set, but without using bundle adjustment.. Actually they are using de BA but the result shows the model partitioned in diferent layers and scales, for example the temple seems "repeated" in diferent larges and displaced.

- The algorithm uses the fact of the 2d and 3d points are aligned and the logic doens't use so much the keypoints, descriptors and that stuff as in other sfm projects.
- The logic is easy of understand for its configuration in python.
-----------------------------------------------------------------------------------------------
## Requeriments
LIBRARYS USED:
OPENCV3.4 
VTK PYTHON
SBA LOURAKIS PYTHON:
I modified one line of sba 1.6.8 python wich had a bug in a "dylan" function.
Follow the readme instructions, i modified a file in one sba file which wasn't compiling.
https://www.dropbox.com/sh/9ocxlorye9ab2bj/AADWYvHrdooNKAVZdDpULYIoa?dl=0

Remeber read firstly main.py -> sfm_neme.py in sfmsolver function.

Any pull request about the process of using bundle adjusment will be analyzed.






-----------------------------------------IT IS ALL THE PROCESS OF THE GRADUATION PROJECT--------------------------------------

Aerial Topographic Survey Using a Monocular Camera.

The objective of the project is to generate a 3D Map, identifying features of trees in an aerial view using a low cost drone (ARDRONE-PARROT V1).
All this implemented as a package in ROS.

# PIPELINE
1. Implement the navigating system for ARDRONE-PARROT-V1 using ROS.
2. Design of SFM algorithm using Opencv3 and python 2.7.
3. Implement an intelligent classifier and image processing algorithm which counts and estimates the height of trees.
4. Encapsulate all as a package in ROS.
UAO - Cali, Colombia
