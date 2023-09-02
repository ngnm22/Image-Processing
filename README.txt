How to run:
1. Run the single python code "processing.py"
2. In the command prompt you can write python processing.py 
3. Yolo video with object detection is attached 

This project uses image processing to improve image frames from videos taken around durham. We use a validation test to test our ground truth images in comparison to our corrupted image. We then use our test set image to find a yolo object detection score. We next use both methods to assess our best final results in our conclusion.
Given our test images we used several techniques to improve the quality of our images. To do this we will be using OpenCV functions, we first dewarped our image to correct their distortions. Next, we removed noise from our image to smoothen it and remove “black dots” and then proceeded to change the contrast and brightness of our image and finish off by sharpening the image to reduce the blurriness from the noise removal. To assess how good the image processing is we will be comparing a file of corrupted images with validation images to see how good the dewarping and noise removal is. We also use yolo object detection to assess the contrast and brightness functions. Though both methods are useful for testing our program, no noise removal, for example, would increase our object detection but would obviously not be good when comparing images. 
