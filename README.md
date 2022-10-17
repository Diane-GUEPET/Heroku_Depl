# Design_An_Autonomous_Vehicle

## Goals
### The use of semantic segmentation for the designing of autonomous vehicle


Autonomous vehicles such as self-driving cars and drones can benefit from automated segmentation. 

For example, self-driving cars can detect drivable regions.


In this work, we will use CNN (Convolutional Neural Networks) to achieve a semantic segmentation.


The deployment of our model and the built of an application which take an image as input and return its segmentation as output will be done with Flask and deploy with Azure or other free cloud application platform as Heroku .


## Semantic segmentation VS Image classification 

As we did image classification in our previous project, I'll use its concepts to define the semantic segmentation.
The following illustrations are worth....

![image](https://user-images.githubusercontent.com/92828445/173925777-74b4e203-1ff2-4efb-a0f6-e45ba6df31b6.png)


![image](https://user-images.githubusercontent.com/92828445/173926169-75fc2bd4-9710-45da-a8f5-aa0cf9842bf9.png)


![image](https://user-images.githubusercontent.com/92828445/173927000-34e25542-969f-4ea2-ae70-e4c520c07cfe.png)


### Transfert learning of image classification on semantic segmentation


![image](https://user-images.githubusercontent.com/92828445/173929169-186b23df-faf9-4415-8261-4db54d342f7c.png)



## Keras for building and training the model (Keras_segmentation)

## Data

#bmp or png format (for the stability of pixels).
#The size of the input image and the segmentation image should be the same.

### Data augmentation (Imgaug)

To avoid an overfitting of our model (case when our data are not sufficient), we use the data augmentation. It consists of adding more versions oof our pictures by (1) changing the color properties like hue, saturation, brightness, etc of the input images, (2) applying transformations such as rotation, scale, and flipping.

![image](https://user-images.githubusercontent.com/92828445/173933563-5dc321ec-3399-4a09-8267-444035a90d53.png)


## Steps

1. Choosing of two models from the list of segmentation models. Training on our preproceed dataset. 

2. Pickling the best model and save it

3. Using Flask to build our app (vs code) add our model and the function of prediction

4. Deploy our App using Heroku (The size of our App is not eligible to a deployment for free)  
