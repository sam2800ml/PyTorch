# Transforms 

The use of transforms in pytorch is very important to be able to manipulate data, and make it suitable for the training that we are going to acomplished.
>
what it actually do is that transform the input depending of what we use.
>
we have 2 trasnforms, one is for the input data, and the other target the inout labels, to make them one hot encoding, and that makes everything easier for the model

we have devided the transforms depending of what they do.
## Basic transformations
- ToTensor -> this converts a PIL or a numpy array in to a pytorch tensor normalizing the pixels
- Normalize -> Normalize the tensor of the image using the following equation = (input - mean) / std
- Resize -> resize the input
- CenterCrop -> crop the center of an image
- RandomCrop -> randomly crops a part of the image and uses it as the input
- Pad -> puts a padding on the image 

## Augmented transformations
- RandomHorizontalFlip -> flips horizontal the image
- RandomVerticalFlip -> flips vertical the image
- RandomRotation -> Rotates the image
- ColorJitter -> Changes the brightness, contrast, saturation, and hue of an image.
- RandomAffine -> Applies a combination of affine transformations like rotation, translation, scaling, and shearing.
- RandomResizedCrop -> randomly crops the image
- RandomPerspective -> Applies a random perspective transformation

## Tensor manipulation
- Lambda -> applies a lambda function to the input
- Grayscale -> converts an image to a gray scale

## Convertion transforms 
- ToPILImage -> converts a tensor or numpy array to a PIL image
- Compose -> Chain multiple transformations
- AutoAugment -> Applies an augmented policy for automatic data augmentation
- RandomApply -> Randomly applies a list of transformations

## Advanced Augmentations
- GaussianBlur -> applies gaussian blur to an image
- RandomGrayscale -> Converts an image to gray base on a probability
- ElasticTransform -> Applies elastic deformation to the image
- RandomInvert -> Inverts the color of an image base on a probability 