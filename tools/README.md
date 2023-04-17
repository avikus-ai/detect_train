The `tools` directory contains 
1. A script that can interact with the semantic and instance masks that match the MassMIND dataset
2. A scripts that extract bounding boxes using these masks.

To extract a bounding box, follow these steps:

1. Make sure that the **image, semantic mask, and instance mask** all have the same name with a .png extension.
2. For the instance_mask, use np.unique to find the instance values, and run a for loop to find the coordinates corresponding to instance_mask == i.
3. Substitute the obtained coordinates into the semantic mask to check whether it is the obstacle class (=3) that you are looking for.
4. If correct, find the bbox coordinates with opencv findContours and boundingRect and save them in yolo format.

You can use the provided scripts to automate this process for all files in a directory, and create yolo format bbox labels for each file. The output can be saved to a specified directory.