
The DEtection TRansformer (DETR) architecture, developed by Facebook AI Research, uses a combination of a convolutional neural network (CNN) backbone and a transformer to perform end-to-end object detection. Here is an overview of the dimensions at each key stage of the DETR architecture:

Input Dimension
Input Image: The input to the DETR model is an image. The dimensions of this image are typically (batch_size, 3, height, width), where 3 represents the RGB color channels. For simplicity, let's assume the input image size is (batch_size, 3, 800, 1333) (a common size used for training).
CNN Backbone (e.g., ResNet)
Output of the Backbone:
The backbone (e.g., ResNet-50) processes the input image and outputs a feature map.
Assume the final output feature map has dimensions (batch_size, 2048, h, w). With a typical 32x downsampling, if the input size is (800, 1333), then h = 800/32 = 25 and w = 1333/32 = 42.
So, the feature map size will be (batch_size, 2048, 25, 42).
Flattening and Linear Projection
Flattening and Linear Projection:
The 2D feature map is flattened to a 1D sequence of vectors.
Flattening: (batch_size, 2048, 25, 42) -> (batch_size, 2048, 1050) where 1050 = 25 * 42.
Permutation: (batch_size, 2048, 1050) -> (1050, batch_size, 2048).
Linear projection reduces the dimension from 2048 to 256.
Linear projection: (1050, batch_size, 2048) -> (1050, batch_size, 256).
Positional Encoding
Adding Positional Encoding:
The positional encoding is added to the input of the transformer encoder to retain spatial information.
Dimensions remain (1050, batch_size, 256).
Transformer Encoder
Transformer Encoder:
The encoder consists of multiple layers (e.g., 6 layers).
Each encoder layer maintains the dimension (1050, batch_size, 256).
Object Queries and Transformer Decoder
Object Queries:

A fixed set of learnable positional encodings called "object queries" is used.
Object queries have dimensions (num_queries, batch_size, 256). Typically, num_queries = 100.
So, the object queries dimension is (100, batch_size, 256).
Transformer Decoder:

The decoder consists of multiple layers (e.g., 6 layers).
Each decoder layer processes the object queries and encoder output.
Output dimension of each decoder layer: (100, batch_size, 256).
Prediction Heads
Prediction Heads:
Two linear layers are applied to each output embedding from the decoder to predict class labels and bounding box coordinates.
Class Prediction:
Linear layer maps 256 dimensions to the number of classes (including a "no object" class).
If there are num_classes object classes, the output dimension is (batch_size, 100, num_classes + 1).
Bounding Box Prediction:
Linear layer maps 256 dimensions to 4 (representing (cx, cy, w, h)).
Output dimension: (batch_size, 100, 4).
Summary of Dimensions
Here’s a summary of the dimensions through the DETR architecture:



Input Image: (batch_size, 3, height, width)
Backbone Output: (batch_size, 2048, 25, 42)
Position encoding: (2, 256, 28,38)
Flattened and Projected Features: (1050, batch_size, 256)
After Positional Encoding: (1050, batch_size, 256)
Encoder Output: (1050, batch_size, 256)
Object Queries: (100, batch_size, 256)
Decoder Output: (100, batch_size, 256)
Class Prediction: (batch_size, 100, num_classes + 1)
Bounding Box Prediction: (batch_size, 100, 4)


These dimensions ensure that DETR can process an image, encode it through a transformer, and output predictions for object classes and bounding boxes.