# mtfld
[Multi-Task Facial Landmark Detection](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html)

This is a recreation of the work performed by Zhang, et al, from the University of Hong Kong. Facial landmark detection is a task that's familiar to most people in the image analysis/processing field. What separates this work from other landmark detection algorithms is the incorporation of additional tasks into the learning process.

These learning tasks force a CNN to produce labels that are related to the landmarks that the CNN is attempting to detect. For this particular dataset, in addition to attempting to regressing the locations of the eyes, nose, and mouth corners, the network is also trained to detect if the subject is smiling, the subject's gender, face direction (pose), and the presence of glasses.

These extra tasks are combined into a cumulative loss function, which boils down to a weighted sum of losses generated from regression and categorical cross-entropy. One of the trickier parts of this process is determining how to weight the various portions of the loss function (the paper does not report the values that were used to obtain their results).


The paper notes that their CNN made use of weight matrices that did NOT share weight values across spatial dimensions, but my first implementation carries on by assuming that only one weight matrix is used at each convolutional layer. The code used for the actual learning is a modified version of the MNIST code from the TensorFlow website and is currently being used for testing purposes only. Currently the network adroitly learns how to position the locations of each landmark relative to each other (e.g. the left-eye mark should be to the left of the right-eye mark and both eyes should be above the nose). Data undergoes two alterations before being passed to the network: rotation and translation. These transformations are applied with random values within a range specified by the class that performs the data aggregation ('facial_landmark_detection_data_reader').
