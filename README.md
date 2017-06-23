# mtfld
[Multi-Task Facial Landmark Detection](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html)

This is a recreation of the work performed by Zhang, et al, from the University of Hong Kong. Facial landmark detection is a task that's familiar to most people in the image analysis/processing field. What separates this work from other landmark detection algorithms is the incorporation of additional tasks into the learning process.

These learning tasks force a CNN to produce labels that are related to the landmarks that the CNN is attempting to detect. For this particular dataset, in addition to attempting to regressing the locations of the eyes, nose, and mouth corners, the network is also trained to detect if the subject is smiling, the subject's gender, face direction (pose), and the presence of glasses.

These extra tasks are combined into a cumulative loss function, which boils down to a weighted sum of losses generated from regression and categorical cross-entropy. One of the trickier parts of this process is determining how to weight the various portions of the loss function (the paper does not report the values that were used to obtain their results).
