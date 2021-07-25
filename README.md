# ML-KMeans-Unsupervised-Learning-Algorithm-Implementation

In this project, I developed the k-means method and tested it on the provided dataset, which consists of a set of 2-D points.
For selecting the initial cluster centers, I used two distinct stratergies.

### Strategy 1: 
Choose the initial centers at random from the available samples. 

### Strategy 2: 
Choose the first center at random; for the i-th center (i>1), choose a sample (among all available samples) with the greatest average distance to all previous (i-1) centers.

I put my implementation to the test on the given data, with k clusters ranging from 2 to 10.
The value of the objective function vs. the number of clusters k was plotted. 
