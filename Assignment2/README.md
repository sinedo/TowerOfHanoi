# Task documentation: 

## Part 1:

error encountered: 
  

``` bash
Error during SAM2 segmentation: CUDA out of memory. Tried to allocate 2.19 GiB. GPU 0 has a total capacity of 11.87 GiB of which 383.44 MiB is free. Process 10406 has 49.21 MiB memory in use. Process 190234 has 31.34 MiB memory in use. Process 335098 has 10.49 GiB memory in use. Of the allocated memory 8.02 GiB is allocated by PyTorch, and 2.34 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

Try checking if the downsampled video was created properly at: /root/catkin_ws/python_files/train_yolo/dataset_blocks_statics/downsampled_video.mp4

SAM2 segmentation failed. Please try again.
 ```

> SOLUTION: reduce amount of frames until no error.

## Part 2:

naming convention:
- Box 1
- Box 2
- Box 3
- Box 4
- Box 5

| validation metrics |  |  |  |  | |
|-----------|----------|---------|---------|---------|---------|
|Precision: | [0.98527 | 0.98525 | 0.98493 | 0.40904 |0.40387] |
|Recall: | [1 | 1 | 1 | 1 | 1] |
|mAP50-95: | 0.8127511827128874   |  | | |  |
|mAP50: | 0.8141014397089927      |  | | |  |
|  |  |  |  |  | |


problem: box 4 and box 5 were not detected properly. the reason is that the first frame of the video had overlappings of the boxes. so SAM did not detect all boxes for all frames. 

> solution: took another video, made as tight as possible bounding boxes.

| validation metrics|  |  |  |  | |
|-----------|----------|---------|---------|---------|---------|
|Precision: | [0.99427 | 0.86956 | 0.69155 | 0.85655 |0.75168] |
|Recall: | [1 | 0.86111 | 0.875 | 0.71662 | 0.62] |
|mAP50-95: | 0.849647588009893   |  | | |  |
|mAP50: | 0.8538284787049149     |  | | |  |
|Speed: |2.2ms preprocess | 11.2ms inference | 0.0ms loss| 0.7ms postprocess per image| 
|  |  |  |  |  | |


## Part 3
