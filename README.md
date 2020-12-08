

# Sound Classification and Audio Search in Pytorch

## Prerequisite Python Libraries:

   - librosa==0.8.0
   - numpy==1.19.4
   - SoundFile==0.10.3.post1
   - torch==1.1.0
   - torchvision==0.2.2.post3
   - scikit-learn==0.21.2
   - moviepy==1.0.3

## Instructions:

  1- Copy the labelled waves into "data" directory in class-directory format. i.e. 
     each directory is named after one of the classes and contains all instances
     of that class.

  2- Copy all un-labelled waves into "data_unlabelled" directory.

  3- Run the "util.py" script in order to build the Tensor datasets. It augments
     data from "data" directory(contains an argument class_number that controls 
     the minimun number of instances in each class). Then it builds the labelled
     Tensor dataset and in the end it also builds the unlabelled tensor from
     "data_unlabelled" directory.

  4- Run the "train.py" script. It first trains the classifier through supervised
     learning, saves the best weights and then load these weights and trains the
     model again through semi-supervised learning. The 99.25 % accuracy was 
     achieved in the end of training trials.

  5- The "detect.py" script contains two functions: The "detector_frame" function that
     classifies single shots waves in "detect_samples" directory. It must be noted
     that all single shots are fixed with equal 3 seconds length so if it is shorter
     it will be zero-paded and if it is longer a 3 second slice will be taken
     from the sample. The argument "block_length" coresponds to the fixed length
     of single shot.
     The second function "audio_search" takes in an audio or video file and returns
     detections for 3 second single shots.

  6- Some examples have been added to "data", "data_unlabelled" and "detect_samples"
     directories as a reference of how it must be done. 
  
