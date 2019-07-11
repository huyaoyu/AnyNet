Work log

# 2019-07-11

The code is tested on the FRC cluster.

The project is hosted on Github and the local folder on the cluster is /data/datasets/yaoyuh/Projects/AnyNet.

The input training and testing images are from the SceneFlow dataset. Only the FlyingThins dataset is used. The input images are saved at /data/datasets/yaoyuh/StereoData/SceneFlow. The original code of AnyNet assumes the folder name which contains the disparity ground truth is 'frames_disparity'. The code is changed to use 'disparity' instead.

A working directory folder is created at /data/datasets/yaoyuh/Projects/AnyNet/WD/InitialTest. To run the training, the scripts to be used are ClusterShell.sh and SLURMScript.sh. These two scripts use the main.py file. To run the test only, TestingShell.sh and SLURMScript_Testing.sh should be used instead. TestingShell.sh will use main_testing.py file which is a variant of the original main.py file.

After doing a test only run, a folder with the name of 'Testing' under the current working directory will be created and filled with result images.

The current testing results show a relative fuzzier disparity map than those obtained by PSMNet. The current result is trained only on FlyingThings dataset for 10 epoches and minibatch 2. The memory consumption for minibatch 2 is about 3.3GB, however, minibatch 4 or larger could not be launched on the same GPU (minibatch 3 is not tested.).

One problem is found when training AnyNet with minibatches. The original code seems to dispatch jobs on multiple GPUs. This is modified that when the model is created, `model = model.cuda()` is used instead of `model = nn.DataParallel(model).cuda()`.
