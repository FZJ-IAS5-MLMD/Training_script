# Training_script

Data parallelism is achieved using the horovod library. It is already installed on Taurus, with the module name being Horovod/0.19.5-fosscuda-2019b-TensorFlow-2.2.0-Python-3.7.4. Horovod has to be linked with a machine learning package (tf, pytorch, keras, etc.) as well as a communication mechanism. Usually it uses MPI for CPU communication and NCCL for GPU communication, though it will default to MPI if NCCL is not available.

PiNN has been updated to allow for training on multiple GPUs. Two modification need to be made to the input script:

1. The data should first be distributed. 
```python
mpt_chunks, trr_chunks, ener_chunks = distribute_data(mpt, trr, ener)
dataset = lambda: load_mimic(mpt_chunks, trr_chunks, ener_chunks, split=split)
```

`mpt`, `trr`, `ener` are lists of the all input files to be distributed among GPUs. The length of each list should be equal, and greater than the number of ranks. The data is split by `distribute_data`, and a subset of the data is returned according to the rank of the process.

2. Update model parameters.
```python
params = {...,
          'parallel': True,
          ...}
```
Most of the changes are in the model functions (`potential_model` and `dipole_model`). If the `parallel` option is set, then the model functions call `pinn.utils.parallelize_model`. It does the following (the order is according to the code in `parallelize_model`):

* Make sure that only rank 0 writes the model checkpoint files, so that multiple workers to do corrupt the files.
* Scale the learning rate by the number of workers.
* Broadcast initial variable states from rank 0 to all other processes. This is necessary to ensure consistent initialization of all workers when training is started. with random weights or restored from a checkpoint. 
* Wrap the tf optimizer around the Horovod `DistributedOptimizer()`. At the end of each iteration, `DistributedOptimizer()` takes care of collecting the gradient from each worker using allreduce, averages those gradients at rank 0, calculates weights and biases, and communicates it back to all other workers.
* Pin a single GPU to the process local rank by setting `session_config.gpu_options.visible_device_list`.