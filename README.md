# CUDA implementation of a tomographic projection algorithm

The purpose of this repository is to provide a GPU version of a tomographic projection algorithm based on the Siddon algorithm.

The work began with the [OpenMP](https://www.openmp.org/) parallel implementation, for CPUs, of the algorithm from
[Lorenzo Colletta](https://github.com/mmarzolla/3D-CT-projection-openmp.git).
In this repository the aim is to develop an equivalent [CUDA](https://developer.nvidia.com/cuda-toolkit) parallel implementation
for NVIDIA GPUs.

## Author

[@EnryMarch10](https://github.com/EnryMarch10)

## Summary

The main folders of this project and their purposes are:
- `doc`: Contains the project documentation, including my personal bachelor's thesis in
  [Computer Science and Engineering](https://corsi.unibo.it/1cycle/ComputerScienceEngineering).
- `src`: Contains the source code of the project.

## License

[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)