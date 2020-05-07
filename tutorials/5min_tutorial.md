# Quick Tutorial

This package provides PyTorch dataset classes and some relevant models for 
self-supervised learning for 3D data as described by Blendowski et al. [1]
who, in addition to proposing their own method, implement a 3D version of 
the 2D method described by Doersch et al. [2].

There are two Jupyter notebooks which closely follows the experiments in [1]. The
code was modeled after the code provided in the mentioned [code repository](https://github.com/multimodallearning/miccai19_self_supervision) [1].

[Example Doersch-style [1,2] Notebook](https://nbviewer.jupyter.org/github/jcreinhold/selfsupervised3d/blob/master/tutorials/doersch.ipynb)

[Example Blendowski-style [2] Notebook](https://nbviewer.jupyter.org/github/jcreinhold/selfsupervised3d/blob/master/tutorials/blendowski.ipynb)

[Example Context Encoder-style [3] Notebook](https://nbviewer.jupyter.org/github/jcreinhold/selfsupervised3d/blob/master/tutorials/context.ipynb)

References
---------------

[1] Blendowski, Maximilian, Hannes Nickisch, and Mattias P. Heinrich. "How to Learn from Unlabeled Volume Data: Self-supervised 3D Context Feature Learning." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2019.

[2] Doersch, Carl, Abhinav Gupta, and Alexei A. Efros. "Unsupervised visual representation learning by context prediction." Proceedings of the IEEE international conference on computer vision. 2015.

[3] Pathak, Deepak, et al. "Context encoders: Feature learning by inpainting." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
