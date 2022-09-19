# Neural tangent kernels of deep convolutional neural networks

This repository is the official implementation of [What can be learnt with wide convolutional neural networks?](https://arxiv.org/abs/2208.01003).

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

## Computing learning curves in the teacher-student setting

This script computes the learning curves of deep convolutional neural tangent kernels in a teacher-student setting for kernel regression. In this setup, the target function is a Gaussian random field with covariance given by the teacher kernel and learning is performed with the student kernel via (ridge) regression.

Usage:

```bash
python teacher_student.py --imagesize [size of the input] --patternsizes [list of teacher filter sizes] --filtersizes [list of student filter sizes]
```

Example for a depth-three teacher and a depth-four student with binary filters:

```bash
python teacher_student.py --imagesize 8 --patternsizes 2 2 --filtersizes 2 2 2
```

Notice that deep convolutional neural tangent kernels are very memory intensive. Running the previous script requires up to 200 GB of RAM.
