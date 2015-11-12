__author__ = 'Tom'

# import pycuda.autoinit
# import pycuda.driver as drv
# import pycuda.gpuarray as gpuarray
# import numpy as np
#
# import skcuda.linalg as culinalg
# import skcuda.misc as cumisc
# culinalg.init()
#
# # Double precision is only supported by devices with compute
# # capability >= 1.3:
# import string
# import scikits.cuda.cula as cula
# demo_types = [np.float32, np.complex64]
# if cula._libcula_toolkit == 'premium' and \
#        cumisc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
#     demo_types.extend([np.float64, np.complex128])
#
# for t in demo_types:
#     print('Testing pinv for type ' + str(np.dtype(t)))
#     a = np.asarray((np.random.rand(50, 50)-0.5)/10, t)
#     a_gpu = gpuarray.to_gpu(a)
#     a_inv_gpu = culinalg.pinv(a_gpu)
#
#     print('Success status: ', np.allclose(np.linalg.pinv(a), a_inv_gpu.get(),
# 		                      atol=1e-2))
#     print('Maximum error: ', np.max(np.abs(np.linalg.pinv(a)-a_inv_gpu.get())))
#     print('')

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
a_doubled = (2*a_gpu).get()
print(a_doubled)
print(a_gpu)