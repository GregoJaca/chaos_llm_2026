import torch, transformers, numpy
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('transformers', transformers.__version__, 'numpy', numpy.__version__)
print('avail', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('tf32', torch.backends.cuda.matmul.allow_tf32)
x = torch.randn(4096, 4096, device='cuda')
import time
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10): y = x @ x
torch.cuda.synchronize()
dt = (time.time() - t0) / 10
print('fp32 4096^3 matmul: %.3f s -> %.1f TFLOPS' % (dt, 2 * 4096**3 / dt / 1e12))
print('STACK_READY')
