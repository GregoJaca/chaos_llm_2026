"""Minimal torch-free loader for .pt files (zip + pickle, torch tensor rebuild).

Format found here: protocol-2 pickle calling
torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride,
                               requires_grad, backward_hooks)
with dtype carried by the persistent-id storage class (e.g. torch.FloatStorage),
raw bytes in '<prefix>/data/<key>' zip entries. Works without torch installed.
"""
import zipfile
import pickle
import io
import numpy as np

TORCH_STORAGE_DTYPES = {
    'FloatStorage': np.float32, 'DoubleStorage': np.float64,
    'HalfStorage': np.float16, 'LongStorage': np.int64,
    'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8,
    'BoolStorage': np.bool_, 'BFloat16Storage': np.uint16,
    'UntypedStorage': None, 'TensorStorage': None,
}


class _StorageType:
    def __init__(self, dtype):
        self.dtype = dtype


class _StubStorage:
    def __init__(self, dtype, numel, key):
        self.dtype = dtype
        self.numel = numel
        self.key = key  # zip entry name


class _StubTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = offset
        self.size = size
        self.stride = stride

    def to_numpy(self, path):
        z = zipfile.ZipFile(path)
        raw = z.read(self.storage.key)
        dt = np.dtype(self.storage.dtype)
        n = int(np.prod(self.size)) if len(self.size) else 1
        base = np.frombuffer(raw, dtype=dt)
        arr = base[self.offset: self.offset + n]
        # honor storage ordering: row-major contiguous strides assumed
        out = arr.reshape(tuple(self.size)).copy()
        while out.ndim > 0 and out.shape[0] == 1 and out.ndim > 2:
            out = out[0]
        return out


class _PtUnpickler(pickle.Unpickler):
    def __init__(self, file, zpath, prefix):
        super().__init__(file)
        self.zpath = zpath
        self.prefix = prefix

    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            import collections
            return collections.OrderedDict
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            def rebuild(storage, offset, size, stride, *a):
                return _StubTensor(storage, offset, tuple(size), tuple(stride))
            return rebuild
        if module == 'torch' and name.endswith('Storage'):
            return _StorageType(TORCH_STORAGE_DTYPES.get(name))
        if module == 'torch' and name in ('UntypedStorage',):
            return _StorageType(None)
        raise pickle.UnpicklingError(f'unsupported: {module}.{name}')

    def persistent_load(self, pid):
        # pid = ('storage', StorageType, key, device, numel)
        _, stype, key, _dev, numel = pid
        dtype = stype.dtype if isinstance(stype, _StorageType) else np.float32
        if dtype is None:
            dtype = np.float32  # assume fp32 for untyped
        entry = f'{self.prefix}data/{key}'
        return _StubStorage(dtype, numel, entry)


def load_pt(path):
    """Returns (numpy array, meta) — squeezes leading batch dim if present."""
    z = zipfile.ZipFile(path)
    name = [n for n in z.namelist() if n.endswith('/data.pkl')][0]
    prefix = name[:-len('data.pkl')]
    data = z.read(name)
    up = _PtUnpickler(io.BytesIO(data), path, prefix)
    obj = up.load()

    def fix(t):
        return t.to_numpy(path) if isinstance(t, _StubTensor) else t

    if isinstance(obj, _StubTensor):
        return fix(obj)
    if isinstance(obj, dict):
        return {k: fix(v) for k, v in obj.items()}
    return obj
