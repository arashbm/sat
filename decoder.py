import json
import torch


class TorchTensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {
                '__tensor__': True,
                'data': obj.tolist(),
                'dtype': str(obj.dtype),
                'shape': obj.shape
            }
        return super().default(obj)


class TorchTensorDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
                self, object_hook=self.object_hook,
                *args, **kwargs)

    def object_hook(self, dct):
        if '__tensor__' in dct:
            data = dct['data']
            _, dtype = dct['dtype'].split('.')
            dtype = getattr(torch, dtype)
            shape = tuple(dct['shape'])
            return torch.tensor(data, dtype=dtype).view(shape)
        return dct
