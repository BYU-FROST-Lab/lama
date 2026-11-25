import inspect
import logging
import torch
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule


def get_training_model_class(kind):
    if kind == 'default' or kind == 'actual_mask':
        return DefaultInpaintingTrainingModule

    raise ValueError(f'Unknown trainer module {kind}')


def make_training_model(config):
    kind = config.training_model.kind
    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'

    logging.info(f'Make training model {kind}')

    cls = get_training_model_class(kind)
    return cls(config, **kwargs)


def load_checkpoint(train_config, path, map_location='cuda', strict=True):
    model: torch.nn.Module = make_training_model(train_config)
    # The checksummed, trusted legacy Lightning checkpoint includes non-tensor metadata.
    # Torch 2.6 defaults weights_only=True, so opt out explicitly. Torch before 1.13 has no such
    # argument (and passing it raises), so keep the legacy call there.
    load_kwargs = {}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_kwargs['weights_only'] = False
    state = torch.load(path, map_location=map_location, **load_kwargs)
    model.load_state_dict(state['state_dict'], strict=strict)
    model.on_load_checkpoint(state)
    return model
