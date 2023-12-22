from utils import Registry, build_from_cfg
from torch import nn

NET = Registry('net')  # Registry 是一个注册表类，用于存储不同模块的构建函数。


def build(cfg, registry, default_args=None):
    # 如果 cfg 是列表，说明需要构建一个由多个模块组成的序列（例如神经网络的层次结构），则会遍历列表中的每个配置，
    # 并调用 build_from_cfg 函数构建模块。最后，使用 nn.Sequential 将这些模块组合成一个序列，并返回
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    # 如果 cfg 不是列表，说明只需要构建单个模块，直接调用 build_from_cfg 函数并返回
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_net(cfg):
    # 通过调用 build 函数，使用 NET 注册表中存储的构建函数来构建神经网络
    return build(cfg.net, NET, default_args=dict(cfg=cfg))
