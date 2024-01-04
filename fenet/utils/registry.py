import inspect

import six

# borrow from mmdetection


def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)

# 注册类
# 将被传入的注册好的类名从config/。。。.py文件中将对应字典取出，映射成一个类
# 放进Registry对象的_module_dict属性中
# 每次其他调用的时候@XXXX._module_dict就会把里面的内容放入对应的字典中
class Registry(object):
    # 传进来的对象名
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
        # 定义一个字典

    def __repr__(self):
    # 返回一个可以用来表示对象的可打印字符串
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    # 因为没有定义他的setter方法，所以是只读属性，不能通过self.name = newname修改
    def name(self):
        return self._name

    @property
    # 同上，self._module_dict也是只读
    def module_dict(self):
        return self._module_dict

    def get(self, key):
    # 普通方法，获取self._module_dict字典中特定key的value
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        # 关键方法，作用是注册module
        # model文件夹下的py文件中，里面的class定义上都会出现 @xxx.register_module
        # 就是将类当做形参，传入方法register_module（）中执行
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        # 判断是否为类
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        # 判断该类是否已经注册在 self._module_dict 里面
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    # 对上面的方法改名字，添加返回值，返回类本身
    def register_module(self, cls):
        self._register_module(cls)
        return cls

# 根据输入的要求创建不同的模块
def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None       #args相当于temp中间变量，是个字典
    args = cfg.copy()                       
    obj_type = args.pop('type')         #字典pop的作用，移除序列中key为‘type’的元素，并返回元素值，以此获取元素名
    if is_str(obj_type):
        obj_cls = registry.get(obj_type)        #获取obj_type里面的值
        if obj_cls is None:                     #如果没有注册就不执行
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type                      #将已经注册的元素加入到registed_module
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():            # items（）返回字典的key值对用于遍历
            args.setdefault(name, value)                    #将默认args的key值加入到args中，将模型和训练配置整合，然后送入类
    return obj_cls(**args)

    # 就是将config/xx.py文件里面的model里的字典除了‘type’的全部都当做形参传入字典
    # 送入名为‘type’的类里面