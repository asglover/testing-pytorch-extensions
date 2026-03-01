from torch.utils import cpp_extension

from num_add_lib.cpp_extension_templates.jinja_utils import (
    get_templated_num_add_extension,
)


def register_cpp_extension(namespace: str, number: int):
    extension_str = get_templated_num_add_extension(namespace, number)
    cpp_extension.load_inline(name=namespace, cpp_sources=extension_str)
