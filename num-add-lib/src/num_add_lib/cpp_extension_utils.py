from torch.utils import cpp_extension

from num_add_lib.cpp_extension_templates.jinja_utils import (
    get_templated_num_add_backward_x_extension,
    get_templated_num_add_forward_extension,
)


def register_cpp_forward_extension(namespace: str, number: int):
    extension_str = get_templated_num_add_forward_extension(namespace, number)
    cpp_extension.load_inline(
        name=f"{namespace}_forward",
        cpp_sources=extension_str,
    )


def register_cpp_backward_extension(namespace: str, number: int):
    extension_str = get_templated_num_add_backward_x_extension(namespace, number)
    cpp_extension.load_inline(
        name=f"{namespace}_backward_x",
        cpp_sources=extension_str,
    )
