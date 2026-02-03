from importlib import resources

import jinja2


def get_num_add_ext_template() -> str:
    return (
        resources.files(__package__)
        .joinpath("num_add_ext.cpp.jinja2")
        .read_text(encoding="utf-8")
    )


def get_num_add_autograd_ext_template() -> str:
    return (
        resources.files(__package__)
        .joinpath("num_add_ext_autograd.cpp.jinja2")
        .read_text(encoding="utf-8")
    )


def get_templated_num_add_extension(namespace: str, num: int, autograd: bool) -> str:
    template_str = (
        get_num_add_autograd_ext_template() if autograd else get_num_add_ext_template()
    )
    template = jinja2.Template(template_str)
    return template.render(LIBRARY_NAME=namespace, NUM=num)
