from importlib import resources

import jinja2


def get_num_add_ext_template() -> str:
    return (
        resources.files(__package__)
        .joinpath("num_add_ext.cpp.jinja2")
        .read_text(encoding="utf-8")
    )


def get_templated_num_add_extension(namespace: str, num: int) -> str:
    template = jinja2.Template(get_num_add_ext_template())
    return template.render(LIBRARY_NAME=namespace, NUM=num)
