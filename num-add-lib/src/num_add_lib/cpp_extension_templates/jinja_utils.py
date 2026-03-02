from importlib import resources

import jinja2


def _get_template_text(template_name: str) -> str:
    return (
        resources.files(__package__)
        .joinpath(template_name)
        .read_text(encoding="utf-8")
    )


def _render_template(template_name: str, namespace: str, num: int) -> str:
    template = jinja2.Template(_get_template_text(template_name))
    return template.render(LIBRARY_NAME=namespace, NUM=num)


def get_templated_num_add_forward_extension(namespace: str, num: int) -> str:
    return _render_template("num_add_ext.cpp.jinja2", namespace, num)


def get_templated_num_add_backward_x_extension(namespace: str, num: int) -> str:
    return _render_template("num_add_backward_x_ext.cpp.jinja2", namespace, num)
