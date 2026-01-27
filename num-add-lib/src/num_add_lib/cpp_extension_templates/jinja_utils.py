from importlib import resources

import jinja2

_NUM_ADD_EXT_TEMPLATE_NAME = "num_add_ext.cpp.jinja2"


def get_num_add_ext_template() -> str:
    return resources.files(__package__).joinpath(_NUM_ADD_EXT_TEMPLATE_NAME).read_text(encoding="utf-8")


NUM_ADD_EXT_CPP_TEMPLATE = get_num_add_ext_template()

def get_templated_num_add_extension(namespace: str, num : int) -> str:
    template = jinja2.Template(NUM_ADD_EXT_CPP_TEMPLATE)
    return template.render(
        LIBRARY_NAME=namespace,
        NUM=num)
    
