import os
from jinja2 import Environment, PackageLoader


class Writer:
    def __init__(self, path, width, height, depth=3,
                 database='Unknown', segmented=0):
        environment = Environment(
            loader=PackageLoader('pascal_voc_writer', 'templates'),
            keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    # object can be bounding box or polygon
    def addObject(self, name, box, pose='Unspecified',
                  truncated=0, difficult=0):
        # figure out if label is bounding box or polygon
        label_type = 'bndbox'



        self.template_parameters['objects'].append({
            'name': name,
            'type': label_type,
            'box': box,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)