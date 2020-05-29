"""
Set of utilities to automate processes for jupyter notebooks. Specifically,
tools that parse over a notebook (upon being rendered), reads in file as the
JSON it is saved as (i.e., format not seen when viewing notebook in most IDE and
browsers), and search it for all output cells, for which image outputs are then
searched for within outputs. In the end, all images generated as output when
notebook was executed are then saved off together in a directory specified when
instantiating interface.

Author: J. Robinson
"""
import base64
import json
import os

from pathlib import Path


def export_images(filepath_notebook: Path, directory_output: Path):
    """
    Wrapper function that parses notebook, and saves the images found therein.
    """
    image_collector = NotebookImageCollector(filepath_notebook,
                                             directory_output)

    image_collector.save_images()


def has_image_key(data_output: Path) -> bool:
    """
        Check whether an image key 'image' is in json block passed in;
    """
    return (len([key for key in list(data_output.keys()) if "/" in key
                 and key.split("/")[0] == "image"]) == 1)


def get_image_key(data_output):
    """
    Return its indices; otherwise, return empty list.
    """
    return [
        key
        for key in list(data_output.keys())
        if "/" in key and key.split("/")[0] == "image"
    ][0]


def read_jupyter_as_json(filepath: Path) -> Path:
    """
    Read in rendered notebook-- read in the JSON representation that is 'under
    the hood'
    :param filepath:    path to jupyter notebook.
    """
    with open(filepath, "r") as fout:
        contents = fout.read()

    return json.loads(contents)


class NotebookImageCollector:
    """
    Interface to handle saving images created upon executing a demo notebook.
    """

    def __init__(self, path_nb: Path, path_output: Path) -> (Path, Path):
        """
        Handles the book-keeping of i/o process for jupyter to directory.
        """

        self.path_notebook = path_nb
        self.directory_output = path_output

        self.filename_notebook = path_nb.name
        self.directory_notebook = (
            os.sep.join(str(path_nb).split(os.sep)[:-1])
            if os.sep in str(path_nb)
            else "."
        )
        self.dict_notebook = read_jupyter_as_json(path_nb)

    def get_image_cell_paths(self) -> dict:
        """
        grep instances of "outputs" from jupyter notebook.

        For each, check whether or not an image key "image" is a part of output
        Parse out, if so; drop, otherwise.
        """
        image_paths = {}

        for cell_idx, cell in enumerate(self.dict_notebook["cells"]):
            if "outputs" not in cell:
                continue
            indexed_data_outputs = [
                (output_idx, cell_output["data"])
                for output_idx, cell_output in enumerate(cell["outputs"])
                if "data" in cell_output
            ]
            cell_image_keys = [
                (indexed_data_output[0], get_image_key(indexed_data_output[1]))
                for indexed_data_output in indexed_data_outputs
                if has_image_key(indexed_data_output[1])
            ]
            if len(cell_image_keys) == 0:
                image_paths[cell_idx] = cell_image_keys

        return image_paths

    def get_image_from_cell_paths(self, image_paths) -> list:
        """
        Iterate over cells containing an image instance, and parse it out.
        """
        images = []
        for k, value in image_paths.items():
            for indexed_image_key in value:
                # lut for image data, along with corresponding metadata
                dict_image = {
                    "cell_idx"  : k,
                    "output_idx": indexed_image_key[0],
                    "content"   : self.dict_notebook["cells"][k]["outputs"][
                        indexed_image_key[0]
                    ]["data"][indexed_image_key[1]],
                    "format"    : indexed_image_key[1].split("/")[-1],
                }
                images.append(dict_image)

        return images

    def get_images(self) -> list:
        """
        Get images and store as list.
        """
        image_paths = self.get_image_cell_paths()
        images = self.get_image_from_cell_paths(image_paths)

        return images

    def save_images(self):
        """
        Save batch of images collected from notebook (i.e., final data dump).
        """
        path_out = (
            Path(self.directory_output)
            if self.directory_output
            else Path(self.directory_notebook).joinpath(
                self.filename_notebook.replace(".ipynb", "") + "_images"
            )
        )

        path_out.mkdir(exist_ok=True)

        images = self.get_images()

        for image in images:
            img_data = base64.b64decode(image["content"])
            filename = "{}_cell_{}_output_{}.{}".format(
                self.filename_notebook.replace(".ipynb", ""),
                image["cell_idx"],
                image["output_idx"],
                image["format"],
            )
            filepath = path_out / filename
            with open(filepath, "wb") as fout:
                fout.write(img_data)
