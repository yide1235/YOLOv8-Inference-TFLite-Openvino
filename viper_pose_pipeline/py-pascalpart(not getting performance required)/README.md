# Py-PascalPart

Py-PascalPart is a simple tool to read annotations files from PASCAL-Part
Dataset in Python, giving an easy access to 10103 images, 24971 object masks
and 181770 body parts.

## Repository and Datasets

Clone the repository and download the PASCAL-VOC 2010 and the PASCAL-Part
Datasets if you don't have them on your computer:

~~~~
git clone https://github.com/micco00x/py-pascalpart
mkdir datasets
cd datasets
wget http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
tar -zxvf trainval.tar.gz
tar -zxvf VOCtrainval_03-May-2010.tar
rm VOCtrainval_03-May-2010.tar
rm trainval.tar.gz
~~~~

Explore the dataset:

~~~~
python3 explore_dataset.py
~~~~

Explore the dataset specifying a different path for PASCAL-Part Dataset
annotation folder and PASCAL VOC 2010 JPEG images folder:

~~~~
python3 explore_dataset.py --annotation_folder=PATH_TO_ANNOTATIONS --images_folder=PATH_TO_IMAGES
~~~~

Note that `utils.load_annotations(path)` returns a dictionary containing all the information
stored in the `.mat` file specified by `path`. See `explore_dataset.py` for a typical
example of usage. More in detail, once the annotation file has been loaded:

* `annotations["objects"]` is a list of objects contained in the image
* `obj["mask"]` is the mask of the object
* `obj["class"]` is the class of the object
* `obj["parts"]` is a list of body parts
* `body_part["mask"]`is the mask of the body part
* `body_part["part_name"]` is the class of the body part

with `obj` element of `annotations["objects"]` and `body_part` element of
`obj["parts"]`.
