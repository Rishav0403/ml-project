import os
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets.coco import load_coco_json
import json

from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from PIL import Image

class CustomDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(self, is_train)

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image_file = os.path.join(self.image_root, dataset_dict["file_name"])

        # Check image dimensions
        with Image.open(image_file) as img:
            width, height = img.size

        if width != dataset_dict["width"] or height != dataset_dict["height"]:
            print(f"Correcting dimensions for {dataset_dict['file_name']}")
            dataset_dict["width"] = width
            dataset_dict["height"] = height

            # Adjust annotations if necessary
            for ann in dataset_dict["annotations"]:
                bbox = ann["bbox"]
                bbox[0] = min(bbox[0], width)
                bbox[1] = min(bbox[1], height)
                bbox[2] = min(bbox[2], width)
                bbox[3] = min(bbox[3], height)
                ann["bbox"] = bbox

        # Proceed with the rest of the data loading
        image = utils.read_image(image_file, format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        if self.is_train:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            transforms = None

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

def load_coco_json_with_dimension_check(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    for img in coco_data['images']:
        img_file = os.path.join(image_root, img['file_name'])
        if os.path.exists(img_file):
            from PIL import Image
            with Image.open(img_file) as pil_img:
                actual_width, actual_height = pil_img.size
                if actual_width != img['width'] or actual_height != img['height']:
                    print(f"Correcting dimensions for {img['file_name']}")
                    img['width'], img['height'] = actual_width, actual_height

    return load_coco_json(json_file, image_root, dataset_name, extra_annotation_keys)



# Register the custom dataset
register_coco_instances("my_dataset_train", {}, "data_c/train/_annotations.coco.json", "data_c/train")
MetadataCatalog.get("my_dataset_train").set(thing_classes=["motorcycle-licence-plate", "licensePlate", "motorcycle", "withHelmet", "withoutHelmet"])

# Configuration
cfg = get_cfg()

# Get the configuration file
config_file = model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file(config_file)

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Number of classes in your dataset

# Create output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Define a custom trainer class
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

# Train the model
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Save the model
model = trainer.build_model(cfg)
model_weights_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
torch.save(model.state_dict(), model_weights_path)

# final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# DetectionCheckpointer(trainer.model.state_dict()).save("model_final")
# print(f"Model saved to {final_model_path}")
