from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from Pillow import Image

class QickDataset(object):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.obj = json.load(f)

    def __getitem__(self, idx):
        # load the image as a PIL Image
        
        img_dict = self.obj['images'][idx]
        image = Image.open(img_dict['file_name'])
        
        w, h = img_dict['width'], img_dict['height']
        img_id = img_dict['id']

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        anns = [ann for ann in obj['annotations'] if ann['image_id'] == img_id]
        
        boxes = [ann['bbox'] for ann in anns]
        boxes = [[x[1], x[0], x[3], x[2]] for x in boxes]
                
        labels = torch.tensor([ann['category_id'] for ann in anns])
        
        masks = [ann['segmentation'] for ann in anns]
       
        for i, obj in enumerate(masks):
            for j, poly in enumerate(obj):
                for k in range(0, len(poly), 2):
                    masks[i][j][k], masks[i][j][k+1] = poly[k+1], poly[k] 

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        boxlist.add_field("masks", SegmentationMask(masks, (w,h), mode='poly'))

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        img_dict = self.obj['images'][idx]
        return {"height": img_dict['height'], "width": img_dict['width']}

