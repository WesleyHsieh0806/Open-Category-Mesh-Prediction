import objaverse
import multiprocessing
import os 
import json

from argparse import ArgumentParser
from tqdm import tqdm

def get_image_url(ann):
    '''
    * Output: the thumbnail image url of this object
        Notice that there are many images for each object, we only download the one with highest resolution
    '''
    highest_res_image = max(ann["thumbnails"]["images"], key= lambda x: x["size"])
    return highest_res_image["url"]

def object_to_save_path(uid, ann, args, url, file_type):
    '''
    * Output: the output path of this object
        the image and the object will be saved to this path
    '''
    cat = ann["lvis-categories"].replace("(", "").replace(")", "")
    file_name = file_type + "." + url.split(".")[-1]
    return os.path.join(args.root, "data", cat, uid, file_name)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, help="the data root of Objaverse")
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = parse_args()

    # create root for objaverse
    if not os.path.isdir(args.root):
        os.makedirs(args.root)

    # get all object uids 
    uids = objaverse.load_uids()

    # load annotations
    annotations = objaverse.load_annotations()  # uid: ann_dict

    print("{:=^40}".format("Attributes in one annotation"))
    for i, attribute in enumerate(annotations[uids[0]]):
        print(i, attribute)
    print()

    # Put LVIS categories into the annotations
    lvis_annotations = objaverse.load_lvis_annotations()
    for cat_name, uids in lvis_annotations.items():
        for uid in uids:
            if uid not in annotations: continue
            
            # this is 1 to 1 mapping
            annotations[uid]["lvis-categories"] = cat_name

    # Filter annotations: we only need Objaverse-LVIS
    annotations = {key: val for key, val in annotations.items() if "lvis-categories" in val}
    print("There are {} LVIS categories".format(len(lvis_annotations)))
    print("There are {} objects with LVIS categories".format(len(annotations)))


    # save the annotation to specified data root
    print("Save annotation to {} ...".format(os.path.join(args.root, "annotation.json")))
    with open(os.path.join(args.root, "annotation.json"), 'w') as f:
        json.dump(annotations, f)
    print("Done!")

    '''
    * Download images for each object
    '''
    print("Downloading {} images...".format(len(annotations)))
    failed_uid = []
    for uid, ann in tqdm(annotations.items()):
        url = get_image_url(ann)
        save_path = object_to_save_path(uid, ann, args, url, "image")
        
        # create repository
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        # wget download
        cmd = "wget -q {} -O \"{}\"".format(url, save_path)

        if os.system(cmd):
            # the download fails
            print("Unable to download {}".format(url))
            failed_uid.append(uid)

    print("Done!")

    # record all the uids of objects without images
    print("Fail to download images of {} objects. Save their uids to {}...".format(len(failed_uid), 
            os.path.join(args.root, "uid_wo_image.json")))
    with open(os.path.join(args.root, "uid_wo_image.json"), 'w') as f:
        json.dump(failed_uid, f)
    print("Done!")
    processes = multiprocessing.cpu_count()

    '''
    * Download the 3D Model (.glb)
    '''
    # objects will be a dict mapping uid to their locations
    print("Downloading {} 3D objects...".format(len(annotations)))
    selected_obj_ids = list(annotations.keys())
    objects = objaverse.load_objects(
        uids=selected_obj_ids,
        download_processes=processes
    )
    print("Done!")

    print("Move cached downloaded 3d object to desired output dir...")
    for uid, tmp_path in tqdm(objects.items()):
        # move cached .glb to specifed output dir
        save_path = object_to_save_path(uid, annotations[uid], args, tmp_path, "3Dobject")
        cmd = "mv {} {}".format(tmp_path, save_path)
        if os.system(cmd):
            break
    print("Done!")
