#!/usr/bin/env python
import sys
import argparse
import tensorflow as tf
import os.path
import json
from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

import numpy as np


IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

class MyEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

        return json.dumps(data, cls=MyEncoder)    

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", help="Path to the input image.\
                        Only jpeg images are supported.")
    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    parser.add_argument("-l", "--image_loader",
                        default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-t", "--input_type",
                        default=InputType.TENSOR.name.lower(),
                        help="input type",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    args = parser.parse_args()

    model = OpenNsfwModel()

    with tf.Session() as sess:

        input_type = InputType[args.input_type.upper()]
        model.build(weights_path=args.model_weights, input_type=input_type)

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if args.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(sess)
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        sess.run(tf.global_variables_initializer())

        image = fn_load_image(args.input_file)

        predictions = \
            sess.run(model.predictions,
                     feed_dict={model.input: image})

        print("Results for '{}'".format(args.input_file))
        print(predictions[0][0])
        print(predictions[0][1])
        print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))
        # SFW : Safe For Work , NSFW : Not Safe For Work 
        nude_json = {'SFW' : predictions[0][0] , 'NSFW' : predictions[0][1] }
        result= json.dumps(nude_json, cls=MyEncoder)
        loaded_json = json.loads(result)
        #for x in loaded_json:
        #  print("%s: %f" % (x, loaded_json[x]))
        print(loaded_json)
        f = open('data.txt', 'r+')
        f.truncate()
        with open('data.txt', 'w') as outfile:      
          json.dump(loaded_json, outfile) 

        return loaded_json  



if __name__ == "__main__":
    main(sys.argv)
