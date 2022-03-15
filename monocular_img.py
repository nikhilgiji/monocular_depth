import argparse  
import tensorflow as tf 
import cv2 
import matplotlib.pyplot as plt 


def monocular_img(img_path): 

    """Function to predict for a single image
    """
    img = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB) / 255.0

    img_resized = tf.image.resize(img, [256,256], method='bicubic', preserve_aspect_ratio=False)
    #img_resized = tf.transpose(img_resized, [2, 0, 1])
    img_input = img_resized.numpy()
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_input = (img_input - mean) / std
    reshape_img = img_input.reshape(1,256,256,3)
    tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

    # load the intel midas model 
    model = "midas_model\lite-model_midas_v2_1_small_1_lite_1.tflite"
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    # inference
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output.reshape(256, 256)
                        
    # output file
    prediction = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    print(" Write image to: output_depth.png")
    depth_min = prediction.min()
    depth_max = prediction.max()
    img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
    cv2.imwrite("output_depth.png", img_out)
    plt.imshow(img_out)
    plt.show() 

    return img_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img') 
    args = parser.parse_args() 

    img_path = cv2.imread(args.img)   

    mono = monocular_img(img_path)