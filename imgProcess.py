from diffusers.utils import load_image
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
import io

def process_request(data, pipe):   
    try:             
        # check if the post request has the file part
        if 'selectImage' not in data:
            return 'result', {'message': '', 'error': 'No file part in the request'}

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # pipe.to(device)
        
        prompt = "christmas decoration, animal, cute, high quality, 8k, 3d animation, masterpiece"
        negative_prompt = "EasyNegative, human, humanization, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

        inputImg = data.get('selectImage')
        image_data = BytesIO(inputImg)
        init_image = load_image(Image.open(image_data).convert("RGB"))
        image = np.array(init_image)

        low_threshold = 100
        high_threshold = 200
            
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        #canny_image.show()
        output = pipe(prompt, 
                negative_prompt=negative_prompt,
                image = init_image,
                control_image = canny_image,
                #num_inference_steps = 30,
                strength = 0.4,
                #controlnet_conditioning_scale = 0.3
                ).images[0]
        
        buffered = io.BytesIO()
        output.save(buffered, format='png')
        
        img_str = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        success = False
    
        # 이미지가 검정색인지 검사
        decoded_img = Image.open(io.BytesIO(base64.b64decode(img_str.split(',')[1])))
        is_black_image = all(pixel == (0, 0, 0) for pixel in decoded_img.getdata())
    
        if is_black_image:
            success = False
            return {'message': '', 'error': 'Sorry, an error occurred. Please try again.'}
        else:
            success = True

        if success:
            return {'message': 'Files successfully changed', 'img_str': img_str, 'error': ''}
        else:
            return {'message': '', 'img_str': '', 'error': 'errors'}
    
    except Exception as e:
        print(str(e))
        return {'message': '', 'error': 'Sorry, an error occurred. Please try again.'}