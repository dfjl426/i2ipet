from flask import Flask, request, render_template, jsonify
from app import app
import time
import torch
from PIL import Image
from io import BytesIO
import io
from diffusers import EulerAncestralDiscreteScheduler, AutoPipelineForImage2Image, DPMSolverMultistepScheduler, ControlNetModel
from diffusers.utils import load_image
import base64
import cv2
import numpy as np
import transformers

token = app.config['TOKEN']

torch.cuda.empty_cache()
torch_dtype = torch.float16

# ControlNet 설정
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch_dtype)

# clip skip 설정
clip_skip = 2

if clip_skip > 1:
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        "models/1",
        subfolder = "text_encoder",
        num_hidden_layers = 12 - (clip_skip - 1),
        torch_dtype = torch_dtype
    )

# pipeline 설정(하나만 사용 가능)

## to do list
## 모델 확정되면 나머지 모델들은 지우기

model_path = "models/1"
#model_path = "stablediffusionapi/disney-pixar-cartoon"
#model_path = "rossiyareich/aniflatmixAnimeFlatColorStyle_v20-fp16"

if clip_skip > 1:
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_path,
        torch_dtype = torch_dtype,
        variant="fp16",
        use_auth_token=token,
        use_safetensors= True,
        text_encoder = text_encoder)
    print('skip')
else:
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_path,
        torch_dtype = torch_dtype,
        variant="fp16",
        use_auth_token=token,
        safety_checker = None,
        use_safetensors= True)

pipe = pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
#pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas='true', algorithm_type="sde-dpmsolver++")

# textual_inversion 설정
pipe.load_textual_inversion(
    "sayakpaul/EasyNegative-test", weight_name="EasyNegative.safetensors", token="EasyNegative")

# lora 설정(여러개 설정 가능)
# 여러개 로딩할 시 adpater_name 설정하고 각자 weight 설정
# 한개 로딩할 때도 관리가 용이해보여 그냥 adpater_name 사용하는 걸로~

## to do list
## 모델 확정되면 나머지 모델들은 지우기

## example code
## pipe.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
pipe.load_lora_weights("loRa", weight_name= "more_details.safetensors", adapter_name="more_detail")
pipe.load_lora_weights("loRa", weight_name= "blindbox_v1_mix.safetensors", adapter_name="blindbox")

pipe.set_adapters(["more_detail", "blindbox"], adapter_weights=[1, 0]) # lora weight 설정

# 최적화
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

@app.route('/')
def upload_form():
    return render_template('main.html')

@app.route('/img2img', methods=['POST'])
def upload_file():
    try:
        # check if the post request has the file part
        if 'selectImage' not in request.files:
            resp = jsonify({'message' : 'No file part in the request', 'error':''})
            resp.status_code = 400
            return resp

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # pipe.to(device)
        
        inputImg = request.files.getlist('selectImage')[0]
        
        prompt = "christmas decoration, animal, cute, high quality, 8k, 3d animation, masterpiece"
        negative_prompt = "EasyNegative, human, humanization, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

        init_image = load_image(Image.open(inputImg).convert("RGB"))

        #depth용
        #image = depth_estimator(init_image)['depth']
        #image = np.array(image)
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
        errors = {}
        success = False
        
        # 이미지가 검정색인지 검사
        decoded_img = Image.open(io.BytesIO(base64.b64decode(img_str.split(',')[1])))
        is_black_image = all(pixel == (0, 0, 0) for pixel in decoded_img.getdata())
        print(is_black_image)

        
        if is_black_image:
            success = False
            return jsonify({'message': '','error': 'Sorry, an error occurred. Please try again.'})
        else:
            success = True

        if success:
            return jsonify({'message': 'Files successfully changed', 'img_str': img_str, 'error': ''}), 201
        else:
            return jsonify({'message': '', 'img_str': img_str, 'error': 'errors'}), 400

    
    except Exception as e:
        print(str(e))
        return jsonify({'message': '','error': 'Sorry, an error occurred. Please try again.'}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
