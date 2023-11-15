import base64
from io import BytesIO
import io
from app import app
from flask import Flask, request, render_template, jsonify, redirect, url_for
import torch
from PIL import Image # Pillow라고 불리우는 이미지처리 라이브러리
from diffusers import EulerAncestralDiscreteScheduler, AutoPipelineForImage2Image
from diffusers.utils import load_image
import requests

token = "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"

torch.cuda.empty_cache()

#pipe = AutoPipelineForImage2Image.from_pretrained("stablediffusionapi/disney-pixar-cartoon", torch_dtype=torch.float16, variant="fp16", use_auth_token=token)
#pipe = AutoPipelineForImage2Image.from_pretrained("rossiyareich/aniflatmixAnimeFlatColorStyle_v20-fp16", torch_dtype=torch.float16, variant="fp16", use_auth_token=token)

# pipe = AutoPipelineForImage2Image.from_pretrained(
#    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)


pipe = AutoPipelineForImage2Image.from_pretrained("danbrown/RevAnimated-v1-2-2", torch_dtype=torch.float16, variant="fp16", use_auth_token=token)

pipe = pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

@app.route('/')
def upload_form():
	return render_template('main.html')

@app.route('/img2img', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'selectImage' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    inputImg = request.files.getlist('selectImage')[0]

    # prompt = request.files.getlist('prompt')
    prompt = "animal, cute, high quality, disney, pixar style, 8k, 3d animation"
    negative_prompt = "bad art, amateur, lowres, bad anatomy, disfigured face, cropped, worst quality, low quality, normal quality, extra digits, extra legs, fewer digits, fewer legs, ugly, watermark, text, error, nsfw, deformed, disfigured,oversaturated,grain, low-res, blurry, low resolution, cropped, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, dehydrated bad proportions, extra limbs, cloned face, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    init_image = load_image(Image.open(inputImg).convert("RGB"))
    init_image = Image.open(inputImg).convert("RGB")
    
    image = pipe(prompt, 
            negative_prompt=negative_prompt,
            image=init_image,
            # num_inference_steps = 20,
            strength = 0.4,
			).images[0]


    buffered = io.BytesIO()
    image.save(buffered, format='png')
    img_str = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    errors = {}
    success = True

    if success and errors:
        errors['message'] = 'File(s) successfully changed'
        print(206)
        return resp
    if success:
        resp = jsonify({'message': 'Files successfully changed', 'img_str': img_str})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)