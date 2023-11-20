from flask import render_template
from flask_socketio import SocketIO, emit
from app import app
import torch
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler, AutoPipelineForImage2Image, DPMSolverMultistepScheduler, ControlNetModel
from diffusers.utils import load_image
import transformers
from concurrent.futures import ThreadPoolExecutor
from imgProcess import process_request

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

result_queue = {}
socketio = SocketIO(app)
executor = ThreadPoolExecutor(2)

@app.route('/')
def upload_form():
    return render_template('main.html')

@socketio.on('upload')
def upload_file(data):
    result = process_request(data, pipe)
    
    emit('result', result, json=True) 

if __name__ == "__main__":
    socketio.run(app, debug=False, host='0.0.0.0', port=8080)
