import requests
import json
import base64
import io
import numpy as np
from PIL import Image


class OpenAILLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Compute model list safely here (no lambda, no callable)
        try:
            model_list = cls._get_model_list(
                cls._get_cached_models_endpoint(),
                cls._get_cached_token()
            )
        except Exception:
            model_list = ["gpt-4o", "gpt-4-vision-preview", "gpt-4", "gpt-3.5-turbo"]

        # Guarantee it's a list of strings
        if not isinstance(model_list, list) or not all(isinstance(m, str) for m in model_list):
            model_list = ["gpt-4o", "gpt-4-vision-preview", "gpt-4", "gpt-3.5-turbo"]

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "placeholder": "Enter your prompt here..."
                }),
                "endpoint": ("STRING", {
                    "multiline": False,
                    "default": "https://api.openai.com/v1/chat/completions",
                    "placeholder": "OpenAI-compatible endpoint URL"
                }),
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Your API token"
                }),
                "model": (model_list, {
                    "default": "gpt-4o"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "max_tokens": ("INT", {
                    "default": 150,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "image_detail": (["low", "high", "auto"], {
                    "default": "auto"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "LLM"

    # ------------------------------
    # 🧠 Helper: Image Encoding
    # ------------------------------
    def _encode_image_to_base64(self, image_tensor):
        """Convert ComfyUI image tensor to base64 encoded string"""
        try:
            # ComfyUI images are typically [batch, height, width, channels] with values 0–1
            if len(image_tensor.shape) == 4:
                image_array = image_tensor[0]
            else:
                image_array = image_tensor

            # Convert torch tensor → numpy
            if hasattr(image_array, 'cpu'):
                image_array = image_array.cpu().numpy()

            # Convert 0–1 float → 0–255 uint8
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)

            # Convert numpy → PIL
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                pil_image = Image.fromarray(image_array, 'RGB')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                pil_image = Image.fromarray(image_array, 'RGBA')
            else:
                pil_image = Image.fromarray(image_array)

            # Encode to base64 PNG
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            raise Exception(f"Failed to encode image: {str(e)}")

    # ------------------------------
    # 💬 Main Function
    # ------------------------------
    def generate_text(self, prompt, endpoint, api_token,
                      model="gpt-4o", max_tokens=150, temperature=0.7,
                      image=None, image_detail="auto"):
        try:
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }

            # Construct message
            if image is not None:
                image_data_url = self._encode_image_to_base64(image)
                message_content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url, "detail": image_detail}
                    }
                ]
            else:
                message_content = prompt

            data = {
                "model": model,
                "messages": [{"role": "user", "content": message_content}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(endpoint, headers=headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return (content,)
            else:
                return ("Error: No response content found",)

        except requests.exceptions.RequestException as e:
            return (f"Request Error: {str(e)}",)
        except json.JSONDecodeError as e:
            return (f"JSON Error: {str(e)}",)
        except Exception as e:
            return (f"Error: {str(e)}",)
