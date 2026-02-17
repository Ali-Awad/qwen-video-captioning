import os
import json
import random
import time
from datetime import datetime
from collections import deque
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import logging
import cv2
import dashscope
from dashscope import MultiModalConversation

# Set up verbose logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_config():
    """Loads settings, prompts, and schema from config files."""
    try:
        with open('configs/settings.json', 'r') as f:
            settings = json.load(f)
        with open('configs/prompts.json', 'r') as f:
            prompts = json.load(f)
        with open('configs/schemas/video_response.schema.json', 'r') as f:
            schema = json.load(f)
        return settings, prompts, schema
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e.filename}")
        exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {e.doc.name}: {e.msg}")
        exit(1)

def get_video_files(input_dir, output_dir, max_items, shuffle):
    """Gets a list of video files to process."""
    if not os.path.exists(input_dir):
        logging.warning(f"Input directory not found: {input_dir}")
        return []

    all_videos = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    processed_videos = {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith('.json')}
    
    videos_to_process = [v for v in all_videos if os.path.splitext(v)[0] not in processed_videos]

    if shuffle:
        random.shuffle(videos_to_process)

    return videos_to_process[:max_items]

def calculate_cost(input_tokens, output_tokens, model):
    """Calculate cost based on token usage and model pricing."""
    
    # Qwen3-VL-Flash Tiered Pricing (USD per Million tokens)
    if "qwen3-vl-flash" in model.lower():
        # Tier determination based on INPUT tokens per request
        if input_tokens <= 32_000:
            input_price = 0.05
            output_price = 0.40
        elif input_tokens <= 128_000:
            input_price = 0.075
            output_price = 0.60
        else: # > 128k (up to 256k usually)
            input_price = 0.12
            output_price = 0.96
            
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        total_cost = input_cost + output_cost
        
        return {
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "pricing_model": "qwen3-vl-flash-tiered"
        }

    # Standard Pricing for other models (approximate USD/1M tokens)
    pricing = {
        "qwen-vl-max": {"input": 2.8, "output": 8.4},
        "qwen-vl-plus": {"input": 1.12, "output": 2.8},
        "qwen3-vl-plus": {"input": 1.12, "output": 2.8}, # Assumed similar to qwen-vl-plus
        "qwen-vl-turbo": {"input": 0.28, "output": 0.84}
    }
    
    model_key = "qwen-vl-plus"
    for key in pricing:
        if key in model.lower():
            model_key = key
            break
            
    input_cost_per_million = pricing[model_key]["input"]
    output_cost_per_million = pricing[model_key]["output"]
    
    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost
    
    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "pricing_model": model_key
    }

def process_video(video_path, prompts, schema, model, settings):
    """Processes a single video by uploading it and returning the caption with metadata."""
    try:
        logging.info(f"Processing video: {os.path.basename(video_path)}")
        
        # Get frame sampling FPS from settings
        vid_settings = settings.get('vid_caption', {})
        fps = vid_settings.get('frame_sampling_fps', 1.0) # Default to 1.0 if not set

        # Get video info locally first
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = f"{video_width}x{video_height}"
        video_length_seconds = total_frames / video_fps if video_fps > 0 else 0
        cap.release()

        # 1. Upload the video file directly to DashScope
        # For international users, we should check if we need to set the oss endpoint differently,
        # but typically the DashScope SDK handles this if the API Key is correct.
        # Note: 'dashscope-intl' endpoint might handle file uploads differently. 
        # But SDK usually abstracts this.
        
        # NOTE: DashScope SDK automatically handles file upload for local paths if using 'file://' 
        # in the messages, but to be safe and explicit we can use `dashscope.File.upload`?
        # Actually, the standard way for Qwen-VL in SDK is to use `file://` URI in the message.
        
        # IMPORTANT: For local files, we must use absolute path with file:// schema
        abs_video_path = os.path.abspath(video_path)
        file_uri = f"file://{abs_video_path}"
        
        # Prepare content
        # Inject schema into system prompt to ensure JSON compliance
        schema_str = json.dumps(schema, indent=2)
        system_instruction = f"{prompts['video']['system']}\n\nIMPORTANT: You must strictly follow this JSON schema:\n{schema_str}"
        
        # Qwen-VL MultiModalConversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"video": file_uri, "fps": fps},
                    {"text": f"{system_instruction}\n\n{prompts['video']['user']}"}
                ]
            }
        ]
        
        logging.info(f"Sending video request to {model} with FPS={fps}...")
        
        response = MultiModalConversation.call(
            model=model,
            messages=messages,
            result_format='message',
            response_format={"type": "json_object"}
        )
        
        if response.status_code != 200:
             logging.error(f"DashScope API Error: {response.code} - {response.message}")
             if response.status_code == 429:
                 raise Exception(f"Rate limit exceeded: {response.message}")
             raise Exception(f"API Error: {response.message}")

        try:
            result_content = response.output.choices[0].message.content
            if isinstance(result_content, list):
                text_result = ""
                for item in result_content:
                    if 'text' in item:
                        text_result += item['text']
                result = text_result
            else:
                 result = result_content
        except Exception as e:
            logging.error(f"Error parsing response content: {e}")
            return None, None

        # Extract usage
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        total_tokens = input_tokens + output_tokens
        
        cost_info = calculate_cost(input_tokens, output_tokens, model)
        
        logging.info(f"API Response for {os.path.basename(video_path)}:")
        logging.info(f"  - Input tokens: {input_tokens:,}")
        logging.info(f"  - Output tokens: {output_tokens:,}")
        logging.info(f"  - Estimated cost: ${cost_info['total_cost_usd']:.6f}")
        
        file_size_bytes = os.path.getsize(video_path)
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        video_length_formatted = f"{int(video_length_seconds // 60)}:{(int(video_length_seconds % 60)):02d}" if video_length_seconds > 0 else "0:00"
        analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        usage_metadata = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost_info['total_cost_usd'], 6)
        }
        
        file_metadata = {
            "filename": os.path.basename(video_path),
            "file_size_mb": file_size_mb,
            "analysis_timestamp": analysis_timestamp,
            "model_used": model,
            "frame_sampling_enabled": True, # Enabled via API parameter
            "frame_sampling_fps": fps,
            "sampling_method": "api_native_sampling",
            "resolution": resolution,
            "video_length_seconds": round(video_length_seconds, 2),
            "video_length_formatted": video_length_formatted
        }
        
        metadata = {
            "usage_metadata": usage_metadata,
            "file_metadata": file_metadata
        }
        
        return result, metadata
    except Exception as e:
        logging.error(f"Error processing video {os.path.basename(video_path)}: {e}")
        return None, None

def main():
    """Main function to run the video captioning process."""
    load_dotenv()
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logging.error("DASHSCOPE_API_KEY not found in .env file.")
        exit(1)
    
    dashscope.api_key = api_key
    
    # IMPORTANT: Set base_url for international users globally for DashScope SDK.
    # The default mainland China endpoint will reject international keys with InvalidApiKey.
    # We default to the DashScope international endpoint, but allow overriding via env.
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/api/v1"
    )
    if hasattr(dashscope, "base_http_api_url"):
        dashscope.base_http_api_url = base_url
        logging.info(f"Using DashScope base URL: {base_url}")
    
    settings, prompts, schema = load_config()
    
    vid_caption_settings = settings.get('vid_caption', {})
    common_settings = settings.get('common', {})
    safety_settings = settings.get('safety', {})
    
    input_dir = common_settings.get('input_root_dir', './input')
    base_output_dir = common_settings.get('output_dir', './output')
    
    model = vid_caption_settings.get('model', 'qwen-vl-plus')

    output_dir = os.path.join(base_output_dir, model)
    os.makedirs(output_dir, exist_ok=True)

    max_items = vid_caption_settings.get('max_items', 100)
    shuffle = vid_caption_settings.get('shuffle', True)
    concurrency = vid_caption_settings.get('concurrency', 1) # Reduce concurrency for full video uploads
    
    request_delay = safety_settings.get('request_delay_seconds', 0.5)
    rate_limits = safety_settings.get('rate_limits', {})
    
    model_limits = rate_limits.get(model, {})
    max_rpm = model_limits.get('rpm', safety_settings.get('max_rpm', 60))
    max_tpm = model_limits.get('tpm', 500000)

    video_files = get_video_files(input_dir, output_dir, max_items, shuffle)
    
    if not video_files:
        logging.info("No new videos to process.")
        return

    logging.info(f"Found {len(video_files)} new videos to process.")
    
    request_timestamps = deque()
    token_timestamps = deque()
    rpm_lock = Lock()
    tpm_lock = Lock()
    
    def wait_for_rate_limits(estimated_tokens=0):
        # ... (same implementation as before)
        wait_time_rpm = 0
        with rpm_lock:
            current_time = time.time()
            while request_timestamps and current_time - request_timestamps[0] > 60:
                request_timestamps.popleft()
            if len(request_timestamps) >= max_rpm:
                wait_time_rpm = max(0, request_timestamps[0] + 60 - current_time)
        
        wait_time = max(wait_time_rpm, 0)
        if wait_time > 0:
            logging.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        
        with rpm_lock:
            request_timestamps.append(time.time())

    def submit_with_rate_limit_check(video_path):
        # Estimate tokens loosely for video (e.g. 5000 tokens per video as a placeholder)
        estimated_tokens = 5000 
        wait_for_rate_limits(estimated_tokens)
        
        return process_video(video_path, prompts, schema, model, settings)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(submit_with_rate_limit_check, os.path.join(input_dir, vf)): vf for vf in video_files}

        for future in as_completed(futures):
            video_file = futures[future]
            video_name = os.path.splitext(video_file)[0]
            output_path = os.path.join(output_dir, f"{video_name}.json")
            
            try:
                result, metadata = future.result()
                if result:
                    # Clean markdown code blocks if present
                    if result.startswith("```json"):
                        result = result[7:-3]
                    elif result.startswith("```"):
                        result = result[3:-3]
                        
                    try:
                        # Attempt to find JSON object
                        start = result.find('{')
                        end = result.rfind('}')
                        if start != -1 and end != -1:
                            json_res = json.loads(result[start:end+1])
                            caption_data = json_res
                        else:
                             caption_data = {"caption": result, "error": "JSON parse failed"}
                    except:
                        caption_data = {"caption": result, "error": "JSON parse failed"}

                    output_data = {
                        **caption_data,
                        "usage_metadata": metadata.get("usage_metadata", {}),
                        "file_metadata": metadata.get("file_metadata", {})
                    }
                    
                    with open(output_path, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    logging.info(f"Successfully processed {video_file}")
                else:
                    logging.error(f"Failed to get result for {video_file}")
            except Exception as e:
                logging.error(f"Error in main loop for {video_file}: {e}")
            
            time.sleep(request_delay)

if __name__ == "__main__":
    main()
