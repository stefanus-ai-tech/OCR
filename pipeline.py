import cv2
import os
from crop_ijazah_info import detect_and_crop_diploma_section
from isIjazah import is_ijazah
import pytesseract
import numpy as np
import json
from dotenv import load_dotenv
import base64
from groq import Groq
import time
import logging

load_dotenv()  # Load environment variables from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_ocr_from_groq(image, model_name, prompt, max_retries=3):
    """Gets OCR results from Groq using the specified model and prompt.

    Args:
        image: The input image (NumPy array).
        model_name: The name of the Groq model to use.
        prompt: The prompt to use for the OCR task.

    Returns:
        dict: The OCR result as a dictionary (parsed JSON), or None on error.
    """
    if not GROQ_API_KEY:
        logging.error("GROQ_API_KEY not found in .env file.")
        return None

    client = Groq(api_key=GROQ_API_KEY)

    # Encode the image as base64
    _, encoded_image = cv2.imencode(".jpg", image)
    image_bytes = encoded_image.tobytes()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None
            )
            
            response_text = completion.choices[0].message.content
            # Try to extract JSON from the response text
            try:
                # Look for JSON-like structure in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx + 1]
                    return json.loads(json_str)
                else:
                    # If no JSON structure found, create a simple JSON with the text
                    return {"raw_text": response_text.strip()}
            except json.JSONDecodeError as je:
                logging.warning(f"Failed to parse JSON from response: {je}")
                return {"raw_text": response_text.strip()}

        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            else:
                logging.error(f"All attempts failed for model {model_name}")
                return None

def get_ocr_from_groq_11b(image, prompt):
    """Gets OCR results from Groq llama-3.2-11b-vision-preview."""
    return get_ocr_from_groq(image, "llama-3.2-11b-vision-preview", prompt)

def get_ocr_from_groq_90b(image, prompt):
    """Gets OCR results from Groq llama-3.2-90b-vision-preview."""
    return get_ocr_from_groq(image, "llama-3.2-90b-vision-preview", prompt)

def compare_ocr_results(ocr_results):
    """
    Compare OCR results from different methods and select the best one.
    Uses a scoring system based on completeness and consistency.
    """
    required_fields = [
        "Nama", "Tempat dan tanggal lahir", "Nama orang tua/wali",
        "Nomor induk siswa", "Nomor induk siswa nasional",
        "Nomor peserta ujian nasional", "Sekolah asal"
    ]
    
    scores = {
        'pytesseract': 0,
        'groq_11b': 0,
        'groq_90b': 0
    }
    
    # Score based on completeness
    for method in scores.keys():
        if method == 'pytesseract':
            # Simple presence check for pytesseract
            text = ocr_results[method]['text']
            scores[method] += sum(1 for field in required_fields if field.lower() in text.lower())
        else:
            result = ocr_results[method]
            if isinstance(result, dict):
                scores[method] += sum(1 for field in required_fields if field in result)
                # Add bonus for structured data
                scores[method] += 2
    
    # Find the best method
    best_method = max(scores.items(), key=lambda x: x[1])[0]
    
    # Get the best serial number
    serial_11b = ocr_results.get('groq_11b_serial', {}).get('Serial', '')
    serial_90b = ocr_results.get('groq_90b_serial', {}).get('Serial', '')
    best_serial = serial_90b if len(serial_90b) > len(serial_11b) else serial_11b

    return {
        'comparison': {
            'method_scores': scores,
            'best_method': best_method,
        },
        'best_result': ocr_results[best_method] if best_method != 'pytesseract' else {'raw_text': ocr_results[best_method]['text']},
        'serial_number': best_serial
    }

def process_image(image_path):
    """Processes an image according to the flowchart.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: The result of the processing in JSON format.
    """
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        # Check if it's an Ijazah
        is_ijazah_result, confidence = is_ijazah(image_path)
        if not is_ijazah_result:
            return json.dumps({"status": "invalid", "reason": "Not an Ijazah"})

        # Preprocess: Context Selection
        cropped_info, cropped_serial = detect_and_crop_diploma_section(image_path)
        if cropped_info is None or cropped_serial is None:
            return json.dumps({"status": "invalid", "reason": "Failed to process image"})

        # OCR with 3 models
        ocr_results = {}

        # pytesseract (on info section)
        ocr_results['pytesseract'] = {
            "text": pytesseract.image_to_string(cropped_info, lang='ind'),
        }

        # Define prompts for Groq
        student_info_prompt = """Transcribe the image as exact as possible with JSON format

        {
          "Nama"
          "Tempat dan tanggal lahir"
          "Nama orang tua/wali"
          "Nomor induk siswa"
          "Nomor induk siswa nasional"
          "Nomor peserta ujian nasional"
          "Sekolah asal"
        }
        """

        serial_number_prompt = """Transcribe the serial as exact as possible with JSON format

        {
          "Serial"
        }
        """

        # Groq llama-3.2-11b-vision-preview (on info section)
        groq_11b_info = get_ocr_from_groq_11b(cropped_info, student_info_prompt)
        ocr_results['groq_11b'] = groq_11b_info

        # Groq llama-3.2-90b-vision-preview (on info section)
        groq_90b_info = get_ocr_from_groq_90b(cropped_info, student_info_prompt)
        ocr_results['groq_90b'] = groq_90b_info

        # Groq llama-3.2-11b-vision-preview (on serial section)
        groq_11b_serial = get_ocr_from_groq_11b(cropped_serial, serial_number_prompt)
        ocr_results['groq_11b_serial'] = groq_11b_serial

        # Groq llama-3.2-90b-vision-preview (on serial section)
        groq_90b_serial = get_ocr_from_groq_90b(cropped_serial, serial_number_prompt)
        ocr_results['groq_90b_serial'] = groq_90b_serial

        # Compare results and select the best one
        comparison_result = compare_ocr_results(ocr_results)
        
        return json.dumps({
            "status": "valid",
            "ocr_comparison": comparison_result['comparison'],
            "final_output": {
                "student_info": comparison_result['best_result'],
                "serial_number": comparison_result['serial_number']
            },
            "confidence": {
                "ijazah_detection": confidence,
                "has_valid_sections": bool(cropped_info is not None and cropped_serial is not None),
                "best_method_score": comparison_result['comparison']['method_scores'][comparison_result['comparison']['best_method']]
            },
            "all_results": ocr_results  # Keep all results for reference
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return json.dumps({
            "status": "error",
            "reason": str(e),
            "details": {
                "error_type": type(e).__name__,
                "error_location": "process_image"
            }
        })

def main():
    """Main function to run from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Process an image and determine if it is a valid Ijazah.')
    parser.add_argument('image_path', help='Path to the image file')
    args = parser.parse_args()

    result = process_image(args.image_path)
    # Pretty print the JSON result
    parsed_result = json.loads(result)
    print(json.dumps(parsed_result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
