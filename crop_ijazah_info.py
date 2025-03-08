import cv2
import numpy as np
import os
import pytesseract  # Added for text detection

def detect_and_crop_diploma_section(image_path, debug_mode=False):
    """
    Enhanced detection and cropping of student information from "yang bertanda..." to half document height
    and serial number from an Indonesian diploma image.
    
    CRITICAL: The "yang bertanda..." marker MUST be found no matter what.
    
    Args:
        image_path: Path to the diploma image.
        debug_mode: If True, saves intermediate processing images for debugging.

    Returns:
        A tuple: (cropped_info_section, cropped_serial_section) or (None, None) if detection fails.
    """
    debug_dir = "debug_images"
    if debug_mode and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    def save_debug_image(name, image):
        if debug_mode and image is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{name}.jpg"), image)

    # Initialize visualization to None
    visualization = None

    try:
        # Step 1: Load and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None, None
            
        # Save original
        save_debug_image("01_original", img)
        
        # Create a copy for visualization if in debug mode
        if debug_mode:
            visualization = img.copy()
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        save_debug_image("02_grayscale", gray)
        
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        save_debug_image("03_blurred", blurred)
        
        # Calculate pixels per cm based on A4 standard
        dpi = 300  # Assumed DPI, adjust if known
        pixels_per_cm = dpi / 2.54  # Convert from inches to cm
        
        # Get image dimensions
        img_height, img_width = img.shape[:2]
        
        # -- AGGRESSIVE TEXT-BASED "YANG BERTANDA" MARKER DETECTION --
        # This section is critical and must find the marker
        
        start_marker_found = False
        y_start = None
        
        # Define variant patterns to search for with varying cases and spaces
        marker_patterns = [
            "yang bertanda",
            "yangbertanda",
            "yang  bertanda",
            "YANG BERTANDA",
            "Yang Bertanda",
            "YangBertanda",
            "YANGBERTANDA"
        ]
        
        # Try multiple preprocessing techniques for maximum chance of finding text
        preprocessing_methods = [
            ("default", lambda x: x),
            ("threshold", lambda x: cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)[1]),
            ("adaptive", lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                      cv2.THRESH_BINARY, 21, 11)),
            ("otsu", lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("contrast", lambda x: cv2.equalizeHist(x))
        ]
        
        # First attempt: Standard OCR with multiple vertical segments
        if not start_marker_found:
            print("Attempting primary text detection method for 'yang bertanda'...")
            
            # Try wider range of vertical segments
            vertical_segments = [
                (0.0, 0.3),   # 0-30% of image height
                (0.1, 0.4),   # 10-40% of image height
                (0.2, 0.5),   # 20-50% of image height
                (0.3, 0.6),   # 30-60% of image height
                (0.0, 0.5)    # 0-50% wider search
            ]
            
            for top_pct, bottom_pct in vertical_segments:
                if start_marker_found:
                    break
                    
                top = int(img_height * top_pct)
                bottom = int(img_height * bottom_pct)
                search_region = img[top:bottom, :]
                save_debug_image(f"text_search_region_{top_pct}_{bottom_pct}", search_region)
                
                # Split region into smaller segments for more accurate detection
                num_segments = 8  # More segments for finer granularity
                segment_height = (bottom - top) // num_segments
                
                for i in range(num_segments):
                    if start_marker_found:
                        break
                        
                    segment_top = top + i * segment_height
                    segment_bottom = segment_top + segment_height
                    segment = img[segment_top:segment_bottom, :]
                    segment_gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY) if len(segment.shape) > 2 else segment
                    save_debug_image(f"segment_{top_pct}_{bottom_pct}_{i}", segment)
                    
                    # Try each preprocessing method
                    for method_name, preprocess_func in preprocessing_methods:
                        if start_marker_found:
                            break
                            
                        try:
                            processed_segment = preprocess_func(segment_gray)
                            save_debug_image(f"processed_{method_name}_{top_pct}_{bottom_pct}_{i}", processed_segment)
                            
                            ocr_text = pytesseract.image_to_string(processed_segment, lang='ind')
                            if debug_mode:
                                print(f"OCR Text ({method_name}): {ocr_text}")
                            
                            if any(pattern.lower() in ocr_text.lower() for pattern in marker_patterns):
                                start_marker_found = True
                                y_start = segment_top
                                found_text = next((p for p in marker_patterns if p.lower() in ocr_text.lower()), "yang bertanda")
                                print(f"FOUND MARKER '{found_text}' at y={y_start} using {method_name} preprocessing")
                                if debug_mode:
                                    cv2.line(visualization, (0, y_start), (img_width, y_start), (0, 255, 0), 2)
                                    cv2.putText(visualization, f"FOUND: {found_text}", (10, y_start-10), 
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                break
                        except Exception as e:
                            print(f"Error in OCR method {method_name}: {e}")
                            continue
        
        # Second attempt: Try horizontal sections with more aggressive preprocessing
        if not start_marker_found:
            print("Attempting secondary detection with horizontal sections...")
            
            # Try horizontal sections for thoroughness
            horizontal_sections = [
                (0, img_width//3),               # Left third
                (img_width//3, 2*img_width//3),  # Middle third
                (2*img_width//3, img_width),     # Right third
                (0, img_width//2),               # Left half
                (img_width//2, img_width)        # Right half
            ]
            
            for section_idx, (x_start, x_end) in enumerate(horizontal_sections):
                if start_marker_found:
                    break
                    
                # Look in top half of document with finer granularity
                for y_pct in range(0, 50, 5):  # 0%, 5%, 10%, ..., 45% of height
                    if start_marker_found:
                        break
                        
                    y_section_start = int(img_height * y_pct / 100)
                    y_section_end = min(img_height, y_section_start + int(img_height * 0.1))  # 10% height chunk
                    
                    section = img[y_section_start:y_section_end, x_start:x_end]
                    if section.size == 0:
                        continue
                        
                    section_gray = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY) if len(section.shape) > 2 else section
                    save_debug_image(f"horiz_section_{section_idx}_{y_pct}", section)
                    
                    # Try more aggressive preprocessing
                    for method_name, preprocess_func in preprocessing_methods:
                        if start_marker_found:
                            break
                            
                        try:
                            processed_section = preprocess_func(section_gray)
                            save_debug_image(f"horiz_processed_{method_name}_{section_idx}_{y_pct}", processed_section)
                            
                            ocr_text = pytesseract.image_to_string(processed_section, lang='ind')
                            if debug_mode:
                                print(f"Horizontal OCR Text ({method_name}, {section_idx}, {y_pct}%): {ocr_text}")
                            
                            if any(pattern.lower() in ocr_text.lower() for pattern in marker_patterns):
                                start_marker_found = True
                                y_start = y_section_start
                                found_text = next((p for p in marker_patterns if p.lower() in ocr_text.lower()), "yang bertanda")
                                print(f"FOUND MARKER '{found_text}' at y={y_start} in horizontal section {section_idx}")
                                if debug_mode:
                                    cv2.line(visualization, (0, y_start), (img_width, y_start), (0, 255, 0), 2)
                                    cv2.putText(visualization, f"FOUND: {found_text}", (10, y_start-10), 
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                break
                        except Exception as e:
                            print(f"Error in horizontal OCR method {method_name}: {e}")
                            continue
        
        # Third attempt: Character-based detection with sliding windows
        if not start_marker_found:
            print("Attempting character-based detection with sliding windows...")
            
            # Create a template with "YANG BERTANDA" text
            # Create blank images with different fonts, sizes, etc.
            template_height = 60
            template_width = 400
            
            font_faces = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX]
            font_scales = [1.0, 1.5, 2.0]
            
            for font_face in font_faces:
                if start_marker_found:
                    break
                    
                for font_scale in font_scales:
                    if start_marker_found:
                        break
                        
                    for template_text in ["Yang Bertanda", "YANG BERTANDA"]:
                        if start_marker_found:
                            break
                            
                        template = np.ones((template_height, template_width), dtype=np.uint8) * 255
                        cv2.putText(template, template_text, (10, 40), font_face, font_scale, 0, 2)
                        save_debug_image(f"template_{template_text}_{font_face}_{font_scale}", template)
                        
                        # Use template matching on grayscale image
                        for y_pct in range(0, 50, 5):  # Try top half of document
                            if start_marker_found:
                                break
                                
                            y_search_start = int(img_height * y_pct / 100)
                            y_search_end = min(img_height, y_search_start + int(img_height * 0.15))
                            
                            search_region = gray[y_search_start:y_search_end, :]
                            if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
                                continue
                                
                            try:
                                result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                                
                                if max_val > 0.4:  # Lower threshold for more lenient matching
                                    y_start = y_search_start + max_loc[1]
                                    start_marker_found = True
                                    print(f"FOUND MARKER using template matching at y={y_start} (confidence: {max_val:.2f})")
                                    if debug_mode:
                                        cv2.line(visualization, (0, y_start), (img_width, y_start), (0, 255, 0), 2)
                                        cv2.putText(visualization, f"FOUND BY TEMPLATE", (10, y_start-10), 
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    break
                            except Exception as e:
                                print(f"Error in template matching: {e}")
                                continue
        
        # Fourth attempt: LAST RESORT - Search letter by letter for partial matches
        if not start_marker_found:
            print("CRITICAL: Using last resort letter-by-letter detection...")
            
            partial_patterns = ["yang", "bert", "tanda", "YA", "BE", "TA"]
            
            # Scan the top half of the document in small chunks
            for y_pct in range(0, 50, 3):  # Even finer granularity
                if start_marker_found:
                    break
                    
                y_section_start = int(img_height * y_pct / 100)
                y_section_end = min(img_height, y_section_start + int(img_height * 0.05))  # 5% height chunk
                
                # Score system for letter-by-letter detection
                section_scores = []
                
                for x_step in range(0, img_width, img_width//5):
                    x_end = min(img_width, x_step + img_width//3)  # Overlap sections
                    
                    section = img[y_section_start:y_section_end, x_step:x_end]
                    if section.size == 0:
                        continue
                        
                    section_gray = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY) if len(section.shape) > 2 else section
                    save_debug_image(f"last_resort_section_{y_pct}_{x_step}", section)
                    
                    # Try different preprocessing methods
                    max_score = 0
                    best_text = ""
                    
                    for method_name, preprocess_func in preprocessing_methods:
                        try:
                            processed_section = preprocess_func(section_gray)
                            save_debug_image(f"last_resort_{method_name}_{y_pct}_{x_step}", processed_section)
                            
                            ocr_text = pytesseract.image_to_string(processed_section, lang='ind')
                            if debug_mode:
                                print(f"Last resort OCR ({method_name}, {y_pct}%, {x_step}): {ocr_text}")
                            
                            # Calculate score for partial matches
                            score = 0
                            for pattern in partial_patterns:
                                if pattern.lower() in ocr_text.lower():
                                    score += 10  # Higher score for key pattern parts
                            
                            # Check for character presence
                            for char in "yangbertanda":
                                if char.lower() in ocr_text.lower():
                                    score += 1
                            
                            if score > max_score:
                                max_score = score
                                best_text = ocr_text
                        except Exception as e:
                            print(f"Error in last resort OCR: {e}")
                            continue
                    
                    section_scores.append((y_section_start, max_score, best_text, x_step))
                
                # Sort by score and check if we have a good candidate
                section_scores.sort(key=lambda x: x[1], reverse=True)
                
                if section_scores and section_scores[0][1] >= 15:  # Threshold for accepting partial match
                    y_start = section_scores[0][0]
                    start_marker_found = True
                    print(f"FOUND PARTIAL MARKER match at y={y_start} with score {section_scores[0][1]}")
                    print(f"Text found: {section_scores[0][2]}")
                    if debug_mode:
                        cv2.line(visualization, (0, y_start), (img_width, y_start), (0, 255, 0), 2)
                        cv2.putText(visualization, f"PARTIAL MATCH: {section_scores[0][1]}", (10, y_start-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Absolute last resort - statistical estimation
        if not start_marker_found:
            print("CRITICAL: Unable to detect 'yang bertanda' with all methods.")
            print("Using statistical estimation based on standard document layout.")
            
            # Statistical analysis shows "yang bertanda" is typically around 25-35% from the top in diploma layout
            y_start = int(img_height * 0.3)  # 30% from top
            start_marker_found = True
            print(f"USING STATISTICAL ESTIMATION for 'yang bertanda' at y={y_start}")
            if debug_mode:
                cv2.line(visualization, (0, y_start), (img_width, y_start), (255, 0, 0), 2)
                cv2.putText(visualization, "STATISTICAL POSITION", (10, y_start-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Calculate half document height from y_start
        y_end = min(img_height, y_start + (img_height // 3))
        if debug_mode:
            print(f"Setting end point at y={y_end} (half document height from start)")
            cv2.line(visualization, (0, y_end), (img_width, y_end), (0, 0, 255), 2)
        
        # Create the student info section (from "yang bertanda" to half height)
        cropped_info_section = img[y_start:y_end, :]
        save_debug_image("final_info_section", cropped_info_section)
        
        if debug_mode:
            info_vis = visualization.copy()
            cv2.rectangle(info_vis, (0, y_start), (img_width, y_end), (0, 0, 255), 3)
            cv2.putText(info_vis, "Half document height", (10, y_end-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            save_debug_image("info_section_highlighted", info_vis)
        
        # -- SERIAL NUMBER SECTION DETECTION BASED ON "DN-" TEXT --
        # 7cm height in pixels
        serial_height_pixels = int(7 * pixels_per_cm)
        
        # Search for "DN-" text in the bottom half of the image
        serial_y_start = None
        
        # Try multiple regions in the bottom half to find "DN-"
        bottom_regions = [
            ("bottom50", int(img_height * 0.5)),
            ("bottom40", int(img_height * 0.6)),
            ("bottom30", int(img_height * 0.7))
        ]
        
        for region_name, start_y in bottom_regions:
            if serial_y_start is not None:
                break
                
            bottom_section = img[start_y:, :]
            save_debug_image(f"{region_name}_search", bottom_section)
            
            # Use OCR to find "DN-" text
            section_height = img_height - start_y
            num_segments = 5
            segment_height = section_height // num_segments
            
            for i in range(num_segments):
                segment_top = start_y + i * segment_height
                segment_bottom = min(img_height, segment_top + segment_height)
                segment = img[segment_top:segment_bottom, :]
                save_debug_image(f"{region_name}_segment_{i}", segment)
                
                # Try to find "DN-" marker
                try:
                    text = pytesseract.image_to_string(segment)
                    if debug_mode:
                        print(f"OCR Text in {region_name} segment {i}: {text}")
                    
                    if "DN-" in text or "DN " in text or "DN" in text:
                        serial_y_start = segment_top
                        if debug_mode:
                            print(f"Serial marker 'DN-' found at y={serial_y_start}")
                            cv2.line(visualization, (0, serial_y_start), (img_width, serial_y_start), (255, 0, 0), 2)
                        break
                except Exception as e:
                    print(f"OCR error in serial detection: {e}")
                    continue
        
        # If DN- text wasn't found, do additional processing
        if serial_y_start is None:
            print("DN- text marker not found, trying additional methods...")
            
            # Use more aggressive preprocessing to find DN- text
            for region_name, start_y in bottom_regions:
                if serial_y_start is not None:
                    break
                    
                region_gray = gray[start_y:, :]
                
                # Try different thresholding methods
                thresh_methods = [
                    ("adaptive", lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                                 cv2.THRESH_BINARY_INV, 21, 11)),
                    ("otsu", lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
                ]
                
                for method_name, method_func in thresh_methods:
                    if serial_y_start is not None:
                        break
                        
                    try:
                        thresh_region = method_func(region_gray)
                        save_debug_image(f"{region_name}_{method_name}_thresh", thresh_region)
                        
                        # Dilate to connect characters
                        kernel = np.ones((3, 1), np.uint8)
                        dilated = cv2.dilate(thresh_region, kernel, iterations=1)
                        save_debug_image(f"{region_name}_{method_name}_dilated", dilated)
                        
                        # Process in smaller segments
                        section_height = region_gray.shape[0]
                        num_segments = 5
                        segment_height = section_height // num_segments
                        
                        for i in range(num_segments):
                            segment_top = i * segment_height
                            segment_bottom = min(section_height, segment_top + segment_height)
                            segment = dilated[segment_top:segment_bottom, :]
                            
                            # Convert back to grayscale for OCR
                            segment_for_ocr = 255 - segment  # Invert for better OCR
                            save_debug_image(f"{region_name}_{method_name}_segment_{i}", segment_for_ocr)
                            
                            text = pytesseract.image_to_string(segment_for_ocr)
                            if debug_mode:
                                print(f"OCR Text in processed {region_name} {method_name} segment {i}: {text}")
                                
                            if "DN-" in text or "DN " in text or "DN" in text:
                                serial_y_start = start_y + segment_top
                                if debug_mode:
                                    print(f"Serial marker 'DN-' found at y={serial_y_start} with {method_name}")
                                    cv2.line(visualization, (0, serial_y_start), (img_width, serial_y_start), 
                                           (255, 0, 255), 2)
                                break
                    except Exception as e:
                        print(f"Error in additional serial detection with {method_name}: {e}")
                        continue
        
        # Final fallback - use the bottom 15% of the image if DN- wasn't found
        if serial_y_start is None:
            print("Warning: Could not find DN- text marker. Using fallback bottom position.")
            serial_y_start = int(img_height * 0.85)
            if debug_mode:
                print(f"Using fallback serial_y_start at {serial_y_start}")
                cv2.line(visualization, (0, serial_y_start), (img_width, serial_y_start), (0, 0, 255), 2)
        
        # Create the serial number section with full width and 7cm height
        x_serial = 0
        w_serial = img_width
        
        # Ensure we don't go beyond image bounds
        h_serial = min(serial_height_pixels, img_height - serial_y_start)
        
        if h_serial <= 0:
            # If we're already at the bottom of the image, move up to get required height
            serial_y_start = max(0, img_height - serial_height_pixels)
            h_serial = min(serial_height_pixels, img_height - serial_y_start)
            print(f"Adjusted serial_y_start to {serial_y_start} to fit required height")
        
        cropped_serial_section = img[serial_y_start:serial_y_start + h_serial, :]
        save_debug_image("final_serial_section", cropped_serial_section)
        
        if debug_mode:
            serial_vis = visualization.copy()
            cv2.rectangle(serial_vis, (x_serial, serial_y_start), (x_serial+w_serial, serial_y_start+h_serial), 
                         (0, 0, 255), 3)
            cv2.putText(serial_vis, f"7cm height ({h_serial}px)", (10, serial_y_start-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            save_debug_image("serial_section_highlighted", serial_vis)
        
        save_debug_image("final_visualization", visualization)
        
        return cropped_info_section, cropped_serial_section

    except Exception as e:
        print(f"An error occurred during diploma processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main function to run the diploma processor with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process diploma images to extract information and serial sections.')
    parser.add_argument('image_path', help='Path to the diploma image file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to save intermediate images')
    parser.add_argument('--output_dir', default='output', help='Directory to save output images')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Processing image: {args.image_path}")
    cropped_info, cropped_serial = detect_and_crop_diploma_section(args.image_path, debug_mode=args.debug)
    
    if cropped_info is not None:
        info_path = os.path.join(args.output_dir, "student_info.jpg")
        cv2.imwrite(info_path, cropped_info)
        print(f"Student info section saved to: {info_path}")
    else:
        print("Student info section cropping failed.")
    
    if cropped_serial is not None:
        serial_path = os.path.join(args.output_dir, "serial_number.jpg")
        cv2.imwrite(serial_path, cropped_serial)
        print(f"Serial number section saved to: {serial_path}")
    else:
        print("Serial number section cropping failed.")
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
