import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import re
from difflib import SequenceMatcher

def is_ijazah(image_path, threshold=0.65):
    """
    Detects if an image is an Indonesian Ijazah (diploma) document with improved accuracy.
    
    Args:
        image_path (str): Path to the image file
        threshold (float): Confidence threshold (0.0-1.0)
    
    Returns:
        bool: True if the image is detected as an Ijazah, False otherwise
        float: Confidence score
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return False, 0.0
        
        # Image preprocessing to improve text detection
        # Resize for consistent processing
        height, width = img.shape[:2]
        max_dimension = 1800
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # Apply adaptive thresholding to improve text visibility
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        img_thresh = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        img_erode = cv2.erode(img_thresh, kernel, iterations=1)
        img_dilate = cv2.dilate(img_erode, kernel, iterations=1)
        
        # Extract text using multiple approaches for redundancy
        pil_img_original = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_img_processed = Image.fromarray(img_dilate)
        
        text_original = pytesseract.image_to_string(pil_img_original, lang='ind', config='--psm 1').upper()
        text_processed = pytesseract.image_to_string(pil_img_processed, lang='ind', config='--psm 1').upper()
        
        # Combine texts for better coverage
        text = text_original + " " + text_processed
        
        # Key identifiers for Ijazah documents with common variations and typos
        ijazah_identifiers = [
            ["IJAZAH", "IJASA", "IJASAH", "LJAZAH"],
            ["KEMENTERIAN PENDIDIKAN", "KEMENTRIAN PENDIDIKAN", "KEMENDIKBUD"],
            ["REPUBLIK INDONESIA", "REPUBLIK INDONESI", "REP. INDONESIA"],
            ["SEKOLAH MENENGAH", "SMA", "SMK", "MADRASAH", "PERGURUAN TINGGI", "UNIVERSITAS"],
            ["LULUS", "TELAH LULUS", "DINYATAKAN LULUS"],
            ["NOMOR", "NO.", "NO :", "NOMOR:"],
            ["DN-", "DN/", "D.N"],
            ["TAHUN PELAJARAN", "THN PELAJARAN", "TAHUN AKADEMIK"],
            ["NAMA", "NAMA PESERTA"],
            ["TEMPAT TANGGAL LAHIR", "TTL", "LAHIR DI"]
        ]
        
        # Score matches using fuzzy matching for better accuracy
        match_scores = []
        
        for identifier_group in ijazah_identifiers:
            best_match_score = 0
            for variant in identifier_group:
                # Look for exact matches
                if variant in text:
                    best_match_score = 1.0
                    break
                    
                # Look for fuzzy matches
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    best_match_score = 0.9
                    break
                
                # Use sequence matcher for approximate matching
                for word in text.split():
                    if len(word) > 3:  # Ignore very short words
                        similarity = SequenceMatcher(None, variant, word).ratio()
                        best_match_score = max(best_match_score, similarity)
            
            match_scores.append(best_match_score)
        
        # Calculate text match score
        text_match_score = sum(match_scores) / len(ijazah_identifiers)
        
        # Visual feature analysis
        visual_score = 0.0
        
        # 1. Check for decorative borders
        edges = cv2.Canny(img_gray, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate the contour to simplify
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 points) of substantial size
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                # Check if it might be a border (covers significant area of the image)
                if w > img.shape[1] * 0.7 and h > img.shape[0] * 0.7:
                    visual_score += 0.4
                    break
        
        # 2. Check for typical layout pattern of ijazah (header, body, signature areas)
        # Divide image into 3 horizontal sections
        h_sections = np.array_split(img, 3, axis=0)
        
        # Check for header text density (typically higher in header)
        header = h_sections[0]
        header_gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
        header_thresh = cv2.threshold(header_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        header_text_density = np.sum(header_thresh == 0) / (header.shape[0] * header.shape[1])
        
        if header_text_density > 0.1:
            visual_score += 0.2
        
        # 3. Look for signature area in bottom section
        bottom = h_sections[2]
        bottom_gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        bottom_thresh = cv2.threshold(bottom_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        signature_contours, _ = cv2.findContours(bottom_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        signature_found = False
        
        for contour in signature_contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            extent = cv2.contourArea(contour) / (w * h)
            
            # Signatures typically have these characteristics
            if 0.9 < aspect_ratio < 8.0 and extent < 0.6 and w > 30 and h > 10:
                signature_found = True
                break
        
        if signature_found:
            visual_score += 0.2
        
        # 4. Check for national emblem (Garuda Pancasila)
        # Using color-based detection (simplified approach)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Yellow-gold color range (for Garuda emblem)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        if np.sum(yellow_mask) > 100:
            visual_score += 0.2
        
        # Calculate final confidence score (weighted combination)
        confidence_score = 0.7 * text_match_score + 0.3 * visual_score
        
        # Decision based on threshold
        return confidence_score >= threshold, confidence_score
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return False, 0.0

def check_multiple_images(directory, threshold=0.65):
    """
    Processes all images in a directory and reports which ones are Ijazah documents.
    
    Args:
        directory (str): Path to directory containing images
        threshold (float): Confidence threshold (0.0-1.0)
    """
    results = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            file_path = os.path.join(directory, filename)
            is_ijazah_result, confidence = is_ijazah(file_path, threshold)
            results.append((filename, is_ijazah_result, confidence))
            print(f"{filename}: {'Ijazah detected' if is_ijazah_result else 'Not an Ijazah'} (Confidence: {confidence:.2f})")
    
    return results

def optimize_threshold(directory, known_ijazah=None, known_non_ijazah=None):
    """
    Finds optimal threshold using a small validation set.
    
    Args:
        directory (str): Path to directory containing images
        known_ijazah (list): List of filenames known to be ijazah
        known_non_ijazah (list): List of filenames known not to be ijazah
        
    Returns:
        float: Optimal threshold value
    """
    if not known_ijazah or not known_non_ijazah:
        print("Need labeled examples to optimize threshold")
        return 0.65  # Default threshold
    
    # Get confidence scores for known examples
    scores = []
    
    for filename in known_ijazah:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            _, confidence = is_ijazah(file_path)
            scores.append((confidence, True))
    
    for filename in known_non_ijazah:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            _, confidence = is_ijazah(file_path)
            scores.append((confidence, False))
    
    # Try different thresholds to find optimal value
    best_threshold = 0.5
    best_accuracy = 0.0
    
    for threshold in np.arange(0.4, 0.9, 0.05):
        correct = 0
        for score, is_actual_ijazah in scores:
            prediction = score >= threshold
            if prediction == is_actual_ijazah:
                correct += 1
        
        accuracy = correct / len(scores)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.2f}")
    return best_threshold

# Example usage
if __name__ == "__main__":
    # Check a single image
    # result, confidence = is_ijazah("path/to/image.jpg")
    # print(f"Detection result: {'Ijazah detected' if result else 'Not an Ijazah'} (Confidence: {confidence:.2f})")
    
    # Optimize threshold with a small set of labeled examples (only need 5-10 of each)
    # known_ijazah = ["ijazah1.jpg", "ijazah2.jpg", "ijazah3.jpg"]
    # known_non_ijazah = ["document1.jpg", "document2.jpg", "document3.jpg"]
    # optimal_threshold = optimize_threshold("path/to/images", known_ijazah, known_non_ijazah)
    
    # Check a directory of images with optimal threshold
    # check_multiple_images("path/to/images_directory", threshold=optimal_threshold)
    
    # For testing with sample images
    sample_image = "Flowchart.png"
    if os.path.exists(sample_image):
        result, confidence = is_ijazah(sample_image)
        print(f"Detection result: {'Ijazah detected' if result else 'Not an Ijazah'} (Confidence: {confidence:.2f})")
    else:
        print("Please provide a valid image path to test the detector")