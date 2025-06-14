import json
import os
import re
import asyncio
import sys
import logging
import hashlib
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

import aiohttp
import psutil
from firebase_admin import credentials, initialize_app, firestore, storage
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('categorize.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Firebase with connection confirmation
try:
    cred_path = Path(__file__).parent / "cardcapture-a4dd3-firebase-adminsdk-fbsvc-6dbc83fbab.json"
    if not cred_path.exists():
        raise FileNotFoundError(f"Firebase credentials not found at {cred_path}")
    
    cred = credentials.Certificate(str(cred_path))
    firebase_app = initialize_app(cred, {
        'storageBucket': 'cardcapture-a4dd3.firebasestorage.app'  # Updated bucket name
    })
    db = firestore.client()
    bucket = storage.bucket()
    
    # Test connection
    test_ref = db.collection('connection_test').document()
    test_ref.set({'timestamp': datetime.now()})
    test_ref.delete()
    logger.info("✅ Successfully connected to Firebase")
except Exception as e:
    logger.error(f"🔥 Firebase initialization failed: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Memory requirements for models
MODEL_MEMORY_REQUIREMENTS = {
    "deepseek-r1:14b": 10,  # GB with 20% buffer
}

STANDARD_CATEGORIES = {
    # Core identification
    'id': 'id',
    'type': 'type',
    'material': 'material',
    'color': 'color',
    
    # Physical characteristics
    'size': {
        'numerical': None,
        'descriptive': None,
        'unit': None,
        'notes': None
    },
    'weight': {
        'numerical': None,
        'unit': None,
        'notes': None
    },
    'condition': {
        'status': None,  # new, used, damaged, etc.
        'wear_level': None,  # minimal, moderate, heavy
        'damage_notes': None,
        'restoration_status': None  # original, restored, partially restored
    },
    
    # Provenance and History
    'provenance': {
        'acquisition_date': None,
        'acquisition_location': None,  # place where item was acquired
        'acquisition_method': None,  # purchased, gift, inherited, etc.
        'previous_owner': None,
        'historical_notes': None,
        'authenticity_notes': None
    },
    
    # Value and Market
    'value': {
        'purchase_price': None,
        'estimated_value': None,
        'currency': None,
        'valuation_date': None,
        'valuation_notes': None
    },
    
    # Maintenance and Care
    'maintenance': {
        'last_check_date': None,
        'cleaning_instructions': None,
        'storage_requirements': None,
        'special_care_notes': None
    },
    
    # Additional Details
    'description': None,
    'tags': None,  # array of relevant tags
    'notes': None,  # any additional notes
    'related_items': None  # array of related item IDs
}

class LLMProvider:
    """Base class for LLM providers with common functionality"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_tokens = 0
        self.total_cost = 0.0
        self._verify_memory()
    
    def _verify_memory(self):
        """Check if system has enough memory for the model"""
        if self.model_name in MODEL_MEMORY_REQUIREMENTS:
            required = MODEL_MEMORY_REQUIREMENTS[self.model_name]
            available = psutil.virtual_memory().available / (1024**3)
            if available < required:
                logger.warning(f"⚠️ Low memory: {available:.1f}GB < {required:.1f}GB required")

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError
    
    @classmethod
    def create_provider(cls, provider_type: str, model_name: str) -> 'LLMProvider':
        """Factory method for creating providers"""
        providers = {
            'ollama': OllamaProvider,
            'deepseek': DeepseekProvider
        }
        if provider_type not in providers:
            raise ValueError(f"Invalid provider: {provider_type}")
        return providers[provider_type](model_name)

class OllamaProvider(LLMProvider):
    """Local provider using Ollama API"""
    API_BASE = "http://localhost:11434/api"
    
    async def generate(self, prompt: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.API_BASE}/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": True
                    }
                ) as response:
                    response.raise_for_status()
                    full_response = ""
                    
                    # Collect response without progress bar
                    async for chunk in response.content:
                        if chunk:
                            try:
                                data = json.loads(chunk)
                                chunk_text = data.get("response", "")
                                full_response += chunk_text
                            except json.JSONDecodeError:
                                continue
                    
                    # Extract and print only the JSON part
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', full_response)
                    if json_match:
                        json_str = json_match.group(1)
                        # Clean up the JSON string
                        json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL)
                        json_str = re.sub(r'//.*?\n', '\n', json_str)
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        
                        try:
                            # Pretty print the JSON
                            json_obj = json.loads(json_str)
                            print("\n" + "="*50)
                            print("Final JSON to be uploaded:")
                            print("-"*50)
                            print(json.dumps(json_obj, indent=2))
                            print("="*50 + "\n")
                        except json.JSONDecodeError:
                            print("\n" + "="*50)
                            print("Warning: Could not parse JSON response")
                            print("="*50 + "\n")
                    
                    return full_response
                    
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

class DeepseekProvider(LLMProvider):
    """Cloud provider using Deepseek API"""
    API_BASE = "https://api.deepseek.com/v1"
    
    async def generate(self, prompt: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 2000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.API_BASE}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                    
        except Exception as e:
            logger.error(f"Deepseek API error: {e}")
            raise

class ItemProcessor:
    """Processes items from OCR data and categorizes them"""
    def __init__(self, ocr_path: str):
        self.ocr_path = Path(ocr_path)
        self.ocr_data = self._load_ocr_data()
        self.llm = self._init_llm()
        self.processed_filenames = self._get_processed_filenames()
    
    def _load_ocr_data(self) -> Dict[str, Any]:
        """Load and validate OCR data"""
        if not self.ocr_path.exists():
            raise FileNotFoundError(f"OCR file missing: {self.ocr_path}")
        
        logger.debug(f"Loading OCR data from: {self.ocr_path}")
        with open(self.ocr_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError("OCR data must be a dictionary")
        
        logger.debug(f"OCR data keys: {list(data.keys())[:5]}...")  # Show first 5 keys
        logger.info(f"📦 Loaded OCR data with {len(data)} items")
        return data
    
    def _init_llm(self) -> LLMProvider:
        """Initialize LLM provider (Ollama only)"""
        try:
            provider = LLMProvider.create_provider(
                provider_type='ollama',
                model_name='deepseek-r1:14b'
            )
            logger.info(f"🧠 Using Ollama provider with model: deepseek-r1:14b")
            return provider
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            sys.exit(1)
    
    def _get_processed_filenames(self) -> set:
        """Get set of filenames that have already been processed"""
        try:
            processed = set()
            # Query Firestore for all documents that have a filename field
            docs = db.collection('items').where('filename', '!=', None).stream()
            for doc in docs:
                data = doc.to_dict()
                if 'filename' in data:
                    processed.add(data['filename'])
            logger.info(f"📚 Found {len(processed)} previously processed items")
            return processed
        except Exception as e:
            logger.error(f"Failed to get processed filenames: {e}")
            return set()
    
    def _combine_text_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Combine related text blocks for better processing"""
        combined = []
        current_text = []
        current_confidence = 0.0
        current_position = None
        
        for block in blocks:
            if block['type'] != 'TEXT' or block['text'].strip() == '-':
                continue
                
            if current_position and self._is_near(current_position, block['position']):
                current_text.append(block['text'])
                current_confidence = max(current_confidence, block['confidence'])
                current_position = self._merge_positions(current_position, block['position'])
            else:
                if current_text:
                    combined.append({
                        'text': ' '.join(current_text),
                        'confidence': current_confidence,
                        'position': current_position
                    })
                current_text = [block['text']]
                current_confidence = block['confidence']
                current_position = block['position']
        
        if current_text:
            combined.append({
                'text': ' '.join(current_text),
                'confidence': current_confidence,
                'position': current_position
            })
            
        return combined
    
    def _is_near(self, pos1: Dict, pos2: Dict, threshold: int = 50) -> bool:
        """Check if two text blocks are spatially close"""
        y1 = (pos1['top_left'][1] + pos1['bottom_left'][1]) / 2
        y2 = (pos2['top_left'][1] + pos2['bottom_left'][1]) / 2
        return abs(y1 - y2) < threshold
    
    def _merge_positions(self, pos1: Dict, pos2: Dict) -> Dict:
        """Merge two block positions into one bounding box"""
        return {
            'top_left': [
                min(pos1['top_left'][0], pos2['top_left'][0]),
                min(pos1['top_left'][1], pos2['top_left'][1])
            ],
            'top_right': [
                max(pos1['top_right'][0], pos2['top_right'][0]),
                min(pos1['top_right'][1], pos2['top_right'][1])
            ],
            'bottom_right': [
                max(pos1['bottom_right'][0], pos2['bottom_right'][0]),
                max(pos1['bottom_right'][1], pos2['bottom_right'][1])
            ],
            'bottom_left': [
                min(pos1['bottom_left'][0], pos2['bottom_left'][0]),
                max(pos1['bottom_left'][1], pos2['bottom_left'][1])
            ]
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
    async def _categorize_item(self, filename: str, content: Dict) -> Dict:
        """Categorize item using LLM with retry logic"""
        try:
            blocks = self._combine_text_blocks(content['content']['blocks'])
            prompt = self._build_prompt(blocks)
            
            # Debug output for specific item
            if filename == '437D6BA7-D4D9-479C-AE14-4D1F93F64C00.heic':
                logger.debug(f"Prompt for {filename}:\n{prompt}")
            
            response = await self.llm.generate(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"Categorization failed for {filename}: {e}")
            return self._fallback_response()

    def _build_prompt(self, blocks: List[Dict]) -> str:
        """Construct LLM prompt from text blocks"""
        blocks_json = json.dumps(blocks, indent=2)
        return f"""
        You are an expert data extraction system. Extract information from the text blocks into a JSON structure.
        
        CRITICAL INSTRUCTIONS:
        1. DO NOT include any thinking process, explanations, or <think> tags
        2. DO NOT include any text outside the JSON markers
        3. Return ONLY the JSON output between ```json and ``` markers
        4. The JSON MUST have exactly two top-level keys: "standard_categories" and "custom_categories"
        5. If you're unsure about any field, use null instead of guessing
        
        Data Format Guidelines:
        1. ID: The number in the top right corner of the text blocks is usually the item's ID
        2. Dates:
           - If date appears as "OK then [date]", this is the last check date
           - If date appears alone, this is the date the item was acquired
        3. Item Types: Items are typically one of:
           - figurine (for small decorative items)
           - sculpture (for larger decorative items)
           - keychain (for items with chains/rings)
           - shoes (for actual footwear)
           - other (specify in description)
        4. Condition Status: Use one of:
           - new (unused, original condition)
           - like_new (minimal wear)
           - used (shows signs of use)
           - damaged (has visible damage)
           - restored (has been repaired/restored)
        5. Wear Level: Use one of:
           - minimal (almost no wear)
           - moderate (normal wear)
           - heavy (significant wear)
        6. Acquisition Method: Use one of:
           - purchased (bought from store/market)
           - gift (received as gift)
           - inherited (passed down)
           - found (discovered)
           - other (specify in notes)
        
        Required JSON structure (this exact format):
        ```json
        {{
            "standard_categories": {{
                "id": "number from top right corner",
                "type": "one of: figurine, sculpture, keychain, shoes, other",
                "material": "primary material",
                "color": "primary color or array of colors",
                "size": {{
                    "numerical": "numeric value if available",
                    "descriptive": "descriptive size if available",
                    "unit": "unit of measurement if available",
                    "notes": "any size-related notes"
                }},
                "weight": {{
                    "numerical": "weight value if available",
                    "unit": "unit of measurement if available",
                    "notes": "any weight-related notes"
                }},
                "condition": {{
                    "status": "one of: new, like_new, used, damaged, restored",
                    "wear_level": "one of: minimal, moderate, heavy",
                    "damage_notes": "description of any damage",
                    "restoration_status": "one of: original, restored, partially restored"
                }},
                "provenance": {{
                    "acquisition_date": "date when item was acquired",
                    "acquisition_location": "place where item was acquired",
                    "acquisition_method": "one of: purchased, gift, inherited, found, other",
                    "previous_owner": "name of previous owner if known",
                    "historical_notes": "any historical information",
                    "authenticity_notes": "notes about item's authenticity"
                }},
                "value": {{
                    "purchase_price": "original purchase price if known",
                    "estimated_value": "current estimated value. estimate using inflation and market trends",
                    "currency": "currency code (e.g., USD, EUR)",
                    "valuation_date": "date of last valuation",
                    "valuation_notes": "any notes about value"
                }},
                "maintenance": {{
                    "last_check_date": "date of last inspection (if format is 'OK then [date]')",
                    "cleaning_instructions": "how to clean the item",
                    "storage_requirements": "how to store the item",
                    "special_care_notes": "any special care instructions"
                }},
                "description": "detailed description of the item",
                "tags": ["array", "of", "relevant", "tags"],
                "notes": "any additional notes",
                "related_items": ["array", "of", "related", "item", "IDs"]
            }},
            "custom_categories": {{
                // Only include if there are unique features that don't fit in standard categories
                // For example: special markings, provenance, condition notes, etc.
                // This section is optional - only use if needed
            }}
        }}
        ```
        
        Rules:
        1. The response MUST have "standard_categories" and "custom_categories" as top-level keys
        2. ALL standard categories must be present under "standard_categories"
        3. 'type' field is REQUIRED and should be one of the specified item types
        4. Use the number from top right as the ID
        5. Interpret dates according to the format guidelines
        6. Use the specified values for condition.status, wear_level, and acquisition_method
        7. Put all basic information in standard_categories
        8. Only use custom_categories for truly unique features that don't fit in standard categories
        9. Return ONLY valid JSON between ```json and ``` markers
        10. DO NOT include any thinking process or explanations
        
        Input text blocks:
        {blocks_json}
        """

    def _parse_response(self, response: str) -> Dict:
        """Parse and validate LLM response"""
        try:
            logger.debug(f"Raw response length: {len(response)}")
            
            # Log the raw response for debugging
            logger.info("Raw LLM response:")
            logger.info("-" * 50)
            logger.info(response)
            logger.info("-" * 50)
            
            # First, try to find JSON between ```json and ``` markers
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if not json_match:
                logger.error("No JSON markers found in response")
                logger.error("Expected JSON between ```json and ``` markers")
                return self._fallback_response()
            
            json_str = json_match.group(1)
            
            # Clean up the JSON string
            json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL)  # Remove think tags
            json_str = re.sub(r'//.*?\n', '\n', json_str)  # Remove comments
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            
            # Log the cleaned JSON string for debugging
            logger.debug("Cleaned JSON string:")
            logger.debug("-" * 50)
            logger.debug(json_str)
            logger.debug("-" * 50)
            
            # Try to parse the JSON
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                logger.error("Invalid JSON structure:")
                logger.error("-" * 50)
                logger.error(json_str)
                logger.error("-" * 50)
                return self._fallback_response()
            
            # Validate required structure
            if not isinstance(result, dict) or 'standard_categories' not in result:
                logger.error("Response missing required structure")
                logger.error("Expected dict with 'standard_categories' key")
                logger.error("Actual structure:")
                logger.error("-" * 50)
                logger.error(json.dumps(result, indent=2))
                logger.error("-" * 50)
                return self._fallback_response()
            
            # Ensure all standard categories exist
            for key in STANDARD_CATEGORIES:
                if key not in result['standard_categories']:
                    result['standard_categories'][key] = None
                elif key == 'id' and result['standard_categories'][key] is not None:
                    result['standard_categories'][key] = str(result['standard_categories'][key])
            
            # Ensure type exists and is valid
            if not result['standard_categories'].get('type'):
                logger.warning("No type found, setting to unknown")
                result['standard_categories']['type'] = "unknown"
            
            # Add confidence score
            result['confidence'] = 0.8 if result['standard_categories'].get('type') != "unknown" else 0.5
            
            return result
            
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            logger.error("Full raw response:")
            logger.error("-" * 50)
            logger.error(response)
            logger.error("-" * 50)
            return self._fallback_response()

    def _fallback_response(self) -> Dict:
        """Fallback when categorization fails"""
        return {
            'standard_categories': {k: None for k in STANDARD_CATEGORIES},
            'custom_categories': {},
            'confidence': 0.0,
            'error': 'Processing failed'
        }
    
    async def _upload_image(self, image_path: str, item_id: str) -> Dict:
        """Upload image to Firebase Storage and return image metadata"""
        try:
            # Get the file name and extension
            original_path = Path(image_path)
            if not original_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
                
            file_ext = original_path.suffix
            
            # Create processed directory if it doesn't exist
            processed_dir = original_path.parent / 'processed'
            processed_dir.mkdir(exist_ok=True)
            
            # Create new filename with item ID
            new_filename = f"{item_id}{file_ext}"
            new_path = processed_dir / new_filename
            
            # Copy the file to processed directory with new name
            import shutil
            shutil.copy2(original_path, new_path)
            logger.info(f"📁 Copied {original_path.name} to processed/{new_filename}")
            
            # Create a unique path in storage using the new filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            storage_path = f"images/processed/{timestamp}_{new_filename}"
            logger.info(f"Storage path will be: {storage_path}")
            
            # Create a blob and upload the file
            blob = bucket.blob(storage_path)
            blob.upload_from_filename(str(new_path))
            
            # Make the file publicly accessible
            blob.make_public()
            
            # Get the public URL
            url = blob.public_url
            
            # Create an image document in the images collection
            image_data = {
                'item_id': item_id,
                'url': url,
                'original_filename': original_path.name,
                'processed_filename': new_filename,
                'storage_path': storage_path,
                'uploaded_at': datetime.now().isoformat(),
                'is_cropped': 'cropped_' in str(image_path),
                'local_path': str(new_path)  # Store the local path for reference
            }
            
            # Use a simpler document ID format for better readability
            doc_id = f"{item_id}_{timestamp}"
            logger.info(f"Creating image document with ID: {doc_id}")
            
            # Create the document in the images collection
            doc_ref = db.collection('images').document(doc_id)
            doc_ref.set(image_data)
            logger.info(f"Created image document in Firestore: {doc_id}")
            
            # Delete the original file after successful upload
            try:
                original_path.unlink()
                logger.info(f"🗑️ Deleted original file: {original_path.name}")
            except Exception as e:
                logger.error(f"Failed to delete original file {original_path.name}: {e}")
            
            # Return both the URL and the document reference
            return {
                'url': url,
                'doc_id': doc_id,
                'is_cropped': 'cropped_' in str(image_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to upload image {image_path}: {e}")
            logger.exception("Full error details:")  # This will log the full stack trace
            return None

    async def _upload_to_firestore(self, filename: str, data: Dict, image_path: str):
        """Upload processed data to Firestore"""
        try:
            # Ensure ID is a string and clean it
            doc_id = str(data['standard_categories'].get('id') or hashlib.md5(filename.encode()).hexdigest())
            doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)[:100]
            
            # Add metadata
            data['filename'] = filename
            data['processed_at'] = datetime.now().isoformat()
            
            # Ensure confidence is a float
            if 'confidence' in data:
                data['confidence'] = float(data['confidence'])
            
            # Get the base images directory
            images_dir = Path(self.ocr_path).parent / 'IMAGES'
            
            # Upload images if they exist
            image_refs = []  # List of image document references
            processed_images = []  # Track which images were processed
            
            # Try cropped image first
            if image_path and Path(image_path).exists():
                image_meta = await self._upload_image(image_path, doc_id)
                if image_meta:
                    image_refs.append({
                        'doc_id': image_meta['doc_id'],
                        'is_cropped': image_meta['is_cropped']
                    })
                    processed_images.append(image_path)
            
            # Try original image if cropped doesn't exist or as additional image
            original_path = images_dir / filename
            if original_path.exists() and str(original_path) not in processed_images:
                image_meta = await self._upload_image(str(original_path), doc_id)
                if image_meta:
                    image_refs.append({
                        'doc_id': image_meta['doc_id'],
                        'is_cropped': image_meta['is_cropped']
                    })
                    processed_images.append(str(original_path))
            
            # Add image references to the item data
            if image_refs:
                data['images'] = image_refs  # List of image document references
                data['primary_image'] = image_refs[0]['doc_id']  # Reference to primary image
                data['processed_images'] = processed_images  # Track which images were processed
            
            # Save to Firestore
            db.collection('items').document(doc_id).set(data)
            logger.info(f"Successfully uploaded {filename} to Firestore")
            return True
        except Exception as e:
            logger.error(f"Firestore upload failed for {filename}: {e}")
            return False

    async def process_item(self, filename: str):
        """Process single item end-to-end"""
        content = self.ocr_data[filename]
        image_path = content.get('cropped')
        
        try:
            logger.info(f"Processing item: {filename}")
            # Categorize item
            result = await self._categorize_item(filename, content)
            
            # Only proceed with upload if we got a valid result
            if result and result.get('standard_categories'):
                logger.info(f"Got valid categorization for {filename}")
                # Add a small delay before upload to ensure model output is complete
                await asyncio.sleep(0.1)
                
                # Upload to Firebase
                success = await self._upload_to_firestore(filename, result, image_path)
                
                if result.get('confidence', 0) < 0.7:
                    logger.warning(f"⚠️ Low confidence ({result['confidence']:.2f}) for {filename}")
                
                return success
            else:
                logger.error(f"Invalid categorization result for {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Item processing failed: {filename} - {e}")
            return False

    async def process_all(self):
        """Process only items that haven't been processed before"""
        filenames = list(self.ocr_data.keys())
        logger.info(f"🚀 Found {len(filenames)} items in OCR data")
        
        # Filter out already processed items
        items_to_process = [
            filename for filename in filenames 
            if filename not in self.processed_filenames
        ]
        
        if not items_to_process:
            logger.info("✨ All items have already been processed")
            return
            
        logger.info(f"🔄 Processing {len(items_to_process)} new items...")
        success_count = 0
        
        try:
            # Process in batches of 5
            for i in range(0, len(items_to_process), 5):
                batch = items_to_process[i:i+5]
                
                # Process batch sequentially to avoid output interference
                for filename in batch:
                    try:
                        # Process single item
                        result = await self.process_item(filename)
                        if result:
                            success_count += 1
                            # Add to processed set after successful upload
                            self.processed_filenames.add(filename)
                        
                        # Small delay between items to prevent output overlap
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"Failed to process {filename}: {e}")
                
                # Clean memory between batches
                gc.collect()
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
        
        logger.info(f"✅ Completed: {success_count}/{len(items_to_process)} new items processed")

async def main():
    # Use the correct path relative to the script location
    script_dir = Path(__file__).parent
    ocr_path = script_dir / "ocr_results.json"
    processor = ItemProcessor(str(ocr_path))
    await processor.process_all()

if __name__ == "__main__":
    asyncio.run(main())