# CardCapture

CardCapture is an intelligent document processing system that automatically categorizes and organizes business cards and receipts using OCR and AI. It leverages Firebase for storage and provides a robust backend for managing document metadata and images.

## Features

- 🔍 **OCR Processing**: Extracts text from images using OCR technology
- 🤖 **AI-Powered Categorization**: Uses Ollama LLM to intelligently categorize documents
- 📱 **Image Management**: Handles image processing, cropping, and storage
- 🔄 **Automated Workflow**: Processes documents from raw images to categorized data
- 📊 **Firebase Integration**: Stores data and images in Firebase Firestore and Storage
- 📝 **Detailed Metadata**: Captures and stores comprehensive document information

## Project Structure

```
cardcapture/
├── IMAGES/              # Image storage directory
│   ├── processed/      # Processed and renamed images
│   ├── cropped/        # Cropped document images
│   └── converted/      # Converted image formats
├── categorize.py       # Main processing script
├── cardcapture.py      # Core functionality
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup

1. **Prerequisites**
   - Python 3.8+
   - Firebase account and credentials
   - Ollama LLM running locally

2. **Installation**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/cardcapture.git
   cd cardcapture

   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configuration**
   - Place your Firebase credentials JSON file in the project root
   - Ensure Ollama is running locally
   - Configure Firebase Storage and Firestore rules

## Usage

1. Place your images in the `IMAGES` directory
2. Run the categorization script:
   ```bash
   python3 categorize.py
   ```

The script will:
- Process images using OCR
- Categorize documents using AI
- Upload images to Firebase Storage
- Store metadata in Firestore
- Move processed images to appropriate directories

## Data Structure

### Firestore Collections

1. **items**
   - Document ID: Generated UUID
   - Fields:
     - `filename`: Original image filename
     - `category`: Document category (receipt/business_card)
     - `confidence`: AI confidence score
     - `metadata`: Document-specific data
     - `image_ids`: Array of associated image document IDs
     - `created_at`: Timestamp
     - `updated_at`: Timestamp

2. **images**
   - Document ID: Generated UUID
   - Fields:
     - `item_id`: Reference to parent item
     - `type`: Image type (original/cropped)
     - `url`: Firebase Storage URL
     - `path`: Storage path
     - `size`: File size
     - `created_at`: Timestamp

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Firebase for backend services
- Ollama for AI capabilities
- Python community for excellent libraries 