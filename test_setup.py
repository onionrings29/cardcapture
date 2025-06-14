import os
from dotenv import load_dotenv
import anthropic
from firebase_admin import credentials, initialize_app, firestore
import json

def test_claude_connection():
    """Test Claude API connection"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found in .env file")
            return False
            
        # Initialize client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Test API with a simple request
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": "Say 'Connection test successful' if you can read this."
            }]
        )
        
        print("Claude API Connection: SUCCESS")
        print(f"Response: {response.content[0].text}")
        return True
        
    except Exception as e:
        print(f"Claude API Connection: FAILED")
        print(f"Error: {str(e)}")
        return False

def test_firebase_connection():
    """Test Firebase connection"""
    try:
        # Check if credentials file exists
        if not os.path.exists("firebase-credentials.json"):
            print("Error: firebase-credentials.json not found")
            return False
            
        # Initialize Firebase
        cred = credentials.Certificate("firebase-credentials.json")
        initialize_app(cred)
        db = firestore.client()
        
        # Test connection with a simple write/read
        test_doc = db.collection('test').document('connection_test')
        test_doc.set({'timestamp': 'test'})
        test_doc.delete()  # Clean up
        
        print("Firebase Connection: SUCCESS")
        return True
        
    except Exception as e:
        print(f"Firebase Connection: FAILED")
        print(f"Error: {str(e)}")
        return False

def main():
    print("Testing API Connections...\n")
    
    # Test Claude
    claude_success = test_claude_connection()
    print()
    
    # Test Firebase
    firebase_success = test_firebase_connection()
    print()
    
    # Summary
    if claude_success and firebase_success:
        print("All connections successful! You can now run categorize.py")
    else:
        print("Some connections failed. Please check the errors above.")

if __name__ == "__main__":
    main() 