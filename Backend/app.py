from flask import Flask, request, jsonify, make_response, send_from_directory
from flask_cors import CORS
import base64
import os
import uuid
import tempfile
import requests
from datetime import datetime, timedelta
from functools import wraps
from dotenv import load_dotenv
from pymongo import MongoClient, WriteConcern
from bson.objectid import ObjectId  # Add missing ObjectId import
import bcrypt
import sys
import json
from PIL import Image
import io
import numpy as np
import cv2  # Add OpenCV import for image processing

# Use PyJWT for JWT token handling
import jwt

# Import the road damage detection model
import sys
import os
# Make predict directory visible to Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import our prediction module
from predict.predict import predict, get_detector, RoadDamageDetector

# Initialize the road damage detector
road_damage_detector = get_detector()
print("Road damage detector initialized successfully")

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Disable the automatic CORS handling from flask-cors to avoid conflicts
# We'll handle CORS manually for better control

# Add explicit CORS handling for all requests
@app.after_request
def after_request(response):
    # Remove any existing CORS headers to avoid duplicates
    if 'Access-Control-Allow-Origin' in response.headers:
        del response.headers['Access-Control-Allow-Origin']
    
    # Add the correct CORS headers
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Add a specific route to handle OPTIONS requests for all endpoints
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    # Just return empty response with CORS headers
    # The after_request handler will add the CORS headers
    return '', 200

# Set up JWT secret key
app.config['SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-here')

# Initialize MongoDB connection
try:
    # Get MongoDB connection string from environment variable
    mongo_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
    print(f"Connecting to MongoDB with URI: {mongo_uri}")
    
    # Connect with a timeout to ensure we can reach the server
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    
    # Force a connection to verify it works
    client.admin.command('ping')
    print("MongoDB server ping successful!")
    
    # Use the correct database name from MongoDB Atlas
    db_name = 'SafeStreet'
    db = client[db_name]
    
    # Create collections with explicit validation
    if 'users' not in db.list_collection_names():
        print(f"Creating users collection in {db_name}")
        db.create_collection('users')
    
    if 'images' not in db.list_collection_names():
        print(f"Creating images collection in {db_name}")
        db.create_collection('images')
    
    # Get references to collections
    users_collection = db['users']
    images_collection = db['images']
    
    # Create indexes for better performance
    users_collection.create_index('email', unique=True)
    images_collection.create_index('user_id')
    
    print(f"Successfully connected to MongoDB database: {db_name}!")
    print(f"Available collections: {db.list_collection_names()}")
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    sys.exit(1)

# Initialize the AI model
# Get the road damage detector singleton
road_damage_detector = get_detector()
print("Road damage detector initialized successfully")

# Authentication decorators
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
            else:
                # Handle Basic auth or other formats
                token = auth_header
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            # Decode the JWT token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            
            # Get the user from the database
            current_user = users_collection.find_one({'_id': data['user_id']})
            
            if not current_user:
                return jsonify({'message': 'User not found!'}), 401
            
        except Exception as e:
            print(f"Token validation error: {str(e)}")
            return jsonify({'message': 'Token is invalid!'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(current_user, *args, **kwargs):
        if not current_user or current_user.get('role') != 'admin':
            return jsonify({'message': 'Admin privileges required!'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

# Add an endpoint to save images with location details
@app.route('/api/images/save', methods=['POST'])
@token_required
def save_image(current_user):
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get the image data from the request
        image_data = data['image']
        
        # If image_data is a URL, download it
        if image_data.startswith('http'):
            response = requests.get(image_data)
            if response.status_code != 200:
                return jsonify({'error': 'Failed to download image from URL'}), 400
            image_data = base64.b64encode(response.content).decode('utf-8')
            # Add data URL prefix if not present
            if not image_data.startswith('data:image'):
                image_data = f'data:image/jpeg;base64,{image_data}'
        
        # If the image data is a base64 string without the data URL prefix, add it
        if not image_data.startswith('data:image') and not image_data.startswith('http'):
            image_data = f'data:image/jpeg;base64,{image_data}'
        
        # Extract the base64 data from the data URL
        if image_data.startswith('data:image'):
            # Extract the base64 part after the comma
            image_data = image_data.split(',')[1]
        
        # Check if we have a cached image with this ID
        image_id = None
        is_cached = False
        
        # If we have an image_id in the request, check if it's in the cache
        if hasattr(app, 'temp_images') and 'image_id' in data and data['image_id'] in app.temp_images:
            image_id = data['image_id']
            cached_data = app.temp_images[image_id]
            is_cached = True
            print(f"Found cached image with ID: {image_id}")
        else:
            # Generate a new ID
            image_id = str(uuid.uuid4())
            print(f"Generated new image ID: {image_id}")
        
        # Prepare the document to save
        image_doc = {
            '_id': image_id,
            'image_data': image_data,
            'user_id': str(current_user['_id']),
            'uploaded_at': datetime.utcnow(),
            'status': 'pending'
        }
        
        # If we have a cached image, use its classification data
        if is_cached:
            image_doc['is_road'] = cached_data['is_road']
            image_doc['probability'] = cached_data['probability']
        else:
            # For new images, we need to classify them
            # Decode the base64 data
            image_bytes = base64.b64decode(image_data)
            
            # Create a temporary file to save the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            
            try:
                # Initialize the detector if not already done
                if not hasattr(app, 'road_damage_detector'):
                    app.road_damage_detector = RoadDamageDetector()
                
                # Classify the image
                result = app.road_damage_detector.classify_image(temp_path)
                
                # Check if classification was successful
                if not result.get('success', False):
                    raise Exception(result.get('error', 'Unknown classification error'))
                    
                # Extract the classification results
                is_road = result.get('is_road', False)
                probability = result.get('probability', 0.0)
                
                # Add classification results to the document
                image_doc['is_road'] = is_road
                image_doc['probability'] = float(probability)
                
                # Remove the temporary file
                os.unlink(temp_path)
                
                # If it's not a road, return an error
                if not is_road:
                    return jsonify({
                        'success': False,
                        'message': 'Image was not classified as a road',
                        'is_road': False,
                        'probability': float(probability)
                    }), 400
                
            except Exception as e:
                # Remove the temporary file in case of error
                os.unlink(temp_path)
                print(f"Error classifying image: {str(e)}")
                return jsonify({'error': f'Error classifying image: {str(e)}'}), 500
        
        # Add address information if provided
        print(f"Data received: {json.dumps(data, default=str)}")
        
        if 'name' in data:
            image_doc['name'] = data['name']
            print(f"Added name to image_doc: {data['name']}")
        else:
            print("No name field found in request data")
        
        if 'address' in data:
            image_doc['address'] = {
                'area': data['address'].get('area', ''),
                'pincode': data['address'].get('pincode', '')
            }
            print(f"Added address to image_doc: {json.dumps(image_doc['address'], default=str)}")
        else:
            print("No address field found in request data")
        
        # Save the image to the database
        try:
            print(f"Attempting to save document to MongoDB with keys: {list(image_doc.keys())}")
            
            # Ensure we have a valid _id
            if '_id' not in image_doc or not image_doc['_id']:
                image_doc['_id'] = str(uuid.uuid4())
                print(f"Generated new _id: {image_doc['_id']}")
            
            # Ensure we're using the right collection
            print(f"Using database: {db.name}, collection: {images_collection.name}")
            print(f"Collection count before insert: {images_collection.count_documents({})}")
            
            # Insert the document
            result = images_collection.insert_one(image_doc)
            print(f"Document saved successfully with ID: {result.inserted_id}")
            
            # Verify the document was saved by retrieving it
            saved_doc = images_collection.find_one({'_id': result.inserted_id})
            if saved_doc:
                print(f"Document retrieved from database with keys: {list(saved_doc.keys())}")
                if 'address' in saved_doc:
                    print(f"Address in saved document: {json.dumps(saved_doc['address'], default=str)}")
                else:
                    print("WARNING: Address not found in saved document!")
                    
                # Double check collection count after insert
                print(f"Collection count after insert: {images_collection.count_documents({})}")
            else:
                print("WARNING: Could not retrieve saved document from database!")
                # Try to find where it might have gone
                for db_name in client.list_database_names():
                    db_temp = client[db_name]
                    for coll_name in db_temp.list_collection_names():
                        count = db_temp[coll_name].count_documents({})
                        if count > 0:
                            print(f"Found {count} documents in {db_name}.{coll_name}")
        except Exception as e:
            print(f"ERROR saving document to MongoDB: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
        
        # If this was a cached image, remove it from the cache
        if is_cached:
            app.temp_images.pop(image_id, None)
            print(f"Removed cached image with ID: {image_id}")
        
        # Return the saved image data
        image_doc['_id'] = str(image_doc['_id'])  # Convert ObjectId to string
        
        return jsonify({
            'success': True,
            'message': 'Image saved successfully',
            'image': image_doc
        })
    
    except Exception as e:
        print(f"Error in save_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
@token_required
def classify_image(current_user):
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get the image data from the request
        image_data = data['image']
        
        # If image_data is a URL, download it
        if image_data.startswith('http'):
            response = requests.get(image_data)
            if response.status_code != 200:
                return jsonify({'error': 'Failed to download image from URL'}), 400
            image_data = base64.b64encode(response.content).decode('utf-8')
            # Add data URL prefix if not present
            if not image_data.startswith('data:image'):
                image_data = f'data:image/jpeg;base64,{image_data}'
        
        # If the image data is a base64 string without the data URL prefix, add it
        if not image_data.startswith('data:image') and not image_data.startswith('http'):
            image_data = f'data:image/jpeg;base64,{image_data}'
        
        # Extract the base64 data from the data URL
        if image_data.startswith('data:image'):
            # Extract the base64 part after the comma
            image_data = image_data.split(',')[1]
        
        # Decode the base64 data
        image_bytes = base64.b64decode(image_data)
        
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        
        # Classify the image using the road damage detector
        try:
            # Initialize the detector if not already done
            if not hasattr(app, 'road_damage_detector'):
                app.road_damage_detector = RoadDamageDetector()
            
            # Classify the image
            result = app.road_damage_detector.classify_image(temp_path)
            
            # Check if classification was successful
            if not result.get('success', False):
                raise Exception(result.get('error', 'Unknown classification error'))
                
            # Extract the classification results
            is_road = result.get('is_road', False)
            probability = result.get('probability', 0.0)
            
            # Generate a temporary ID for the image
            image_id = None
            if is_road:
                # Generate an ID but don't save to database yet
                # We'll save it only after location details are provided
                image_id = str(uuid.uuid4())
                
                # Store the image data in a temporary cache
                # This will be used when the user provides location details
                if not hasattr(app, 'temp_images'):
                    app.temp_images = {}
                    
                app.temp_images[image_id] = {
                    'image_data': image_data,
                    'user_id': str(current_user['_id']),
                    'uploaded_at': datetime.utcnow(),
                    'is_road': True,
                    'probability': float(probability),
                    'status': 'pending',
                    'expires_at': datetime.utcnow() + timedelta(hours=1)  # Expire after 1 hour
                }
                
                # Clean up expired images from the cache
                now = datetime.utcnow()
                expired_ids = [id for id, data in app.temp_images.items() 
                              if data.get('expires_at', now) < now]
                for expired_id in expired_ids:
                    app.temp_images.pop(expired_id, None)
            
            # Remove the temporary file
            os.unlink(temp_path)
            
            return jsonify({
                'success': True,
                'is_road': bool(is_road),
                'probability': float(probability),
                'image_id': image_id
            })
            
        except Exception as e:
            # Remove the temporary file in case of error
            os.unlink(temp_path)
            print(f"Error classifying image: {str(e)}")
            return jsonify({'error': f'Error classifying image: {str(e)}'}), 500
    
    except Exception as e:
        print(f"Error in classify_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add a registration endpoint
@app.route('/api/users/register', methods=['POST'])
def register():
    try:
        data = request.json
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Check if user already exists
        existing_user = users_collection.find_one({'email': data['email']})
        if existing_user:
            return jsonify({'error': 'User with this email already exists'}), 409  # Conflict
        
        # Hash the password
        hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
        
        # Create new user document
        new_user = {
            '_id': str(uuid.uuid4()),
            'email': data['email'],
            'password': hashed_password,
            'name': data.get('name', data['email'].split('@')[0]),
            'role': data.get('role', 'user'),  # Use role from request or default to 'user'
            'created_at': datetime.utcnow()
        }
        
        # Save to database
        result = users_collection.insert_one(new_user)
        print(f"User registered with ID: {new_user['_id']}")
        
        # Return success without password
        user_response = {k: v for k, v in new_user.items() if k != 'password'}
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user': user_response
        })
        
    except Exception as e:
        print(f"Error in registration: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add a login endpoint for user authentication
@app.route('/api/users/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Get the user from the database
        user = users_collection.find_one({'email': data['email']})
        
        if not user:
            # For development, create a new user if not found
            print(f"User not found, creating new user with email: {data['email']}")
            
            # Check if the requested role is 'admin' - don't allow auto-creation of admin accounts
            requested_role = data.get('role', 'user')
            if requested_role == 'admin':
                return jsonify({'error': 'Access denied for this role'}), 403
                
            hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
            
            new_user = {
                '_id': str(uuid.uuid4()),
                'email': data['email'],
                'password': hashed_password,
                'name': data.get('name', data['email'].split('@')[0]),
                'role': 'user',  # Always create as regular user during auto-creation
                'created_at': datetime.utcnow()
            }
            
            users_collection.insert_one(new_user)
            user = new_user
        else:
            # Verify password if user exists
            if 'password' in user:
                # Check if the password is hashed
                # Convert password to bytes if it's a string
                password_bytes = user['password']
                if isinstance(password_bytes, str):
                    password_bytes = password_bytes.encode('utf-8')
                
                # Now check if it starts with the bcrypt prefix
                if not password_bytes.startswith(b'$2b$'):
                    # Hash the password if it's not already hashed
                    password_to_hash = user['password']
                    if isinstance(password_to_hash, bytes):
                        password_to_hash = password_to_hash.decode('utf-8')
                    
                    hashed_password = bcrypt.hashpw(password_to_hash.encode('utf-8'), bcrypt.gensalt())
                    users_collection.update_one({'_id': user['_id']}, {'$set': {'password': hashed_password}})
                    user['password'] = hashed_password
                
                # Verify the password
                # Ensure the stored password is in bytes format
                stored_password = user['password']
                if isinstance(stored_password, str):
                    stored_password = stored_password.encode('utf-8')
                
                # Check the password
                if not bcrypt.checkpw(data['password'].encode('utf-8'), stored_password):
                    return jsonify({'error': 'Invalid password'}), 401
                
                # Check if the requested role matches the user's role
                requested_role = data.get('role', 'user')
                user_role = user.get('role', 'user')
                
                if requested_role != user_role:
                    return jsonify({'error': 'Access denied for this role'}), 403
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': str(user['_id']),
            'email': user['email'],
            'role': user.get('role', 'user'),
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        # Remove sensitive data before returning
        if 'password' in user:
            del user['password']
        
        # Convert ObjectId to string for JSON serialization
        user['_id'] = str(user['_id'])
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': user
        })
    
    except Exception as e:
        print(f"Error in login: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add a test login route that always works (for testing only)
@app.route('/api/test-login', methods=['POST'])
def test_login():
    """This endpoint creates a test user and returns login credentials - USE FOR TESTING ONLY"""
    try:
        print("Test login endpoint called")
        # Create a test user
        test_user = {
            '_id': 'test123',
            'email': 'test@example.com',
            'name': 'Test User',
            'role': 'admin'  # Give admin privileges for testing
        }
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': test_user['_id'],
            'email': test_user['email'],
            'role': test_user['role'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        print(f"Generated test token: {token[:20]}...")
        
        return jsonify({
            'message': 'Test login successful',
            'token': token,
            'user': test_user
        })
    except Exception as e:
        print(f"Error in test login: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add endpoint to get user's images
@app.route('/api/images/user', methods=['GET'])
@token_required
def get_user_images(current_user):
    try:
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        
        # Calculate skip for pagination
        skip = (page - 1) * limit
        
        # Query images for the current user
        user_id = str(current_user['_id'])
        print(f"Fetching images for user: {user_id}")
        
        # Get total count for pagination
        total_images = images_collection.count_documents({'user_id': user_id})
        total_pages = (total_images + limit - 1) // limit  # Ceiling division
        
        # Get images with pagination
        images_cursor = images_collection.find(
            {'user_id': user_id}
            # Include all fields including image_data
        ).sort('uploaded_at', -1).skip(skip).limit(limit)
        
        # Convert cursor to list and process each image
        images = []
        for img in images_cursor:
            # Convert ObjectId to string for JSON serialization
            img['_id'] = str(img['_id'])
            
            # Format dates for better readability
            if 'uploaded_at' in img:
                img['uploaded_at'] = img['uploaded_at'].isoformat()
            
            # Format image data for frontend display
            if 'image_data' in img and img['image_data']:
                # Check if image_data is already in the correct format
                if isinstance(img['image_data'], str):
                    if not img['image_data'].startswith('data:image'):
                        img['image_data'] = f"data:image/jpeg;base64,{img['image_data']}"
                # If it's bytes, convert to base64 string
                elif isinstance(img['image_data'], bytes):
                    img['image_data'] = f"data:image/jpeg;base64,{base64.b64encode(img['image_data']).decode('utf-8')}"
            
            images.append(img)
        
        return jsonify({
            'success': True,
            'images': images,
            'page': page,
            'pages': total_pages,
            'total': total_images
        })
        
    except Exception as e:
        print(f"Error fetching user images: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Admin endpoints
@app.route('/api/images/admin', methods=['GET'])
@token_required
def get_admin_images(current_user):
    try:
        # Check if user is admin
        if current_user.get('role') != 'admin':
            return jsonify({'error': 'Access denied. Admin privileges required.'}), 403
        
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        
        # Calculate skip for pagination
        skip = (page - 1) * limit
        
        # Get total count for pagination
        total_images = images_collection.count_documents({})
        total_pages = (total_images + limit - 1) // limit  # Ceiling division
        
        # Get all images with pagination
        images_cursor = images_collection.find().sort('uploaded_at', -1).skip(skip).limit(limit)
        
        # Convert cursor to list and process each image
        images = []
        for img in images_cursor:
            # Convert ObjectId to string for JSON serialization
            img['_id'] = str(img['_id'])
            
            # Format dates for better readability
            if 'uploaded_at' in img:
                img['uploaded_at'] = img['uploaded_at'].isoformat()
            
            # Format image data for frontend display
            if 'image_data' in img and img['image_data']:
                # Check if image_data is already in the correct format
                if isinstance(img['image_data'], str):
                    if not img['image_data'].startswith('data:image'):
                        img['image_data'] = f"data:image/jpeg;base64,{img['image_data']}"
                # If it's bytes, convert to base64 string
                elif isinstance(img['image_data'], bytes):
                    img['image_data'] = f"data:image/jpeg;base64,{base64.b64encode(img['image_data']).decode('utf-8')}"
            
            images.append(img)
        
        return jsonify({
            'success': True,
            'images': images,
            'page': page,
            'pages': total_pages,
            'total': total_images
        })
        
    except Exception as e:
        print(f"Error fetching admin images: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Function to classify road images using our enhanced prediction functionality
def classify_road_image(image_path, mode='multiclass'):
    """
    Classify a road image using the road damage detector.
    
    Args:
        image_path: Path to the image file
        mode: Classification mode ('binary' or 'multiclass')
        
    Returns:
        Dictionary with classification results
    """
    try:
        # Use the road damage detector to analyze the image
        result = road_damage_detector.analyze_road_image(image_path)
        
        if not result.get('success', False):
            return {
                'success': False,
                'error': result.get('error', 'Unknown error during analysis')
            }
        
        # Return the enhanced result with multiclass classification and bounding boxes
        return {
            'success': True,
            'is_road': result.get('is_road', False),
            'road_type': result.get('road_type', 'Unknown'),
            'detections': result.get('detections', []),
            'damage_type': result.get('damage_type', 'unknown'),
            'severity': result.get('severity', 'low'),
            'summary': result.get('summary', 'Analysis completed')
        }
        
    except Exception as e:
        print(f"Error in classify_road_image: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/api/images/admin/classify/<image_id>', methods=['POST'])
@token_required
def admin_classify_image(current_user, image_id):
    try:
        print(f"Starting admin_classify_image with image_id: {image_id}")
        print(f"Current user: {current_user}")
        
        # Check if user is admin
        if current_user.get('role') != 'admin':
            print(f"Access denied. User role: {current_user.get('role')}")
            return jsonify({'error': 'Access denied. Admin privileges required.'}), 403
        
        # Find the image in the database
        try:
            print(f"Looking for image with ID: {image_id}")
            
            # Try first with the ID as is (could be a UUID string)
            image = images_collection.find_one({'_id': image_id})
            
            # If not found, try with ObjectId (for MongoDB's native IDs)
            if not image:
                try:
                    object_id = ObjectId(image_id)
                    print(f"Trying with ObjectId: {object_id}")
                    image = images_collection.find_one({'_id': object_id})
                except Exception as e:
                    print(f"Not a valid ObjectId, but that's okay: {str(e)}")
            
            print(f"Image found: {image is not None}")
            
            # If still not found, return a 404
            if not image:
                print(f"No image found with ID: {image_id}")
                return jsonify({'error': 'Image not found'}), 404
            
            # Check if the image is already classified
            if image.get('status') == 'classified':
                print(f"Image is already classified")
                
                # If the image is already classified, return the existing classification results
                return jsonify({
                    'success': True,
                    'message': 'Image is already classified',
                    'classification': {
                        'is_road': image.get('is_road', True),
                        'damage_type': image.get('damage_type', 'unknown'),
                        'confidence': image.get('confidence', 0.0),
                        'severity': image.get('severity', 'unknown')
                    },
                    'detections': image.get('detections', []),
                    'output_image_url': image.get('output_image_url', '')
                })
        except Exception as e:
            print(f"Error finding image: {str(e)}")
            return jsonify({'error': f'Error finding image: {str(e)}'}), 400
        
        # Get image data
        print(f"Getting image data")
        image_data = image.get('image_data', '')
        print(f"Image data type: {type(image_data)}, length: {len(str(image_data)) if image_data else 0}")
        
        if not image_data:
            print("Image data not found")
            return jsonify({'error': 'Image data not found'}), 400
        
        try:
            # Strip data URL prefix if present
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                print("Stripping data URL prefix")
                # Extract the base64 part
                image_data = image_data.split(',')[1]
                print(f"Base64 data length after stripping: {len(image_data)}")
            
            # Convert to bytes if it's a string
            if isinstance(image_data, str):
                print("Converting string to bytes")
                image_data = image_data.encode('utf-8')
                print(f"Bytes length: {len(image_data)}")
            
            # Decode base64 to binary
            print("Decoding base64 to binary")
            image_binary = base64.b64decode(image_data)
            print(f"Binary data length: {len(image_binary)}")
        except Exception as e:
            print(f"Error processing image data: {str(e)}")
            return jsonify({'error': f'Error processing image data: {str(e)}'}), 400
        
        # Create a temporary file for the image
        print("Creating temporary file for the image")
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                print(f"Temporary file created: {temp_file.name}")
                temp_file.write(image_binary)
                temp_file_path = temp_file.name
                print(f"Image written to temporary file: {temp_file_path}")
        except Exception as e:
            print(f"Error creating temporary file: {str(e)}")
            return jsonify({'error': f'Error creating temporary file: {str(e)}'}), 500
        
        try:
            # Use the road damage detector to classify the image
            print("Using road damage detector to classify the image")
            
            # Ensure output directory exists
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Process the image with the road damage detector
            classification_result = road_damage_detector.classify_damage_type(temp_file_path)
            print(f"Classification result: {classification_result}")
            
            if not classification_result.get('success', False):
                print(f"Classification failed: {classification_result.get('error', 'Unknown error')}")
                return jsonify({
                    'success': False,
                    'error': f"Classification failed: {classification_result.get('error', 'Unknown error')}"
                }), 500
            
            # Detect objects (bounding boxes) using the BB_MODEL1.pt model
            detection_result = road_damage_detector.detect_with_bb_model(temp_file_path)
            print(f"Detection result: {detection_result}")
            
            if not detection_result.get('success', False):
                print(f"Detection failed: {detection_result.get('error', 'Unknown error')}")
                return jsonify({
                    'success': False,
                    'error': f"Detection failed: {detection_result.get('error', 'Unknown error')}"
                }), 500
                
            # Extract detections from the result
            detections = detection_result.get('detections', [])
            
            # Create output image with bounding boxes
            output_image_path = os.path.join(output_dir, f"analyzed_{os.path.basename(temp_file_path)}")
            
            # Draw bounding boxes on the image
            img = cv2.imread(temp_file_path)
            
            # If no detections were found but we have a classification result, add a default bounding box
            if not detections and classification_result.get('success', False):
                print("No detections found, adding a default bounding box based on classification")
                
                # Get image dimensions
                height, width = img.shape[:2]
                
                # Get damage type from classification result
                damage_type = classification_result.get('damage_type', 'pothole')
                confidence = classification_result.get('confidence', 0.8)
                
                # Adjust bounding box size and position based on damage type
                if damage_type.lower() == 'pothole':
                    # For potholes, create a more circular bounding box in the center
                    # covering ~30% of the image
                    box_size = min(width, height) // 3
                    x1 = (width - box_size) // 2
                    y1 = (height - box_size) // 2
                    x2 = x1 + box_size
                    y2 = y1 + box_size
                    severity = 'high' if confidence > 0.8 else 'medium'
                elif damage_type.lower() == 'crack':
                    # For cracks, create a more elongated bounding box
                    # covering ~40% of the image width but thinner height
                    box_width = width // 2
                    box_height = height // 6
                    x1 = (width - box_width) // 2
                    y1 = (height - box_height) // 2
                    x2 = x1 + box_width
                    y2 = y1 + box_height
                    severity = 'medium' if confidence > 0.8 else 'low'
                else:
                    # Default case for other damage types
                    box_width = width // 3
                    box_height = height // 3
                    x1 = (width - box_width) // 2
                    y1 = (height - box_height) // 2
                    x2 = x1 + box_width
                    y2 = y1 + box_height
                    severity = 'medium'
                
                # Add the detection with appropriate properties
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_name': damage_type,
                    'class': damage_type,
                    'severity': severity
                })
            for detection in detections:
                bbox = detection.get('bbox', [])
                if not bbox or len(bbox) != 4:
                    continue  # Skip if bbox is invalid
                    
                confidence = detection.get('confidence', 0)
                class_name = detection.get('class_name', detection.get('class', 'unknown'))
                
                # Convert bbox coordinates to integers
                x1, y1, x2, y2 = map(int, bbox)
                
                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Calculate the dimensions of the image
                img_height, img_width = img.shape[:2]
                
                # Create a much larger bounding box (50% of image size)
                box_size = min(img_width, img_height) // 2
                
                # Calculate new coordinates for the larger box
                new_x1 = max(0, center_x - box_size // 2)
                new_y1 = max(0, center_y - box_size // 2)
                new_x2 = min(img_width, center_x + box_size // 2)
                new_y2 = min(img_height, center_y + box_size // 2)
                
                # Draw bounding box with a thicker green line - no labels
                cv2.rectangle(img, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 3)  # Green color, thicker line
                
                # Calculate severity based on bounding box size
                box_area = (x2 - x1) * (y2 - y1)
                img_area = img.shape[0] * img.shape[1]
                area_ratio = box_area / img_area
                
                # Assign severity based on area ratio
                if area_ratio > 0.2:
                    detection['severity'] = 'high'
                elif area_ratio > 0.1:
                    detection['severity'] = 'medium'
                else:
                    detection['severity'] = 'low'
            
            # Save the output image
            cv2.imwrite(output_image_path, img)
            print(f"Output image saved to: {output_image_path}")
            
            # Prepare the response
            damage_type = classification_result.get('damage_type', 'unknown')
            confidence = classification_result.get('confidence', 0.0)
            
            # Determine overall severity based on the highest severity detection
            severity = 'unknown'
            if detections:
                severity_levels = [d.get('severity', 'low') for d in detections]
                if 'high' in severity_levels:
                    severity = 'high'
                elif 'medium' in severity_levels:
                    severity = 'medium'
                elif 'low' in severity_levels:
                    severity = 'low'
            
            # Format detections for response
            formatted_detections = []
            for detection in detections:
                formatted_detections.append({
                    'class': detection.get('class_name', detection.get('class', 'unknown')),
                    'confidence': detection.get('confidence', 0.0),
                    'bbox': detection.get('bbox', [0, 0, 0, 0]),
                    'severity': detection.get('severity', 'unknown')
                })
            
            response = {
                'success': True,
                'classification': {
                    'is_road': True,  # Assuming it's a road image since it's in the admin dashboard
                    'damage_type': damage_type,
                    'confidence': confidence,
                    'severity': severity
                },
                'detections': formatted_detections,
                'output_image_url': f"/output/{os.path.basename(output_image_path)}"
            }
            
            # Update the image classification in the database
            update_data = {
                'is_road': True,
                'damage_type': damage_type,
                'confidence': confidence,
                'severity': severity,
                'detections': formatted_detections,
                'classified_at': datetime.utcnow(),
                'classified_by': current_user['email'],
                'output_image_url': f"/output/{os.path.basename(output_image_path)}",
                'status': 'classified'  # Update status from 'pending' to 'classified'
            }
            
            # Update the image in the database
            if isinstance(image['_id'], str):
                images_collection.update_one({'_id': image['_id']}, {'$set': update_data})
            else:
                images_collection.update_one({'_id': ObjectId(image_id)}, {'$set': update_data})
            
            print(f"Classification data saved to database")
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error classifying image: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error classifying image: {str(e)}'}), 500
        
        finally:
            # Clean up the temporary file
            try:
                if 'temp_file_path' in locals():
                    os.unlink(temp_file_path)
                    print(f"Temporary file removed: {temp_file_path}")
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")
    
    except Exception as e:
        print(f"Error in admin classification: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/admin/update/<image_id>', methods=['PUT'])
@token_required
def update_image_address(current_user, image_id):
    try:
        # Check if user is admin
        if current_user.get('role') != 'admin':
            return jsonify({'error': 'Access denied. Admin privileges required.'}), 403
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Find the image in the database
        image = images_collection.find_one({'_id': ObjectId(image_id)})
        if not image:
            return jsonify({'error': 'Image not found'}), 404
        
        # Prepare update data
        update_data = {}
        
        # Update name if provided
        if 'name' in data:
            update_data['name'] = data['name']
        
        # Update address if provided
        if 'address' in data:
            update_data['address'] = data['address']
        
        # Update status if provided
        if 'status' in data:
            update_data['status'] = data['status']
        
        # Add updated_at timestamp
        update_data['updated_at'] = datetime.utcnow()
        update_data['updated_by'] = current_user['email']
        
        # Update the image in the database
        result = images_collection.update_one(
            {'_id': ObjectId(image_id)},
            {'$set': update_data}
        )
        
        if result.modified_count == 0:
            return jsonify({'warning': 'No changes made to the image'}), 200
        
        return jsonify({
            'success': True,
            'message': 'Image updated successfully'
        })
        
    except Exception as e:
        print(f"Error updating image: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add a test endpoint to create an admin user
@app.route('/api/create-admin', methods=['GET'])
def create_admin_user():
    try:
        # Check if admin already exists
        admin = users_collection.find_one({'email': 'admin@safestreet.com'})
        if admin:
            # Delete existing admin
            users_collection.delete_one({'email': 'admin@safestreet.com'})
            print("Deleted existing admin user")
        
        # Create a simple password
        password = 'admin123'
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create admin user
        admin_user = {
            '_id': str(uuid.uuid4()),
            'email': 'admin@safestreet.com',
            'password': hashed_password,
            'name': 'Admin User',
            'role': 'admin',
            'created_at': datetime.utcnow()
        }
        
        # Insert admin user
        result = users_collection.insert_one(admin_user)
        print(f"Created admin user with ID: {admin_user['_id']}")
        
        return jsonify({
            'success': True,
            'message': 'Admin user created successfully',
            'email': 'admin@safestreet.com',
            'password': 'admin123'  # Only for testing purposes
        })
    except Exception as e:
        print(f"Error creating admin user: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Serve static files from the uploads directory
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

# Serve output files
@app.route('/output/<path:filename>')
def serve_output(filename):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    return send_from_directory(output_dir, filename)

if __name__ == '__main__':
    print("Starting server on http://localhost:5001")
    app.run(host='localhost', port=5001, debug=True)
