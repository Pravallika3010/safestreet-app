from pymongo import MongoClient
import os
import json

# Connect to MongoDB
try:
    # Get MongoDB connection string from environment variable or use default
    mongo_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
    client = MongoClient(mongo_uri)
    
    # List all databases
    print("\nAvailable databases:")
    for db_name in client.list_database_names():
        print(f"- {db_name}")
    
    # Check road_damage_db
    db = client['road_damage_db']
    
    # List all collections in road_damage_db
    print("\nCollections in road_damage_db:")
    for collection_name in db.list_collection_names():
        print(f"- {collection_name}")
    
    # Check images collection
    images_collection = db['images']
    
    # Count documents in images collection
    image_count = images_collection.count_documents({})
    print(f"\nTotal images in collection: {image_count}")
    
    # List the first few images (without image data to keep output manageable)
    if image_count > 0:
        print("\nSample images (showing first 3):")
        for img in images_collection.find({}, {'image_data': 0}).limit(3):
            # Convert ObjectId to string for JSON serialization
            img['_id'] = str(img['_id'])
            
            # Format dates for better readability
            if 'uploaded_at' in img:
                img['uploaded_at'] = img['uploaded_at'].isoformat()
            
            print(json.dumps(img, indent=2))
    else:
        print("\nNo images found in the collection.")
    
    # Check if there are any documents with address information
    address_count = images_collection.count_documents({'address': {'$exists': True}})
    print(f"\nImages with address information: {address_count}")
    
    # Check if there are any documents with is_road=True
    road_count = images_collection.count_documents({'is_road': True})
    print(f"Images classified as roads: {road_count}")
    
    print("\nMongoDB check completed successfully.")
    
except Exception as e:
    print(f"Error checking MongoDB: {str(e)}")
