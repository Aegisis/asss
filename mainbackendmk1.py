import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

class Backend:
    def __init__(self):
        self.conn = sqlite3.connect('users.db', check_same_thread=False)
        self.c = self.conn.cursor()
        self.setup_database()
        self.load_menu_data()

    def setup_database(self):
        """Initialize database tables"""
        # Create users table
        self.c.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT NOT NULL,
                          phone TEXT,
                          dob TEXT,
                          face_encoding BLOB,
                          preferences TEXT)''')
        
        # Create orders table with TEXT type for order_date
        self.c.execute('''CREATE TABLE IF NOT EXISTS orders
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id INTEGER,
                          items TEXT,
                          order_date TEXT,
                          total_amount REAL,
                          FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        self.conn.commit()

    def load_menu_data(self):
        """Load and process menu data"""
        try:
            self.menu_data = pd.read_csv(r"C:\Users\omila\OneDrive\Desktop\gpro\New folder\indian_food.csv")
            print(f"Loaded menu data: {self.menu_data.head()}")  # Debugging line
            self.process_menu_data()
        except Exception as e:
            print(f"Error loading menu data: {e}")
            self.menu_data = None

    def process_menu_data(self):
        """Process menu data for recommendations"""
        if self.menu_data is not None:
            # Create feature vectors for recommendations
            self.menu_data['Features'] = self.menu_data['Ingredients'].fillna('') + ' ' + \
                                       self.menu_data['Course'].fillna('') + ' ' + \
                                       self.menu_data['Flavour'].fillna('')
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = tfidf.fit_transform(self.menu_data['Features'])

    def verify_face(self, face_embedding):
        """Verify face against database using DeepFace's embeddings comparison"""
        try:
            # Get all users with face encodings
            self.c.execute("SELECT id, name, face_encoding FROM users")
            users = self.c.fetchall()
            
            print(f"Comparing with {len(users)} stored faces")
            
            best_match = None
            highest_similarity = 0
            
            # Convert input embedding to correct shape if needed
            face_embedding = np.array(face_embedding).flatten()
            print(f"Input embedding size: {face_embedding.shape}")
            
            for user_id, name, stored_encoding_blob in users:
                if stored_encoding_blob is None:
                    continue
                    
                try:
                    # Convert blob back to numpy array
                    stored_encoding = np.frombuffer(stored_encoding_blob, dtype=np.float64)
                    print(f"Stored embedding size for user {user_id}: {stored_encoding.shape}")
                    
                    # Ensure consistent size
                    if len(stored_encoding) != len(face_embedding):
                        print(f"Warning: Mismatched sizes for user {user_id}. Expected {len(face_embedding)}, got {len(stored_encoding)}.")
                        continue
                    
                    # Calculate cosine similarity
                    similarity = 1 - spatial.distance.cosine(face_embedding, stored_encoding)
                    confidence = similarity * 100
                    
                    print(f"Comparing with user {name} (ID: {user_id}), confidence: {confidence:.2f}%")
                    
                    # Update best match if this is the highest similarity and above 50% threshold
                    if similarity > highest_similarity and similarity > 0.50:  
                        highest_similarity = similarity
                        best_match = (user_id, name, confidence)
                        
                except Exception as e:
                    print(f"Error comparing face encodings for user {user_id}: {e}")
                    continue
            
            if best_match:
                user_id, name, confidence = best_match
                print(f"Found match: {name} (ID: {user_id}) with confidence: {confidence:.2f}%")
                return user_id, name
            
            print("No matching face found in database")
            return None, None
            
        except Exception as e:
            print(f"Error in face verification: {e}")
            return None, None

    def register_user(self, name, phone, dob, face_encoding):
        """Register a new user with face encoding"""
        try:
            # First check if user already exists by face
            existing_user = self.verify_face(face_encoding)
            if existing_user[0] is not None:
                print(f"User already exists with ID: {existing_user[0]}")
                return existing_user[0]
            
            # Ensure face encoding is flattened and consistent size
            face_encoding = np.array(face_encoding).flatten()
            print(f"Registering face encoding of size: {face_encoding.shape}")
            face_encoding_bytes = face_encoding.tobytes()
            
            # Insert new user
            self.c.execute('''INSERT INTO users (name, phone, dob, face_encoding)
                            VALUES (?, ?, ?, ?)''',
                         (name, phone, dob, face_encoding_bytes))
            self.conn.commit()
            
            # Get the ID of the newly inserted user
            user_id = self.c.lastrowid
            print(f"New user registered with ID: {user_id}")
            return user_id
            
        except Exception as e:
            print(f"Error registering user: {e}")
            self.conn.rollback()
            return None

    def get_recommendations(self, user_id, detected_emotion=None, n_recommendations=5):
        """Get food recommendations based on user history and emotion"""
        try:
            # Get user's past orders
            self.c.execute("SELECT items FROM orders WHERE user_id = ? ORDER BY order_date DESC", (user_id,))
            orders_result = self.c.fetchone()
            
            if not orders_result:
                return self.menu_data.sample(n_recommendations)['Name'].tolist()
            
            # Convert orders string to list
            orders = orders_result[0].split(',') if orders_result[0] else []
            
            if not orders:
                return self.menu_data.sample(n_recommendations)['Name'].tolist()
            
            # Get indices of previously ordered items
            ordered_indices = []
            for order in orders:
                idx = self.menu_data[self.menu_data['Name'] == order.strip()].index
                if not idx.empty:
                    ordered_indices.append(idx[0])
            
            if ordered_indices:
                # Convert to numpy array and reshape
                ordered_features = np.asarray(self.tfidf_matrix[ordered_indices].mean(axis=0)).flatten()
                
                # Calculate similarity scores
                sim_scores = cosine_similarity(
                    ordered_features.reshape(1, -1),
                    self.tfidf_matrix.toarray()
                ).flatten()
                
                # Get top N similar items
                sim_indices = sim_scores.argsort()[-n_recommendations:][::-1]
                recommended_items = self.menu_data.iloc[sim_indices]['Name'].tolist()
                
                # Adjust recommendations based on detected emotion
                if detected_emotion:
                    # Example: Modify recommendations based on emotion
                    if detected_emotion == 'happy':
                        # Recommend more comfort food or desserts
                        comfort_foods = self.menu_data[self.menu_data['Flavour'] == 'Sweet']
                        recommended_items += comfort_foods.sample(min(2, len(comfort_foods)))['Name'].tolist()
                    elif detected_emotion == 'sad':
                        # Recommend more hearty or spicy food
                        hearty_foods = self.menu_data[self.menu_data['Flavour'] == 'Spicy']
                        recommended_items += hearty_foods.sample(min(2, len(hearty_foods)))['Name'].tolist()
                
                return list(set(recommended_items))[:n_recommendations]  # Remove duplicates and limit
                
            return self.menu_data.sample(n_recommendations)['Name'].tolist()
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return self.menu_data.sample(n_recommendations)['Name'].tolist()

    def save_order(self, user_id, items, total_amount):
        """Save order to database"""
        try:
            # Clean up items list to ensure no timestamps
            cleaned_items = [item.strip() for item in items 
                            if not any(c.isdigit() for c in item.strip())]
            
            # Convert items list to comma-separated string
            items_str = ','.join(cleaned_items)
            
            # Get current timestamp as string
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Insert order using the main connection
            self.c.execute("""INSERT INTO orders (user_id, items, order_date, total_amount) 
                             VALUES (?, ?, ?, ?)""",
                             (user_id, items_str, current_time, total_amount))
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error saving order: {e}")
            return False

    def get_user_orders(self, user_id):
        """Get user's order history"""
        try:
            # Get orders using the main connection
            self.c.execute("""SELECT id, items, order_date 
                             FROM orders 
                             WHERE user_id = ? 
                             ORDER BY order_date DESC""", (user_id,))
            
            orders = self.c.fetchall()
            
            # Format orders with proper timestamp handling
            formatted_orders = []
            for order_id, items_str, timestamp in orders:
                # Clean up items string
                if items_str:
                    # Remove any accidental timestamps from items string
                    items = [item.strip() for item in items_str.split(',') 
                            if not any(c.isdigit() for c in item.strip())]
                    items_str = ','.join(items)
                
                # Format timestamp
                if timestamp is None:
                    timestamp = "No date"
                elif not isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError, OSError):
                        timestamp = "No date"
                
                formatted_orders.append((order_id, items_str, timestamp))
            
            return formatted_orders
            
        except Exception as e:
            print(f"Error fetching orders: {e}")
            return []

    def update_user_preferences(self, user_id, preferences):
        """Update user preferences"""
        try:
            preferences_str = ','.join(preferences)
            self.c.execute("UPDATE users SET preferences = ? WHERE id = ?", 
                          (preferences_str, user_id))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating preferences: {e}")
            return False

    def close(self):
        """Close database connection"""
        self.conn.close()

    def get_user_details(self, user_id):
        """Get user details from database"""
        try:
            self.c.execute("""SELECT name, phone, dob 
                             FROM users 
                             WHERE id = ?""", (user_id,))
            result = self.c.fetchone()
            if result:
                return {
                    'name': result[0],
                    'phone': result[1],
                    'dob': result[2]
                }
            return None
        except Exception as e:
            print(f"Error fetching user details: {e}")
            return {
                'name': 'Unknown',
                'phone': 'Not available',
                'dob': 'Not available'
            }

    def get_user_name(self, user_id):
        """Get user name"""
        try:
            self.c.execute("SELECT name FROM users WHERE id = ?", (user_id,))
            result = self.c.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Error fetching user name: {e}")
            return None 

    def save_guest_order(self, guest_id, items, total_amount):
        """Save order for guest users"""
        try:
            # Convert items list to comma-separated string
            items_str = ','.join(items)
            
            # Get current timestamp as string
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Insert order using guest_id as -1 to indicate guest order
            self.c.execute("""INSERT INTO orders (user_id, items, order_date, total_amount) 
                             VALUES (?, ?, ?, ?)""",
                             (-1, items_str, current_time, total_amount))
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error saving guest order: {e}")
            return False 