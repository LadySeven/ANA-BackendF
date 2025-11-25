#!/usr/bin/env python3
"""
Simple script to test the ANA Backend API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_signup():
    """Test user signup"""
    print("Testing signup...")
    url = f"{BASE_URL}/auth/signup"
    data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_login():
    """Test user login"""
    print("\nTesting login...")
    url = f"{BASE_URL}/auth/login"
    data = {
        "username": "testuser",
        "password": "password123"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            return response.json().get("access_token")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_me(token):
    """Test getting current user info"""
    print("\nTesting /auth/me...")
    url = f"{BASE_URL}/auth/me"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("ANA Backend API Test")
    print("=" * 50)
    
    # Test signup
    signup_success = test_signup()
    
    if signup_success:
        # Test login
        token = test_login()
        
        if token:
            # Test /auth/me
            me_success = test_me(token)
            
            if me_success:
                print("\n✅ All tests passed!")
            else:
                print("\n❌ /auth/me test failed")
        else:
            print("\n❌ Login test failed")
    else:
        print("\n❌ Signup test failed")

if __name__ == "__main__":
    main()
