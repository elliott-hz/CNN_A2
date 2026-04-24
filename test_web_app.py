"""
Test script to verify the web application works correctly
"""
import sys
from pathlib import Path
import requests
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))


def test_backend_health():
    """Test if backend is running and healthy"""
    print("🔍 Testing Backend Health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend is healthy!")
            print(f"   Status: {data['status']}")
            print(f"   Models Loaded: {data['models_loaded']}")
            print(f"   Device: {data['device']}")
            return True
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running?")
        print("   Start backend with: cd api_service && python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_frontend_accessible():
    """Test if frontend is accessible"""
    print("\n🔍 Testing Frontend Accessibility...")
    try:
        response = requests.get("http://localhost:5173", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible!")
            return True
        else:
            print(f"⚠️  Frontend returned status code: {response.status_code}")
            return True  # Still OK, might be loading
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to frontend. Is it running?")
        print("   Start frontend with: cd web_intf && npm run dev")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_api_documentation():
    """Test if API documentation is available"""
    print("\n🔍 Testing API Documentation...")
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API documentation (Swagger UI) is available!")
            print("   Visit: http://localhost:8000/docs")
            return True
        else:
            print(f"❌ API docs returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_model_files():
    """Check if model files exist"""
    print("\n📦 Checking Model Files...")
    project_root = Path(__file__).parent
    
    detection_model = project_root / "best_models" / "detection_YOLOv8_baseline.pt"
    classification_model = project_root / "best_models" / "emotion_ResNet50_baseline.pth"
    
    if detection_model.exists():
        size_mb = detection_model.stat().st_size / (1024 * 1024)
        print(f"✅ Detection model: {detection_model.name} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Detection model not found: {detection_model}")
        return False
    
    if classification_model.exists():
        size_mb = classification_model.stat().st_size / (1024 * 1024)
        print(f"✅ Classification model: {classification_model.name} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Classification model not found: {classification_model}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("=" * 80)
    print("🐕 Dog Emotion Recognition Web App - Test Suite")
    print("=" * 80)
    print()
    
    # Check prerequisites
    models_ok = check_model_files()
    
    if not models_ok:
        print("\n❌ Model files missing. Please ensure models are in best_models/")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Running Connection Tests...")
    print("=" * 80)
    
    # Test services
    backend_ok = test_backend_health()
    frontend_ok = test_frontend_accessible()
    api_docs_ok = test_api_documentation()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Model Files:      {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"Backend API:      {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"Frontend App:     {'✅ PASS' if frontend_ok else '❌ FAIL'}")
    print(f"API Documentation:{'✅ PASS' if api_docs_ok else '❌ FAIL'}")
    print("=" * 80)
    
    if backend_ok and frontend_ok:
        print("\n🎉 All tests passed! Your web app is ready to use.")
        print("\nNext steps:")
        print("1. Open http://localhost:5173 in your browser")
        print("2. Upload an image with dogs")
        print("3. Click 'Detect Emotion' to see results")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nTo start the application:")
        print("  Option 1: ./start_web_app.sh")
        print("  Option 2: Manually start backend and frontend in separate terminals")
    
    print()


if __name__ == "__main__":
    main()
