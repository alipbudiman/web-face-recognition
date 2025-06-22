from flask import Flask, render_template, request, jsonify
import face_recognition
import cv2
import numpy as np
import base64
import os
import pickle
import json
from datetime import datetime

app = Flask(__name__)

# Folder untuk menyimpan data wajah
FACES_DIR = "faces_data"
ENCODINGS_FILE = "face_encodings.pkl"

# Buat folder jika belum ada
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

# Dictionary untuk menyimpan encoding wajah dan nama
# Format: {'nama': [list_of_encodings]}
known_faces_data = {}

def load_known_faces():
    """Memuat data wajah yang sudah tersimpan"""
    global known_faces_data
    
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                known_faces_data = pickle.load(f)
            
            total_encodings = sum(len(encodings) for encodings in known_faces_data.values())
            print(f"Berhasil memuat {len(known_faces_data)} orang dengan {total_encodings} encoding wajah")
            
            # Print detail setiap orang
            for name, encodings in known_faces_data.items():
                print(f"  - {name}: {len(encodings)} foto")
                
        except Exception as e:
            print(f"Error loading faces: {e}")
            known_faces_data = {}

def save_known_faces():
    """Menyimpan data wajah ke file"""
    try:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(known_faces_data, f)
        
        total_encodings = sum(len(encodings) for encodings in known_faces_data.values())
        print(f"Data wajah berhasil disimpan: {len(known_faces_data)} orang, {total_encodings} encoding")
    except Exception as e:
        print(f"Error saving faces: {e}")

def get_all_encodings_and_names():
    """Mengkonversi format data ke list untuk kompatibilitas"""
    all_encodings = []
    all_names = []
    
    for name, encodings in known_faces_data.items():
        for encoding in encodings:
            all_encodings.append(encoding)
            all_names.append(name)
    
    return all_encodings, all_names

def base64_to_image(base64_string):
    """Mengkonversi base64 string ke image array"""
    try:
        # Hapus prefix data:image/jpeg;base64, jika ada
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Konversi ke numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode ke image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Konversi BGR ke RGB (face_recognition menggunakan RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/capture')
def capture():
    """Halaman untuk capture wajah baru"""
    return render_template('capture.html')

@app.route('/validate')
def validate():
    """Halaman untuk validasi wajah"""
    return render_template('validate.html')

@app.route('/api/register_face', methods=['POST'])
def register_face():
    """API untuk mendaftarkan wajah baru"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data or 'name' not in data:
            return jsonify({'success': False, 'message': 'Data tidak lengkap'})
        
        name = data['name'].strip()
        image_data = data['image']
        
        if not name:
            return jsonify({'success': False, 'message': 'Nama tidak boleh kosong'})
        
        # Konversi base64 ke image
        img = base64_to_image(image_data)
        if img is None:
            return jsonify({'success': False, 'message': 'Gagal memproses gambar'})
        
        # Deteksi wajah dan buat encoding
        face_locations = face_recognition.face_locations(img)
        
        if len(face_locations) == 0:
            return jsonify({'success': False, 'message': 'Tidak ada wajah yang terdeteksi'})
        
        if len(face_locations) > 1:
            return jsonify({'success': False, 'message': 'Terdeteksi lebih dari satu wajah'})
        
        # Buat encoding wajah
        face_encodings = face_recognition.face_encodings(img, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({'success': False, 'message': 'Gagal membuat encoding wajah'})
        
        # Simpan encoding dengan sistem multiple images
        if name not in known_faces_data:
            known_faces_data[name] = []
        
        known_faces_data[name].append(face_encodings[0])
        
        # Batas maksimal foto per orang (opsional)
        MAX_PHOTOS_PER_PERSON = 10
        if len(known_faces_data[name]) > MAX_PHOTOS_PER_PERSON:
            # Hapus foto terlama jika sudah mencapai batas
            known_faces_data[name] = known_faces_data[name][-MAX_PHOTOS_PER_PERSON:]
        
        # Simpan gambar untuk referensi (terorganisir per person)
        person_folder = os.path.join(FACES_DIR, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_number = len(known_faces_data[name])
        filename = f"{name}_photo_{photo_number}_{timestamp}.jpg"
        filepath = os.path.join(person_folder, filename)
        
        # Konversi kembali ke BGR untuk penyimpanan
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_bgr)
        
        # Simpan data ke file
        save_known_faces()
        
        return jsonify({
            'success': True, 
            'message': f'Wajah {name} berhasil ditambahkan (Total: {len(known_faces_data[name])} foto)',
            'total_faces': len(known_faces_data),
            'photos_for_person': len(known_faces_data[name])
        })
        
    except Exception as e:
        print(f"Error in register_face: {e}")
        return jsonify({'success': False, 'message': 'Terjadi kesalahan server'})

@app.route('/api/validate_face', methods=['POST'])
def validate_face():
    """API untuk validasi wajah"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Data gambar tidak ada'})
        
        image_data = data['image']
        
        # Konversi base64 ke image
        img = base64_to_image(image_data)
        if img is None:
            return jsonify({'success': False, 'message': 'Gagal memproses gambar'})
        
        # Deteksi wajah
        face_locations = face_recognition.face_locations(img)
        
        if len(face_locations) == 0:
            return jsonify({'success': False, 'message': 'Tidak ada wajah yang terdeteksi'})
        
        # Buat encoding untuk wajah yang terdeteksi
        face_encodings = face_recognition.face_encodings(img, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({'success': False, 'message': 'Gagal membuat encoding wajah'})
        
        # Cek apakah ada wajah yang terdaftar
        if len(known_faces_data) == 0:
            return jsonify({'success': False, 'message': 'Belum ada wajah yang terdaftar'})
        
        # Bandingkan dengan wajah yang sudah terdaftar menggunakan improved matching
        results = []
        
        for face_encoding in face_encodings:
            # default treshold 60%
            name, confidence, recognized = improved_face_matching(face_encoding, threshold=0.6)
            results.append({
                'name': name,
                'confidence': confidence,
                'recognized': recognized
            })
        
        if any(result['recognized'] for result in results):
            best_result = max(results, key=lambda x: x['confidence'])
            return jsonify({
                'success': True,
                'message': f'Wajah dikenali sebagai {best_result["name"]}',
                'name': best_result['name'],
                'confidence': best_result['confidence']
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Wajah tidak dikenali',
                'name': 'Unknown',
                'confidence': 0
            })
            
    except Exception as e:
        print(f"Error in validate_face: {e}")
        return jsonify({'success': False, 'message': 'Terjadi kesalahan server'})

@app.route('/api/get_registered_faces')
def get_registered_faces():
    """API untuk mendapatkan daftar wajah yang terdaftar"""
    face_list = []
    for name, encodings in known_faces_data.items():
        face_list.append({
            'name': name,
            'photo_count': len(encodings)
        })
    
    return jsonify({
        'faces': list(known_faces_data.keys()),
        'faces_detail': face_list,
        'total': len(known_faces_data),
        'total_photos': sum(len(encodings) for encodings in known_faces_data.values())
    })

def improved_face_matching(input_encoding, threshold=0.6):
    """
    Improved face matching menggunakan multiple images per person
    Menggunakan voting system dan average distance
    """
    if len(known_faces_data) == 0:
        return None, 0, False
    
    person_scores = {}
    
    # Hitung score untuk setiap orang
    for person_name, person_encodings in known_faces_data.items():
        distances = face_recognition.face_distance(person_encodings, input_encoding)
        
        # Hitung berbagai metrik
        min_distance = np.min(distances)
        avg_distance = np.mean(distances)
        matches_count = np.sum(distances < threshold)
        
        # Score gabungan (semakin rendah semakin baik)
        # Pertimbangkan: jarak minimum, rata-rata, dan jumlah match
        score = (min_distance * 0.4) + (avg_distance * 0.3) + ((len(person_encodings) - matches_count) / len(person_encodings) * 0.3)
        
        person_scores[person_name] = {
            'score': score,
            'min_distance': min_distance,
            'avg_distance': avg_distance,
            'matches_count': matches_count,
            'total_photos': len(person_encodings),
            'match_ratio': matches_count / len(person_encodings)
        }
    
    # Cari yang terbaik
    best_person = min(person_scores.items(), key=lambda x: x[1]['score'])
    best_name = best_person[0]
    best_data = best_person[1]
    
    # Kriteria untuk recognition
    is_recognized = (
        best_data['min_distance'] < threshold and 
        best_data['match_ratio'] > 0.3  # Minimal 30% foto cocok
    )
    
    if is_recognized:
        confidence = round((1 - best_data['min_distance']) * 100, 2)
        return best_name, confidence, True
    else:
        return 'Unknown', 0, False

@app.route('/api/delete_person/<name>', methods=['DELETE'])
def delete_person(name):
    """API untuk menghapus data wajah seseorang"""
    try:
        if name not in known_faces_data:
            return jsonify({'success': False, 'message': f'Orang bernama {name} tidak ditemukan'})
        
        photo_count = len(known_faces_data[name])
        del known_faces_data[name]
        save_known_faces()
        
        # Hapus folder foto jika ada
        person_folder = os.path.join(FACES_DIR, name)
        if os.path.exists(person_folder):
            import shutil
            shutil.rmtree(person_folder)
        
        return jsonify({
            'success': True,
            'message': f'Data {name} berhasil dihapus ({photo_count} foto)',
            'remaining_people': len(known_faces_data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {e}'})

@app.route('/api/reset_all_data', methods=['GET'])
def reset_all_data():
    """API untuk reset semua data wajah"""
    try:
        global known_faces_data
        
        total_people = len(known_faces_data)
        total_photos = sum(len(encodings) for encodings in known_faces_data.values())
        
        known_faces_data = {}
        save_known_faces()
        
        # Hapus semua file foto
        if os.path.exists(FACES_DIR):
            import shutil
            shutil.rmtree(FACES_DIR)
            os.makedirs(FACES_DIR)
        
        return jsonify({
            'success': True,
            'message': f'Semua data berhasil dihapus ({total_people} orang, {total_photos} foto)'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {e}'})


if __name__ == '__main__':
    load_known_faces()
    print("=== Sistem Face Recognition ===")
    print("1. Buka http://localhost:5000 untuk halaman utama")
    print("2. /capture untuk mendaftarkan wajah baru")
    print("3. /validate untuk validasi wajah")
    print("===============================")
    app.run(debug=True, host='0.0.0.0', port=5000)