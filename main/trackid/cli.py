import os
import json
import numpy as np
import librosa
import argparse

def get_db_path():
    path = os.path.join(os.path.expanduser("~"), "Documents")
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.join(path, "trackid_db.json")

DB_FILE = get_db_path()

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_db(db):
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(db, f, indent=4)
    except Exception as e:
        print(f"Error writing to database: {e}")

def get_fingerprint(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    try:
        y, sr = librosa.load(filepath, sr=22050, duration=30)
        stft = np.abs(librosa.stft(y))
        spectrum = np.mean(stft, axis=1)
        
        mx = np.max(spectrum)
        if mx > 0:
            spectrum = spectrum / mx
            
        return spectrum.tolist()
    except Exception as e:
        print(f"Failed to process {filepath}: {e}")
        return None

def add_track(filepath, name):
    db = load_db()
    print(f"Processing: {name}")
    fp = get_fingerprint(filepath)
    if fp:
        db[name] = fp
        save_db(db)
        print("Track saved successfully.")

def scan_track(filepath):
    db = load_db()
    if not db:
        print("Database is empty. Add tracks first.")
        return

    print("Scanning sample...")
    query_fp = get_fingerprint(filepath)
    if not query_fp:
        return

    q_vec = np.array(query_fp)
    best_match = None
    best_score = -1.0

    for name, stored_fp in db.items():
        s_vec = np.array(stored_fp)
        
        dot = np.dot(q_vec, s_vec)
        q_norm = np.linalg.norm(q_vec)
        s_norm = np.linalg.norm(s_vec)
        
        score = dot / (q_norm * s_norm) if (q_norm * s_norm) > 0 else 0
        
        if score > best_score:
            best_score = score
            best_match = name

    print("\n--- Scan Results ---")
    if best_score > 0.85:
        print(f"Match: {best_match} ({best_score:.2%})")
    else:
        print("No definitive match found.")
        if best_match:
            print(f"Closest guess: {best_match} ({best_score:.2%})")
    print("--------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Track ID Utility")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    add_p = subparsers.add_parser("add")
    add_p.add_argument("file")
    add_p.add_argument("--name", required=True)

    scan_p = subparsers.add_parser("scan")
    scan_p.add_argument("file")

    args = parser.parse_args()

    if args.cmd == "add":
        add_track(args.file, args.name)
    elif args.cmd == "scan":
        scan_track(args.file)

if __name__ == "__main__":
    main()