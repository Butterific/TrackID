import os
import json
import numpy as np
import argparse
import sys

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
        # Inline import prevents the CLI from lagging on startup
        import librosa
        
        y, sr = librosa.load(filepath, sr=None, duration=10)
        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=2048))
        spectrum = np.mean(stft, axis=1)
        
        mx = np.max(spectrum)
        if mx > 0:
            spectrum = spectrum / mx
            
        return spectrum.tolist()
    except Exception as e:
        print(f"Failed to process {filepath}: {e}")
        return None

def add_track(args):
    db = load_db()
    print(f"Processing: {args.name}")
    fp = get_fingerprint(args.file)
    if fp:
        db[args.name] = fp
        save_db(db)
        print("Track saved successfully.")

def scan_track(args):
    db = load_db()
    if not db:
        print("Database is empty. Add tracks first.")
        return

    print("Scanning sample...")
    query_fp = get_fingerprint(args.file)
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
    if best_score > 0.80:
        print(f"Match: {best_match} ({best_score:.2%})")
    else:
        print("No definitive match found.")
        if best_match:
            print(f"Closest guess: {best_match} ({best_score:.2%})")
    print("--------------------\n")

def list_tracks(args):
    db = load_db()
    if not db:
        print("Database is empty.")
        return
    print("\n--- Indexed Tracks ---")
    for name in db.keys():
        print(f"- {name}")
    print("----------------------\n")

def remove_track(args):
    db = load_db()
    if args.name in db:
        del db[args.name]
        save_db(db)
        print(f"Removed '{args.name}' from the database.")
    else:
        print(f"Track '{args.name}' not found in the database.")

def clear_db(args):
    confirm = input("Are you sure you want to completely wipe the database? (y/N): ")
    if confirm.lower() == 'y':
        save_db({})
        print("Database cleared.")
    else:
        print("Operation cancelled.")

def show_help(args=None):
    print("""
Available Commands:
  add <file> --name "<name>"   Fingerprint and save a track to the database
  scan <file>                  Scan an audio clip to find a match
  list                         Show all indexed track names
  remove --name "<name>"       Delete a specific track from the database
  clear                        Wipe the entire local database
  about                        Show splash art and version info
  help                         Show this help menu
    """)

def draw_gradient_headphones(args=None):
    headphones_art = [
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣠⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀",
        "⠀⡆⠒⢲⠀⠀⠀⠀⠀⢠⡀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⠿⠿⠿⠿⣿⣷⣦⡀⠀⠀⠀",
        "⠀⡇⠀⡞⠀⠀⠀⠀⠀⠈⡟⠁⠀⠀⢀⣴⣿⠟⠉⠀⠀⠀⠀⠀⠀⠀⠉⠻⣿⣧⡀⠀",
        "⠸⠃⠸⠟⠀⠀⠀⠀⢀⣤⡿⠀⠀⢀⣾⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣧⠀",
        "⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠀⠀⠀⢸⣿⠃⠀⠀⢠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡇",
        "⠀⠀⠀⠀⠀⠀⠀⠐⡒⠒⠀⠀⠀⣹⣏⣀⣴⣿⣿⡄⠀⠀⠀⠀⠀⣴⣶⣶⣄⣐⣼⣿",
        "⠀⠀⠀⠀⠀⠀⠀⣠⣵⡀⠀⠀⠀⢽⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⠃",
        "⠀⠀⠀⠀⠀⠀⠀⠛⠉⠀⠀⠀⠀⠨⣿⣟⣿⣿⣿⣿⡧⠀⠀⠀⢸⣿⣿⣿⣟⣿⡟⠀",
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠄⠀⠀⢳⣿⣿⣿⣿⣿⣿⠀⠀⠀⣸⣿⣿⣿⣿⣿⣿⠀",
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⣿⣿⣿⣿⣿⣿⠁⠀⠀⣿⣿⣿⣿⣿⣿⠇⠀",
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⡿⠟⠀⠀⠀⠈⠻⠿⣿⣿⠿⠀⠀⠀⢸⣿⣿⣿⣿⠯⠀⠀",
        "⠀⠀⣀⠤⠤⠠⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣯⠀⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⠀",
        "⢠⠋⠀⠀⠀⠀⠀⠈⠳⡀⠀⠀⢀⣀⠀⠀⠀⠀⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        "⢸⡀⠀⠀⠀⠀⠀⠀⠀⡇⠀⢰⠉⠈⡆⠀⣀⡜⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠻⠤⣀⠀⠀⠀⠀⠀⢻⡀⠀⢑⡻⠋⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠉⠓⠤⣀⡀⠀⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⠀⠀⠈⠹⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
    ]
    
    total_lines = len(headphones_art)
    reset_code = "\033[0m"
    
    for index, line in enumerate(headphones_art):
        ratio = index / (total_lines - 1)
        r = int(0 + (255 * ratio))
        g = int(220 + (35 * ratio))
        b = 0
        print(f"\033[38;2;{r};{g};{b}m{line}{reset_code}")
        
    print()

    text_lines = ["TrackID", "V.1.0"]
    for line in text_lines:
        line_output = ""
        length = len(line)
        for col_idx, char in enumerate(line):
            ratio = col_idx / max(1, length - 1)
            r = int(0 + (255 * ratio))
            g = int(220 + (35 * ratio))
            b = 0
            line_output += f"\033[38;2;{r};{g};{b}m{char}"
        print(line_output + reset_code)

    prefix = "Made with 🧈 by "
    butter_part = "butter"
    labs_part = "labs"
    
    final_line_output = ""
    prefix_len = len(prefix)
    for col_idx, char in enumerate(prefix):
        ratio = col_idx / (prefix_len - 1)
        r = int(0 + (255 * ratio))
        g = int(220 + (35 * ratio))
        b = 0
        final_line_output += f"\033[38;2;{r};{g};{b}m{char}"
        
    butter_len = len(butter_part)
    for col_idx, char in enumerate(butter_part):
        ratio = col_idx / max(1, butter_len - 1)
        r = int(244 + (11 * ratio))
        g = int(208 + (44 * ratio))
        b = int(63 + (180 * ratio))
        final_line_output += f"\033[38;2;{r};{g};{b}m{char}"
        
    for char in labs_part:
        r, g, b = 160, 255, 160
        final_line_output += f"\033[38;2;{r};{g};{b}m{char}"
        
    print(final_line_output + reset_code)

def main():
    # If user ran the CLI without any arguments, show headphones and exit
    if len(sys.argv) == 1:
        draw_gradient_headphones()
        return

    parser = argparse.ArgumentParser(description="Track ID Utility")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    add_p = subparsers.add_parser("add")
    add_p.add_argument("file")
    add_p.add_argument("--name", required=True)

    scan_p = subparsers.add_parser("scan")
    scan_p.add_argument("file")

    subparsers.add_parser("list")

    remove_p = subparsers.add_parser("remove")
    remove_p.add_argument("--name", required=True)

    subparsers.add_parser("clear")
    subparsers.add_parser("about")
    subparsers.add_parser("help")

    args = parser.parse_args()

    command_map = {
        "add": add_track,
        "scan": scan_track,
        "list": list_tracks,
        "remove": remove_track,
        "clear": clear_db,
        "about": draw_gradient_headphones,
        "help": show_help
    }

    command_map[args.cmd](args)

if __name__ == "__main__":
    main()