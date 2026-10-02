import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.generate_data import generate_students
from shiftproof.config import DATA_DIR
from shiftproof.training import train_all


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / "students.csv"
    df = generate_students(n=12000)
    df.to_csv(data_path, index=False)
    print(f"Generated {len(df):,} synthetic student records.")
    summary = train_all(data_path)
    print("Training complete.")
    print(summary)


if __name__ == "__main__":
    main()