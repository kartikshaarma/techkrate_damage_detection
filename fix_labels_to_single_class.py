import pathlib

dataset_path = pathlib.Path("vehicle_damage_yolo")

for split in ["train", "valid"]:
    label_dir = dataset_path / split / "labels"
    for txt_file in label_dir.glob("*.txt"):
        lines = txt_file.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                parts[0] = "0"
                new_lines.append(" ".join(parts))
        txt_file.write_text("\n".join(new_lines))
        print(f"Fixed {txt_file}")
