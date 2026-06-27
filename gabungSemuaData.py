import pandas as pd
import glob
import os

# ========== SETTING ==========
folder_path = "dataLabel"   # sesuaikan dengan folder Anda
output_file = "merged_sorted.csv"
# ==============================

# Cari semua file CSV
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

if not all_files:
    print(f"Tidak ada file CSV ditemukan di folder: {folder_path}")
    exit()

df_list = []
for file in all_files:
    try:
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)
        print(f"Berhasil membaca: {file} ({len(df_temp)} baris)")
    except Exception as e:
        print(f"Gagal membaca {file}: {e}")

combined_df = pd.concat(df_list, ignore_index=True)
print(f"\nTotal baris sebelum sorting: {len(combined_df)}")

# --- HAPUS KOLOM pressure_hpa dan altitude_m ---
# (gunakan errors='ignore' jika salah satu kolom tidak ada)
combined_df.drop(columns=['pressure_hpa', 'altitude_m'], errors='ignore', inplace=True)
print("Kolom 'pressure_hpa' dan 'altitude_m' telah dihapus.")

# --- KONVERSI WAKTU (format campuran) ---
combined_df['datetime'] = pd.to_datetime(
    combined_df['date'] + ' ' + combined_df['time'],
    format='mixed',          # untuk pandas >=1.3
    errors='coerce'
)

# Fallback untuk pandas lama (jika format='mixed' error)
if combined_df['datetime'].isna().all():
    combined_df['datetime'] = pd.to_datetime(
        combined_df['date'] + ' ' + combined_df['time'],
        infer_datetime_format=True,
        errors='coerce'
    )

# Cek kegagalan konversi
if combined_df['datetime'].isna().any():
    print(f"Peringatan: {combined_df['datetime'].isna().sum()} baris tidak bisa dikonversi ke datetime.")

# Urutkan
combined_df.sort_values('datetime', inplace=True, na_position='last')
combined_df.reset_index(drop=True, inplace=True)

# (Opsional) hapus kolom datetime jika tidak diperlukan
# combined_df.drop('datetime', axis=1, inplace=True)

# Simpan
combined_df.to_csv(output_file, index=False)
print(f"File hasil disimpan: {output_file} ({len(combined_df)} baris, kolom pressure_hpa & altitude_m sudah dihapus, terurut waktu)")