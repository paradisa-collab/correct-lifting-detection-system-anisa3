import os
import re
import pandas as pd

def combine_mpu_data_from_folder(input_folder, output_folder=None):
    """
    Mencari file CSV dalam sebuah folder, menggabungkan data MPU 1 dan 2 berdasarkan timestamp
    (menggunakan pd.merge_asof), dan menyimpan hasilnya dalam format seperti 'dina.csv'.
    ID Subject dinamai ulang berdasarkan waktu pengambilan data terawal dalam setiap file,
    menggunakan format 001, 002, ..., 023.
    Melakukan pengecekan apakah file sudah dalam format gabungan dan menyesuaikan ID jika perlu.
    Menambahkan kolom pressure_hpa dan altitude_m jika tidak ada, diisi dengan pd.NA.

    Args:
        input_folder (str): Path ke folder yang berisi file-file CSV (format seperti 'alifi.csv' atau 'dina.csv').
        output_folder (str, optional): Path ke folder untuk menyimpan file hasil.
                                       Jika tidak disediakan, akan dibuat subfolder 'output_gabung_mpu' di dalam input_folder.

    Returns:
        dict: Dictionary dengan nama file input sebagai key dan DataFrame hasil sebagai value.
    """
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'output_gabung_mpu')

    os.makedirs(output_folder, exist_ok=True)

    results = {}

    # Baca semua file CSV
    csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]

    # Kelompokkan file berdasarkan statusnya: sudah digabung (dg) atau belum (blm)
    files_to_process_blm = [] # Belum digabung, format seperti 'alifi.csv'
    files_to_process_dg = {}  # Sudah digabung, format seperti 'dina.csv' {'filename': df}

    for filename in csv_files:
        filepath = os.path.join(input_folder, filename)
        try:
            df_temp = pd.read_csv(filepath, nrows=1) # Baca hanya baris pertama untuk cek kolom
            # Cek apakah file sudah dalam format 'dina.csv' (sudah digabung)
            if 'ax1' in df_temp.columns and 'ax2' in df_temp.columns and 'subject_id' in df_temp.columns:
                print(f"File {filename} terdeteksi dalam format gabungan. Akan dicek dan disesuaikan.")
                # Baca seluruh file untuk pemrosesan ulang ID
                df_full = pd.read_csv(filepath)
                if 'epoch_ms' in df_full.columns and 'subject_id' in df_full.columns:
                    df_full['epoch_ms'] = pd.to_numeric(df_full['epoch_ms'], errors='coerce')
                    df_full = df_full.dropna(subset=['epoch_ms'])
                    if not df_full.empty:
                         earliest_time = df_full['epoch_ms'].min()
                         files_to_process_dg[filename] = {'df': df_full, 'earliest_epoch': earliest_time}
                    else:
                         print(f"Peringatan: File {filename} (gabungan) tidak memiliki data epoch_ms valid setelah pembersihan.")
                else:
                     print(f"Peringatan: File {filename} (gabungan) tidak memiliki kolom epoch_ms atau subject_id. Dilewatkan.")
            # Cek apakah file dalam format 'alifi.csv' (belum digabung)
            elif 'mpu_id' in df_temp.columns:
                print(f"File {filename} terdeteksi dalam format belum digabung (mpu_id). Akan diproses.")
                files_to_process_blm.append(filename)
            else:
                 print(f"File {filename} tidak sesuai format yang dikenali (alifi/belum_digabung maupun dina/sudah_digabung). Dilewatkan.")
        except Exception as e:
             print(f"Error saat membaca header {filename}: {e}")

    # --- PROSES FILE BELUM DIGABUNG (seperti 'alifi.csv') ---
    start_times_blm = []
    raw_dataframes_blm = {}
    for filename in files_to_process_blm:
        filepath = os.path.join(input_folder, filename)
        try:
            df_temp = pd.read_csv(filepath)
            if 'epoch_ms' in df_temp.columns:
                df_temp['epoch_ms'] = pd.to_numeric(df_temp['epoch_ms'], errors='coerce')
                df_temp = df_temp.dropna(subset=['epoch_ms'])
                if not df_temp.empty:
                    earliest_time = df_temp['epoch_ms'].min()
                    start_times_blm.append({'filename': filename, 'earliest_epoch': earliest_time})
                    raw_dataframes_blm[filename] = df_temp
                else:
                     print(f"Peringatan: File {filename} tidak memiliki data epoch_ms valid setelah pembersihan.")
            else:
                 print(f"Peringatan: Kolom 'epoch_ms' tidak ditemukan di {filename}.")
        except Exception as e:
             print(f"Error saat membaca {filename}: {e}")

    # Urutkan file belum digabung berdasarkan waktu terawal
    start_times_df_blm = pd.DataFrame(start_times_blm)
    if not start_times_df_blm.empty:
        start_times_df_blm = start_times_df_blm.sort_values('earliest_epoch').reset_index(drop=True)


    # --- GABUNGKAN INFORMASI WAKTU DARI KEDUA TIPE FILE ---
    all_start_times = []
    # Tambahkan info dari file yang sudah digabung
    for fn, data in files_to_process_dg.items():
        all_start_times.append({'filename': fn, 'earliest_epoch': data['earliest_epoch'], 'type': 'dg'})
    # Tambahkan info dari file yang belum digabung
    for _, row in start_times_df_blm.iterrows():
        all_start_times.append({'filename': row['filename'], 'earliest_epoch': row['earliest_epoch'], 'type': 'blm'})

    # Buat DataFrame dan urutkan semua file berdasarkan waktu terawal
    all_start_times_df = pd.DataFrame(all_start_times)
    if all_start_times_df.empty:
         print("Tidak ditemukan file CSV valid dengan data epoch_ms.")
         return results
    all_start_times_df = all_start_times_df.sort_values('earliest_epoch').reset_index(drop=True)

    # --- HITUNG TOTAL FILE YANG AKAN DIPROSES ---
    total_files_to_process = len(all_start_times_df)
    print(f"Total file valid yang akan diproses (dan diberi ID): {total_files_to_process}")

    # --- PROSES SEMUA FILE SESUAI URUTAN WAKTU ---
    # Gunakan indeks loop sebagai penentu ID berdasarkan urutan
    for idx, (_, row) in enumerate(all_start_times_df.iterrows()):
        filename = row['filename']
        file_type = row.get('type', 'unknown') # Harusnya 'dg' atau 'blm'

        # --- Tentukan ID Subjek Baru ---
        # IDX dimulai dari 0, jadi tambah 1, lalu format menjadi 3 digit
        subject_id_new = f'{idx + 1:03d}'

        if file_type == 'blm':
            # --- PROSES FILE BELUM DIGABUNG (seperti 'alifi.csv') ---
            df = raw_dataframes_blm[filename]
            try:
                # Asumsikan struktur file input mirip dengan 'alifi.csv'
                if 'mpu_id' not in df.columns:
                    print(f"Peringatan: Kolom 'mpu_id' tidak ditemukan di {filename} saat diproses ulang. Lewati.")
                    continue

                df_mpu1 = df[df['mpu_id'] == 1].copy()
                df_mpu2 = df[df['mpu_id'] == 2].copy()

                if df_mpu1.empty or df_mpu2.empty:
                    print(f"Peringatan: Data untuk mpu_id 1 atau 2 kosong di {filename}. Lewati.")
                    continue

                # Rename kolom sensor dari masing-masing DataFrame
                df_mpu1_renamed = df_mpu1.rename(columns={
                    'ax': 'ax1', 'ay': 'ay1', 'az': 'az1',
                    'gx': 'gx1', 'gy': 'gy1', 'gz': 'gz1'
                })

                df_mpu2_renamed = df_mpu2.rename(columns={
                    'ax': 'ax2', 'ay': 'ay2', 'az': 'az2',
                    'gx': 'gx2', 'gy': 'gy2', 'gz': 'gz2'
                })

                # Gabungkan berdasarkan timestamp (epoch_ms) menggunakan merge_asof
                df_mpu1_sorted = df_mpu1_renamed.sort_values('epoch_ms')
                df_mpu2_sorted = df_mpu2_renamed.sort_values('epoch_ms')

                df_merged = pd.merge_asof(
                    df_mpu1_sorted,
                    df_mpu2_sorted[['epoch_ms', 'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2']],
                    on='epoch_ms',
                    direction='nearest'
                )

                # Ekstrak informasi nama dari nama file
                base_name = os.path.splitext(filename)[0]
                name_parts = base_name.split('_')
                if len(name_parts) > 0:
                    nama_pengguna = name_parts[-1]
                else:
                    nama_pengguna = base_name

                # Tambahkan kolom subject_id, nama, date, time
                df_final = df_merged.copy()
                df_final['subject_id'] = subject_id_new # Gunakan ID baru
                df_final['nama'] = nama_pengguna
                df_final['date'] = pd.to_datetime(df_final['datetime']).dt.date
                df_final['time'] = pd.to_datetime(df_final['datetime']).dt.time

                # Urutan kolom sesuai contoh 'dina.csv' - tambahkan BME
                ordered_columns = [
                    'subject_id', 'nama', 'date', 'time', 'epoch_ms', 'ts_millis',
                    'ax1', 'ay1', 'az1', 'gx1', 'gy1', 'gz1',
                    'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2',
                    'pressure_hpa', 'altitude_m', # <-- TAMBAHKAN: Kolom BME disini
                    'label'
                ]

                # Pastikan semua kolom yang diharapkan ada
                for col in ordered_columns:
                    if col not in df_final.columns:
                        if col in ['ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2']:
                            df_final[col] = pd.NA
                        elif col in ['pressure_hpa', 'altitude_m']: # <-- TAMBAHKAN: Handle BME
                            df_final[col] = pd.NA
                        elif col == 'label' and 'label' in df.columns:
                            df_final[col] = df.set_index('epoch_ms').loc[df_final['epoch_ms']]['label'].values
                        else:
                            df_final[col] = pd.NA

                df_final = df_final.reindex(columns=ordered_columns)

                # Buat nama file output
                output_filename = f"gabungan_mpu_{filename}"
                output_file_path = os.path.join(output_folder, output_filename)

                # Simpan hasil ke file CSV
                df_final.to_csv(output_file_path, index=False)
                print(f"File gabungan untuk {filename} (lama: ?, baru: {subject_id_new}) disimpan sebagai {output_file_path}")

                results[filename] = df_final

            except Exception as e:
                print(f"Error saat memproses file belum digabung {filename}: {e}")

        elif file_type == 'dg':
            # --- PROSES FILE SUDAH DIGABUNG (seperti 'dina.csv') ---
            df_original = files_to_process_dg[filename]['df']
            try:
                # Ekstrak informasi nama dari nama file (jika nama kolom tidak di-update)
                base_name = os.path.splitext(filename)[0]
                if 'nama' not in df_original.columns or df_original['nama'].iloc[0] is None or pd.isna(df_original['nama'].iloc[0]):
                    name_parts = base_name.split('_')
                    if len(name_parts) > 0:
                        nama_pengguna = name_parts[-1]
                    else:
                        nama_pengguna = base_name
                else:
                    nama_pengguna = df_original['nama'].iloc[0]

                # Update subject_id dan nama (jika diperlukan) di seluruh dataframe
                df_final = df_original.copy()
                df_final['subject_id'] = subject_id_new # Gunakan ID baru
                if 'nama' not in df_final.columns:
                    df_final['nama'] = nama_pengguna
                else:
                    pass # Biarkan nama sesuai data aslinya jika sudah ada

                # Urutan kolom sesuai contoh 'dina.csv' - tambahkan BME jika tidak ada
                expected_ordered_columns = [
                    'subject_id', 'nama', 'date', 'time', 'epoch_ms', 'ts_millis',
                    'ax1', 'ay1', 'az1', 'gx1', 'gy1', 'gz1',
                    'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2',
                    'pressure_hpa', 'altitude_m',
                    'label'
                ]
                # Pastikan semua kolom yang diharapkan ada, tambahkan jika tidak
                for col in expected_ordered_columns:
                    if col not in df_final.columns:
                        df_final[col] = pd.NA
                # Reindex sesuai urutan yang diharapkan
                df_final = df_final.reindex(columns=expected_ordered_columns)

                # Buat nama file output
                output_filename = f"gabungan_mpu_{filename}"
                output_file_path = os.path.join(output_folder, output_filename)

                # Simpan hasil ke file CSV
                df_final.to_csv(output_file_path, index=False)
                print(f"File gabungan untuk {filename} (lama: {df_original['subject_id'].iloc[0]}, baru: {subject_id_new}) disimpan sebagai {output_file_path}")

                results[filename] = df_final

            except Exception as e:
                print(f"Error saat memproses file sudah digabung {filename}: {e}")
        else:
            print(f"File {filename} memiliki tipe tidak diketahui saat pemrosesan akhir.")

    return results


# --- Konfigurasi ---
input_directory = r'D:\Semester_7\Skripsian\correct-lifting-detection-system-anisa\tga_anisa'
output_directory = r'D:\Semester_7\Skripsian\correct-lifting-detection-system-anisa\dataLabel'

# --- Eksekusi ---
combined_dataframes = combine_mpu_data_from_folder(input_directory, output_directory)
