import re

ASPECT_KEYWORDS = {
    "Jadwal Ujian": [
        "jadwal ujian", "ujian", "jadwal", "ujian akhir", "uas", "uts",
        "ujian tengah semester", "ujian akhir semester",
    ],
    "Hasil Studi": [
        "hasil studi", "hasil nilai", "nilai", "nilai mata kuliah", "khs",
        "kartu hasil studi", "hasil ujian", "nilai keluar",
    ],
    "Transkrip": [
        "transkrip", "transkrip nilai", "transkrip akademik",
        "cetak transkrip", "dokumen transkrip",
    ],
    "Pembayaran": [
        "pembayaran", "bayar", "biaya kuliah", "ukt", "tagihan",
        "pembayaran kuliah", "pembayaran semester", "keuangan",
    ],
    "KTM": [
        "ktm", "kartu tanda mahasiswa", "kartu mahasiswa",
        "cetak ktm", "kartu identitas mahasiswa",
    ],
    "Perpustakaan": [
        "perpustakaan", "buku", "peminjaman buku", "pengembalian buku",
        "koleksi buku", "e-book", "ruang baca", "pustaka",
    ],
    "Kalender Akademik": [
        "kalender akademik", "kalender", "jadwal akademik", "semester",
        "tahun akademik", "awal semester", "akhir semester", "libur akademik",
    ],
    "Roadmap": [
        "roadmap", "peta studi", "peta akademik", "rencana studi",
        "alur studi", "mata kuliah", "mata kuliah wajib",
        "mata kuliah pilihan", "kurikulum",
    ],
    "Presensi": [
        "presensi", "absensi", "absen", "kehadiran", "hadir",
        "tidak hadir", "daftar hadir", "absensi mahasiswa",
    ],
}

ASPECTS = list(ASPECT_KEYWORDS.keys())


def _contains_keyword(text, keyword):
    # Phrase/word boundary matching avoids matching 'absen' inside unrelated words.
    pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
    return bool(re.search(pattern, text))


def determine_aspects(text):
    if text is None:
        return ["Lainnya"]

    text = str(text).lower().strip()
    if not text:
        return ["Lainnya"]

    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(_contains_keyword(text, keyword) for keyword in keywords):
            found.append(aspect)

    return found if found else ["Lainnya"]
